import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from src.solver import BaseSolver
import math
from src.option import default_hparas

from src.optim import Optimizer
from src.util import human_format, cal_er, feat_to_fig
from bin.asr.asr import ASR
from bin.asr.asr_data import load_dataset

from src.transformer.label_smoothing_loss import LabelSmoothingLoss

import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP

from src.transformer.mask import target_mask
from src.transformer.nets_utils import make_non_pad_mask

import torch.cuda.nvtx as nvtx


class SolverASR():
    ''' Solver for training'''

    def __init__(self, gpu_rank, world_size, rank, config, paras, mode, log):
        # super().__init__(config, paras, mode)
        self.config = config
        self.paras = paras
        self.mode = mode
        self.log = log
        self.emb_decoder = None
        for k, v in default_hparas.items():
            setattr(self, k, v)

        # Logger settings
        self.best_wer = {'att': 3.0, 'ctc': 3.0}
        # Curriculum learning affects data loader
        self.curriculum = self.config['hparas']['curriculum']

        self.gpu_rank = gpu_rank
        self.world_size = world_size
        self.rank = rank

    def fetch_data(self, data):
        ''' Move data to device and compute text seq. length'''
        _, feat, feat_len, txt = data
        if self.paras.gpu and self.mode == 'train':
            feat = feat.cuda(self.gpu_rank)
            feat_len = feat_len.cuda(self.gpu_rank)
            txt = txt.cuda(self.gpu_rank)
            txt_len = torch.sum(txt != 0, dim=-1)
        else:
            feat = feat.to(self.device)
            feat_len = feat_len.to(self.device)
            txt = txt.to(self.device)
            txt_len = torch.sum(txt != 0, dim=-1)

        fetched_data = {'feat': feat,
                        'feat_len': feat_len,
                        'txt': txt,
                        'txt_len': txt_len}
        return fetched_data

    def load_data(self):
        ''' Load data for training/validation, store tokenizer and input/output shape'''
        self.tr_sampler, self.tr_set, self.dv_set, self.feat_dim, self.vocab_size, self.tokenizer, msg = \
            load_dataset(self.gpu_rank, self.world_size, self.rank, self.paras.njobs, self.paras.gpu, self.paras.pin_memory,
                         self.curriculum > 0, **self.config['data'])
        self.verbose(msg)

    def set_model(self):
        ''' Setup ASR model and optimizer '''
        init_adadelta = self.config['hparas']['optimizer'] == 'Adadelta'

        # Model
        if self.paras.gpu and self.mode == 'train':
            torch.cuda.set_device(self.gpu_rank)
            self.model = ASR(self.feat_dim, self.vocab_size, init_adadelta, **
                             self.config['model']).cuda(self.gpu_rank)
        else:
            self.model = ASR(self.feat_dim, self.vocab_size, init_adadelta, **
                             self.config['model']).to(self.device)

        self.verbose(self.model.create_msg())
        model_paras = [{'params': self.model.parameters()}]

        # Losses
        # self.seq_loss = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.seq_loss = LabelSmoothingLoss(self.vocab_size, 0, self.config['model']['lsm_weight'])
        # Note: zero_infinity=False is unstable?
        self.ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=False)

        # Plug-ins
        self.emb_fuse = False
        self.emb_reg = ('emb' in self.config) and (
            self.config['emb']['enable'])
        if self.emb_reg:
            from src.plugin import EmbeddingRegularizer
            self.emb_decoder = EmbeddingRegularizer(
                self.tokenizer, self.model.dec_dim, **self.config['emb']).to(self.device)
            model_paras.append({'params': self.emb_decoder.parameters()})
            self.emb_fuse = self.emb_decoder.apply_fuse
            if self.emb_fuse:
                self.seq_loss = torch.nn.NLLLoss(ignore_index=0)
            self.verbose(self.emb_decoder.create_msg())

        # Optimizer
        self.optimizer = Optimizer(model_paras, **self.config['hparas'])
        self.verbose(self.optimizer.create_msg())

        # Enable AMP if needed
        self.enable_apex()
        # DistributedDataParallel
        if self.paras.gpu and self.mode == 'train':
            if not self.no_amp:
                self.model = DDP(self.model) 
            else:
                self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.gpu_rank]) 

        # Automatically load pre-trained model if self.paras.load is given
        self.load_ckpt()

        # ToDo: other training methods

    def exec(self):
        ''' Training End-to-end ASR system '''
        self.verbose('Total training steps {}.'.format(
            human_format(self.max_step)))
        ctc_loss, att_loss, emb_loss = None, None, None
        n_epochs = 0
        self.timer.set()

        while self.step < self.max_step:
            # Renew dataloader to enable random sampling
            if self.curriculum > 0 and n_epochs == self.curriculum:
                self.verbose(
                    'Curriculum learning ends after {} epochs, starting random sampling.'.format(n_epochs))
                self.tr_sampler, self.tr_set, _, _, _, _, _ = \
                    load_dataset(self.gpu_rank, self.world_size, self.rank, self.paras.njobs, self.paras.gpu, 
                                 self.paras.pin_memory, False, **self.config['data'])
            for data in self.tr_set:
                # Pre-step : update tf_rate/lr_rate and do zero_grad
                tf_rate, cur_lr = self.optimizer.pre_step(self.step)
                total_loss = 0

                # Fetch data
                feat, feat_len, txt, txt_len = self.fetch_data(data)
                self.timer.cnt('rd')

                # Forward model
                # Note: txt should NOT start w/ <sos>
                # print(feat.shape)
                # print(feat_len)
                # print(txt)
                # print(txt.shape)
                # print(txt_len)
                # exit()
                ctc_output, encode_len, att_output, att_align, dec_state = \
                    self.model(feat, feat_len, max(txt_len), tf_rate=tf_rate,
                               teacher=txt, get_dec_state=self.emb_reg, txt_len=txt_len)

                # Plugins
                if self.emb_reg:
                    emb_loss, fuse_output = self.emb_decoder(
                        dec_state, att_output, label=txt)
                    total_loss += self.emb_decoder.weight*emb_loss

                # Compute all objectives
                if ctc_output is not None:
                    if self.paras.cudnn_ctc:
                        ctc_loss = self.ctc_loss(ctc_output.transpose(0, 1),
                                                 txt.to_sparse().values().to(device='cpu', dtype=torch.int32),
                                                 [ctc_output.shape[1]] *
                                                 len(ctc_output),
                                                 txt_len.cpu().tolist())
                    else:
                        ctc_loss = self.ctc_loss(ctc_output.transpose(
                            0, 1), txt, encode_len, txt_len)
                    if self.paras.gpu:
                        total_loss += ctc_loss*self.model.module.ctc_weight
                    else:
                        total_loss += ctc_loss*self.model.ctc_weight

                if att_output is not None:
                    b, t, _ = att_output.shape
                    att_output = fuse_output if self.emb_fuse else att_output
                    att_loss = self.seq_loss(att_output, txt)
                    if self.paras.gpu:
                        total_loss += att_loss*(1-self.model.module.ctc_weight)
                    else:
                        total_loss += att_loss*(1-self.model.ctc_weight)

                self.timer.cnt('fw')

                # Backprop
                grad_norm = self.backward(total_loss)
                self.step += 1

                # Logger
                if (self.rank == 0) and ((self.step == 1) or (self.step % self.PROGRESS_STEP == 0)):
                    self.progress('Tr stat | Loss - {:.2f} | Grad. Norm - {:.2f} | {}'
                                  .format(total_loss.cpu().item(), grad_norm, self.timer.show()))
                    self.write_log('max_memory_allocated', {'memory': torch.cuda.max_memory_allocated(att_loss.device)})
                    self.write_log('max_memory_reserved', {'memory': torch.cuda.max_memory_reserved(att_loss.device)})
                    self.write_log('lr', {'lr': cur_lr})
                    self.write_log(
                        'loss', {'tr_att': att_loss, 'tr_ctc': ctc_loss})
                    self.write_log('emb_loss', {'tr': emb_loss})
                    self.write_log('wer', {'tr_att': cal_er(self.tokenizer, att_output, txt),
                                           'tr_ctc': cal_er(self.tokenizer, ctc_output, txt, ctc=True)})
                    if self.emb_fuse:
                        if self.emb_decoder.fuse_learnable:
                            self.write_log('fuse_lambda', {
                                           'emb': self.emb_decoder.get_weight()})
                        self.write_log(
                            'fuse_temp', {'temp': self.emb_decoder.get_temp()})

                # Validation
                if (self.rank == 0) and ((self.step == 1) or (self.step % self.valid_step == 0)):
                    self.validate()

                # Make sure other processes load the model after process 0 saves it.
                dist.barrier()

                # End of step
                # https://github.com/pytorch/pytorch/issues/13246#issuecomment-529185354
                torch.cuda.empty_cache()
                self.timer.set()
                if self.step > self.max_step:
                    break
            n_epochs += 1
        self.log.close()

    def validate(self):
        # Eval mode
        self.model.eval()
        if self.emb_decoder is not None:
            self.emb_decoder.eval()
        dev_wer = {'att': [], 'ctc': []}

        for i, data in enumerate(self.dv_set):
            self.progress('Valid step - {}/{}'.format(i+1, len(self.dv_set)))
            # Fetch data
            feat, feat_len, txt, txt_len = self.fetch_data(data)

            # Forward model
            with torch.no_grad():
                ctc_output, encode_len, att_output, att_align, dec_state = \
                    self.model(feat, feat_len, int(max(txt_len)*self.DEV_STEP_RATIO),
                               emb_decoder=self.emb_decoder)

            dev_wer['att'].append(cal_er(self.tokenizer, att_output, txt))
            dev_wer['ctc'].append(cal_er(self.tokenizer, ctc_output, txt, ctc=True))

            # Show some example on tensorboard
            if i == len(self.dv_set)//2:
                for i in range(min(len(txt), self.DEV_N_EXAMPLE)):
                    if self.step == 1:
                        self.write_log('true_text{}'.format(
                            i), self.tokenizer.decode(txt[i].tolist()))
                    if att_output is not None:
                        try:
                            self.write_log('att_align{}'.format(i), feat_to_fig(
                                att_align[i, 0, :, :].cpu().detach()))
                        except:
                            pass
                        self.write_log('att_text{}'.format(i), self.tokenizer.decode(
                            att_output[i].argmax(dim=-1).tolist()))
                    if ctc_output is not None:
                        self.write_log('ctc_text{}'.format(i), self.tokenizer.decode(ctc_output[i].argmax(dim=-1).tolist(),
                                                                                     ignore_repeat=True))

        # Ckpt if performance improves
        for task in ['att', 'ctc']:
            dev_wer[task] = sum(dev_wer[task])/len(dev_wer[task])
            if dev_wer[task] < self.best_wer[task]:
                self.best_wer[task] = dev_wer[task]
                self.save_checkpoint('best_{}.pth'.format(task), 'wer', dev_wer[task])
            self.write_log('wer', {'dv_'+task: dev_wer[task]})
        self.save_checkpoint('latest.pth', 'wer', dev_wer['att'], show_msg=False)

        # Resume training
        self.model.train()
        if self.emb_decoder is not None:
            self.emb_decoder.train()

    
    #####################
    # For GeneralSolver #
    #####################

    def transform_data(self, data, model_sample_rate, audio_transform):
        files, text, batch_waveforms, wave_lens, sample_rate = data
        audio_len = torch.stack([torch.tensor(wave_len).long().cuda(self.gpu_rank) for wave_len in wave_lens])
        assert sample_rate == model_sample_rate

        # Read batch
        # audio_feat, audio_len = [], []
        lengths = []
        waveforms = []
        for wave in batch_waveforms:
            lengths.append(len(wave[0]))
            nvtx.range_push('To cuda')
            waveforms.append(wave[0].cuda(self.gpu_rank))
            nvtx.range_pop()
            
        with torch.no_grad():
            nvtx.range_push('Pad sequence')
            padded_waveforms = pad_sequence(waveforms,batch_first=True)
            nvtx.range_pop()
            audio_feat, audio_len = audio_transform((padded_waveforms, audio_len))
            audio_feat = [audio_feat[i] for i in range(len(audio_feat))]
            # for waveform in waveforms:
            #     if self.gpu_rank is not None:
            #         # Specify some functions in SpecAug-> FP32
            #         # with torch.cuda.amp.autocast():
            #         feat = audio_transform(waveform.cuda(self.gpu_rank), torchs.stack(feat_lengths).cuda(self.gpu_rank))
            #     else:
            #         feat = audio_transform(waveform)
            #     audio_feat.append(feat)
            #     audio_len.append(len(feat))
        # Descending audio length within each batch
        audio_len, files, audio_feat, text = zip(*[(feat_len, f_name, feat, txt) \
                for feat_len, f_name, feat, txt in sorted(zip(audio_len, files, audio_feat, text), reverse=True, key=lambda x:x[0])])
        # Zero-padding
        nvtx.range_push('Pad sequence')
        audio_feat = pad_sequence(audio_feat, batch_first=True)
        text = pad_sequence(text, batch_first=True)
        nvtx.range_pop()
        audio_len = torch.LongTensor(audio_len)

        return files, audio_feat, audio_len, text.to(audio_feat.device)

    def forward(self, asr_inputs, model_sample_rate, audio_transform, audio_encoder, audio_decoder, text_decoder, prosody_predictor, output_model, attention, vocab_size, 
                lsm_weight, ctc_weight, enable_ctc, enable_att, 
                ctc_layer, pre_embed, embed_drop, if_reconstruction=True):
        asr_outputs = None
        nvtx.range_push('ASR forward')
        # Transform data
        nvtx.range_push('Transform data')
        files, asr_feat, asr_feat_len, asr_txt = self.transform_data(asr_inputs['asr_data'], model_sample_rate, audio_transform)
        nvtx.range_pop()

        nvtx.range_push('Sum asr_txt_len')
        asr_txt_len = torch.sum(asr_txt != 0, dim=-1)
        nvtx.range_pop()

        if asr_inputs['mode'] == 'train':
            teacher = asr_txt
            decode_step = max(asr_txt_len)
        else:
            teacher = None
            decode_step = int(max(asr_txt_len)*self.DEV_STEP_RATIO)

        # Init
        bs = asr_feat.shape[0]
        ctc_output, att_output, att_seq = None, None, None
        dec_state = [] if asr_inputs['get_dec_state'] else None

        # Encode
        nvtx.range_push('Audio encoder forward')
        encode_len, encode_feature_content, encode_mask_content, \
            encode_feature_speaker, encode_mask_speaker, pooled_encode_feature_speaker = \
            audio_encoder(asr_feat, asr_feat_len)
        nvtx.range_pop()
        # print(encode_feature_content.type())

        # Reconstruct
        if if_reconstruction:
            _, predicted_prosody, _ = prosody_predictor(encode_feature_content + pooled_encode_feature_speaker.unsqueeze(1), 
                    encode_len.to(encode_feature_content.device))
            '''
            predicted_prosody = None
            '''
            _, reconstructed, _ = audio_decoder(encode_len.to(encode_feature_content.device), encode_feature_content, encode_mask_content,
                                          encode_feature_speaker, encode_mask_speaker)
            reconstructed = output_model(reconstructed)
        else:
            reconstructed = None

        # CTC based decoding
        if enable_ctc:
            nvtx.range_push('CTC decode')
            ctc_output = F.log_softmax(ctc_layer(encode_feature_content), dim=-1)
            nvtx.range_pop()

        # Attention based decoding
        if enable_att:
            nvtx.range_push('ATT decode')
            # Init (init char = <SOS>, reset all rnn state and cell)
            nvtx.range_push('Text decoder init')
            text_decoder.init_state(bs)
            if text_decoder.module in ['LSTM', 'GRU']:
                attention.reset_mem()
            last_char = pre_embed(torch.zeros(
                (bs), dtype=torch.long, device=encode_feature_content.device))
            nvtx.range_pop()
            att_seq, output_seq = [], []

            # Preprocess data for teacher forcing
            if teacher is not None:
                nvtx.range_push('Teacher embed')
                asr_teacher = embed_drop(pre_embed(teacher))
                nvtx.range_pop()

            # Decode
            if text_decoder.module == 'Transformer' and teacher is not None:
                # Pad sos
                nvtx.range_push('Txt padding and masking')
                txt_input = torch.cat((torch.zeros(last_char.shape, 
                                                   device=teacher.device).unsqueeze(1), asr_teacher[:, :-1]), 1)
                txt_mask = make_non_pad_mask((asr_txt_len).tolist()).to(teacher.device)
                txt_mask = target_mask(txt_mask, 0) 
                nvtx.range_pop()

                nvtx.range_push('Text decoder forward')
                dec_state = text_decoder(encode_feature_content, encode_mask_content, txt_input, txt_mask)
                nvtx.range_pop()

                nvtx.range_push('Text decoder decode')
                for t in range(decode_step):
                    cur_char = text_decoder.char_trans(text_decoder.final_dropout(dec_state[:, t]))
                    output_seq.append(cur_char)
                att_output = torch.stack(output_seq, dim=1)  # BxTxV
                nvtx.range_pop()
            else:
                nvtx.range_push('Text decoder no-teacher')
                if text_decoder.module == 'Transformer':
                    decoded = torch.zeros((last_char.shape[0], decode_step+1, last_char.shape[1]), 
                                          device=encode_feature_content.device)
                for t in range(decode_step):
                    if text_decoder.module in ['LSTM', 'GRU']:
                        # Attend (inputs current state of first layer, encoded features)
                        attn, context = attention(
                            text_decoder.get_query(), encode_feature_content, encode_len)
                        # Decode (inputs context + embedded last character)
                        text_decoder_input = torch.cat([last_char, context], dim=-1)
                        cur_char, d_state = text_decoder(text_decoder_input)
                    elif text_decoder.module == 'Transformer':
                        txt_input = decoded[:, :t+1]
                        txt_mask = make_non_pad_mask([t+1]*decoded.shape[0]).to(txt_input.device)
                        txt_mask = target_mask(txt_mask, 0) 
                        d_state = text_decoder(encode_feature_content, encode_mask_content, txt_input, txt_mask)[:, -1]
                        cur_char = text_decoder.char_trans(text_decoder.final_dropout(d_state))

                    # Prepare output as input of next step
                    if (teacher is not None):
                        # Training stage
                        if (asr_inputs['tf_rate'] == 1) or (torch.rand(1).item() <= asr_inputs['tf_rate']):
                            # teacher forcing
                            last_char = teacher[:, t, :]
                        else:
                            # self-sampling (replace by argmax may be another choice)
                            with torch.no_grad():
                                if (asr_inputs['emb_decoder'] is not None) and asr_inputs['emb_decoder'].apply_fuse:
                                    _, cur_prob = asr_inputs['emb_decoder'](
                                        d_state, cur_char, return_loss=False)
                                else:
                                    cur_prob = cur_char.softmax(dim=-1)
                                sampled_char = Categorical(cur_prob).sample()
                            last_char = embed_drop(
                                pre_embed(sampled_char))
                    else:
                        # Inference stage
                        if (asr_inputs['emb_decoder'] is not None) and asr_inputs['emb_decoder'].apply_fuse:
                            _, cur_char = asr_inputs['emb_decoder'](
                                d_state, cur_char, return_loss=False)
                        # argmax for inference
                        last_char = pre_embed(torch.argmax(cur_char, dim=-1))
                        if text_decoder.module == 'Transformer':
                            decoded[:, t+1] = last_char


                    # save output of each step
                    output_seq.append(cur_char)
                    if text_decoder.module in ['LSTM', 'GRU']:
                        att_seq.append(attn)
                    if asr_inputs['get_dec_state']:
                        dec_state.append(d_state)
                nvtx.range_pop()

                nvtx.range_push('Output stack')
                att_output = torch.stack(output_seq, dim=1)  # BxTxV
                if text_decoder.module in ['LSTM', 'GRU']:
                    att_seq = torch.stack(att_seq, dim=2)       # BxNxDtxT
                else:
                    att_seq = None
                if asr_inputs['get_dec_state']:
                    dec_state = torch.stack(dec_state, dim=1)
                nvtx.range_pop()
            nvtx.range_pop()
        nvtx.range_pop()

        spk_list = []
        for f in files:
            spk_list.append(f.split('-')[0])

        asr_outputs = {'ctc_output': ctc_output, 
                       'encode_len': encode_len, 
                       'att_output': att_output, 
                       'att_align': att_seq, 
                       'dec_state': dec_state,
                       'txt': asr_txt,
                       'txt_len': asr_txt_len,
                       'asr_feat': asr_feat,
                       'asr_feat_len': asr_feat_len,
                       'reconstructed': reconstructed,
                       'encode_feature_speaker': encode_feature_speaker,
                       'predicted_prosody': predicted_prosody,
                       'spk_emb': pooled_encode_feature_speaker.cpu().detach().numpy().tolist(),
                       'spk': spk_list}
        lm_outputs = {'pred': None,
                      'hidden': None}

        return asr_outputs, lm_outputs

    ### Set model

    def _set_asr_model(self, general_solver, model_paras):
        # self.seq_loss = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.seq_loss = LabelSmoothingLoss(self.vocab_size, 0, self.config['model']['lsm_weight'])
        # Note: zero_infinity=False is unstable?
        self.ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=False)

        # Plug-ins
        self.emb_fuse = False
        self.emb_reg = ('emb' in self.config) and (
            self.config['emb']['enable'])
        if self.emb_reg:
            from src.plugin import EmbeddingRegularizer
            self.emb_decoder = EmbeddingRegularizer(
                self.tokenizer, general_solver.model.dec_dim, **self.config['emb']).to(general_solver.device)
            model_paras.append({'params': self.emb_decoder.parameters()})
            self.emb_fuse = self.emb_decoder.apply_fuse
            if self.emb_fuse:
                self.seq_loss = torch.nn.NLLLoss(ignore_index=0)
            self.verbose(self.emb_decoder.create_msg())
        return model_paras

    ### Exec

    def _renew_asr_dataset(self, general_solver, n_epochs):
        # Renew dataloader to enable random sampling
        if self.curriculum > 0 and n_epochs == self.curriculum:
            self.verbose(
                'Curriculum learning ends after {} epochs, starting random sampling.'.format(n_epochs))
            self.tr_sampler, self.tr_set, _, _, _, _, _ = \
                load_dataset(self.gpu_rank, self.world_size, self.rank, self.paras.njobs, self.paras.gpu, 
                             self.paras.pin_memory, False, **self.config['data'])

    def _exec_asr(self, asr_exec_inputs, ctc_weight):
        # Note: txt should NOT start w/ <sos>
        total_loss = 0

        # Plugins
        if self.emb_reg:
            emb_loss, fuse_output = self.emb_decoder(
                asr_exec_inputs['dec_state'], asr_exec_inputs['att_output'], label=asr_exec_inputs['txt'])
            total_loss += self.emb_decoder.weight*emb_loss
        else:
            emb_loss = None

        # Compute all objectives
        if asr_exec_inputs['ctc_output'] is not None:
            if self.paras.cudnn_ctc:
                ctc_loss = self.ctc_loss(asr_exec_inputs['ctc_output'].transpose(0, 1),
                                         asr_exec_inputs['txt'].to_sparse().values().to(device='cpu', dtype=torch.int32),
                                         [asr_exec_inputs['ctc_output'].shape[1]] *
                                         len(asr_exec_inputs['ctc_output']),
                                         asr_exec_inputs['txt_len'].cpu().tolist())
            else:
                ctc_loss = self.ctc_loss(asr_exec_inputs['ctc_output'].transpose(0, 1), 
                                         asr_exec_inputs['txt'], asr_exec_inputs['encode_len'], asr_exec_inputs['txt_len'])
            total_loss += ctc_loss*ctc_weight

        if asr_exec_inputs['att_output'] is not None:
            b, t, _ = asr_exec_inputs['att_output'].shape
            att_output = fuse_output if self.emb_fuse else asr_exec_inputs['att_output']
            att_loss = self.seq_loss(att_output, asr_exec_inputs['txt'])
            total_loss += att_loss*(1-ctc_weight)

        if asr_exec_inputs['reconstructed'] is not None:
            feat_mask = make_non_pad_mask(asr_exec_inputs['asr_feat_len']).to(asr_exec_inputs['asr_feat'].device)[:, :asr_exec_inputs['reconstructed'].shape[1]]
            whole_feat_mask = feat_mask.unsqueeze(2).expand(-1, -1, asr_exec_inputs['reconstructed'].shape[2])
            reconstructed_loss = F.mse_loss(asr_exec_inputs['reconstructed']*whole_feat_mask, 
                                            asr_exec_inputs['asr_feat'][:, :whole_feat_mask.shape[1], :whole_feat_mask.shape[2]]*whole_feat_mask)

            whole_prosody_mask = feat_mask.unsqueeze(2).expand(-1, -1, asr_exec_inputs['predicted_prosody'].shape[2])
            prosody_loss = F.mse_loss(asr_exec_inputs['predicted_prosody']*whole_prosody_mask, 
                                            asr_exec_inputs['encode_feature_speaker'][:, :whole_prosody_mask.shape[1], 
                                                :whole_prosody_mask.shape[2]]*whole_prosody_mask)
            '''
            prosody_loss = None
            '''

            total_loss += reconstructed_loss + prosody_loss
        else:
            reconstructed_loss = None

        asr_exec_outputs = {'att_loss': att_loss,
                            'ctc_loss': ctc_loss,
                            'emb_loss': emb_loss,
                            'reconstructed_loss': reconstructed_loss, 
                            'prosody_loss': prosody_loss,
                            'att_output': att_output,
                            'ctc_output': asr_exec_inputs['ctc_output']}
        return total_loss, asr_exec_outputs

    def _write_asr_log(self, total_loss, asr_loss, cur_lr, asr_exec_outputs, txt, step, progress_step):
        if (self.rank == 0 or self.rank is None) and \
                ((step == 1) or (step % progress_step == 0)):
            self.progress('Tr stat | Total Loss - {:.2f} | ASR Loss - {:.2f}'
                          .format(total_loss.cpu().item(), asr_loss.cpu().item()), step)
            if self.paras.gpu:
                self.write_log('max_memory_allocated', {'memory': torch.cuda.max_memory_allocated(asr_exec_outputs['att_loss'].device)}, step)
                self.write_log('max_memory_reserved', {'memory': torch.cuda.max_memory_reserved(asr_exec_outputs['att_loss'].device)}, step)
            self.write_log('lr', {'lr': cur_lr}, step)
            self.write_log(
                'loss', {'tr_att': asr_exec_outputs['att_loss'].item(), 
                         'tr_ctc': asr_exec_outputs['ctc_loss'].item()}, step)
            if asr_exec_outputs['reconstructed_loss'] is not None:
                self.write_log('asr_reconstructed_loss', {'tr': asr_exec_outputs['reconstructed_loss'].item()}, step)
            if asr_exec_outputs['prosody_loss'] is not None:
                self.write_log('asr_prosody_loss', {'tr': asr_exec_outputs['prosody_loss'].item()}, step)
            if asr_exec_outputs['emb_loss'] is not None:
                self.write_log('asr_emb_loss', {'tr': asr_exec_outputs['emb_loss'].item()}, step)
            self.write_log('wer', {'tr_att': cal_er(self.tokenizer, asr_exec_outputs['att_output'], txt),
                                   'tr_ctc': cal_er(self.tokenizer, asr_exec_outputs['ctc_output'], txt, ctc=True)}, step)
            if self.emb_fuse:
                if self.emb_decoder.fuse_learnable:
                    self.write_log('fuse_lambda', {
                                   'emb': self.emb_decoder.get_weight()}, step)
                self.write_log(
                    'fuse_temp', {'temp': self.emb_decoder.get_temp()}, step)

    ### Validate

    def _validate_asr(self, general_solver, txt, att_output, ctc_output, dev_wer, count_asr, att_align, step):
        dev_wer['att'].append(cal_er(self.tokenizer, att_output, txt))
        dev_wer['ctc'].append(cal_er(self.tokenizer, ctc_output, txt, ctc=True))

        # Show some example on tensorboard
        if count_asr == len(self.dv_set)//2 and self.rank == 0:
            for count_asr in range(min(len(txt), self.DEV_N_EXAMPLE)):
                if general_solver.step == 1:
                    self.write_log('true_text{}'.format(
                        count_asr), self.tokenizer.decode(txt[count_asr].tolist()), step)
                if att_output is not None:
                    try:
                        self.write_log('att_align{}'.format(count_asr), feat_to_fig(
                            att_align[count_asr, 0, :, :].cpu().detach()), step)
                    except:
                        pass
                    self.write_log('att_text{}'.format(count_asr), self.tokenizer.decode(
                        att_output[count_asr].argmax(dim=-1).tolist()), step)
                if ctc_output is not None:
                    self.write_log('ctc_text{}'.format(count_asr), 
                                   self.tokenizer.decode(ctc_output[count_asr].argmax(dim=-1).tolist(), ignore_repeat=True), step)
        return dev_wer

    def _check_asr_improvement(self, general_solver, dev_wer, step):
        for task in ['att', 'ctc']:
            dev_wer[task] = sum(dev_wer[task])/len(dev_wer[task])
            if dev_wer[task] < self.best_wer[task]:
                self.best_wer[task] = dev_wer[task]
                general_solver.save_checkpoint('best_{}.pth'.format(task), 'wer', dev_wer[task])
            self.write_log('wer', {'dv_'+task: dev_wer[task]}, step)
        general_solver.save_checkpoint('latest.pth', 'wer', dev_wer['att'], show_msg=False)

    def verbose(self, msg):
        ''' Verbose function for print information to stdout'''
        if self.paras.verbose:
            if type(msg) == list:
                for m in msg:
                    print('[INFO]', m.ljust(100))
            else:
                print('[INFO]', msg.ljust(100))

    def progress(self, msg, step):
        ''' Verbose function for updating progress on stdout (do not include newline) '''
        return
        # if self.paras.verbose:
            # # sys.stdout.write("\033[K")  # Clear line
            # print('[{}] {}'.format(human_format(step), msg), end='\r')

    def write_log(self, log_name, log_dict, step):
        '''
        Write log to TensorBoard
            log_name  - <str> Name of tensorboard variable
            log_value - <dict>/<array> Value of variable (e.g. dict of losses), passed if value = None
        '''
        if type(log_dict) is dict:
            log_dict = {key: val for key, val in log_dict.items() if (
                val is not None and not math.isnan(val))}
        if log_dict is None:
            pass
        elif len(log_dict) > 0:
            if 'align' in log_name or 'spec' in log_name:
                img, form = log_dict
                self.log.add_image(
                    log_name, img, global_step=step, dataformats=form)
            elif 'text' in log_name or 'hyp' in log_name:
                self.log.add_text(log_name, log_dict, step)
            else:
                self.log.add_scalars(log_name, log_dict, step)

