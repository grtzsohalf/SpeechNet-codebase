import yaml
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from src.model import Model
# from src.lm import RNNLM
from src.ctc import CTCPrefixScore

from src.transformer.mask import target_mask
from src.transformer.nets_utils import make_non_pad_mask


CTC_BEAM_RATIO = 1.5   # DO NOT CHANGE THIS, MAY CAUSE OOM
LOG_ZERO = -10000000.0  # Log-zero for CTC


class BeamDecoder(nn.Module):
    ''' Beam decoder for ASR '''

    def __init__(self, model, emb_decoder, beam_size, min_len_ratio, max_len_ratio,
                 lm_path='', lm_config='', lm_weight=0.0, ctc_weight=0.0):
        super().__init__()
        # Setup
        self.beam_size = beam_size
        self.min_len_ratio = min_len_ratio
        self.max_len_ratio = max_len_ratio
        self.model = model

        # ToDo : implement pure ctc decode
        assert self.model.enable_att

        # Additional decoding modules
        self.apply_ctc = ctc_weight > 0
        if self.apply_ctc:
            assert self.model.ctc_weight > 0, 'ASR was not trained with CTC decoder'
            self.ctc_w = ctc_weight
            self.ctc_beam_size = int(CTC_BEAM_RATIO * self.beam_size)

        self.apply_lm = lm_weight > 0
        if self.apply_lm:
            self.lm_w = lm_weight
            self.lm_path = lm_path
            lm_config = yaml.load(open(lm_config, 'r'), Loader=yaml.FullLoader)
            self.lm = Model(False, True, None, self.model.vocab_size, False, **lm_config['model'])
            self.lm.load_state_dict(torch.load(
                self.lm_path, map_location='cpu')['model'])

        self.apply_emb = emb_decoder is not None
        if self.apply_emb:
            self.emb_decoder = emb_decoder

    def create_msg(self):
        msg = ['Decode spec| Beam size = {}\t| Min/Max len ratio = {}/{}'.format(
            self.beam_size, self.min_len_ratio, self.max_len_ratio)]
        if self.apply_ctc:
            msg.append(
                '           |Joint CTC decoding enabled \t| weight = {:.2f}\t'.format(self.ctc_w))
        if self.apply_lm:
            msg.append('           |Joint LM decoding enabled \t| weight = {:.2f}\t| src = {}'.format(
                self.lm_w, self.lm_path))
        if self.apply_emb:
            msg.append('           |Joint Emb. decoding enabled \t| weight = {:.2f}'.format(
                self.lm_w, self.emb_decoder.fuse_lambda.mean().cpu().item()))

        return msg

    def forward(self, audio_feature, feature_len):
        # Init.
        assert audio_feature.shape[0] == 1, "Batchsize == 1 is required for beam search"
        batch_size = audio_feature.shape[0]
        device = audio_feature.device
        dec_state = self.model.text_decoder.init_state(
            batch_size)                           # Init zero states
        if self.model.text_decoder.module in ['LSTM', 'GRU']:
            self.model.attention.reset_mem()            # Flush attention mem
        # Max output len set w/ hyper param.
        max_output_len = int(
            np.ceil(feature_len.cpu().item()*self.max_len_ratio))
        # Min output len set w/ hyper param.
        min_output_len = int(
            np.ceil(feature_len.cpu().item()*self.min_len_ratio))
        # Store attention map if location-aware
        if self.model.text_decoder.module in ['LSTM', 'GRU']:
            store_att = self.model.attention.mode == 'loc'
        prev_token = torch.zeros(
            (batch_size, 1), dtype=torch.long, device=device)     # Start w/ <sos>
        # Cache of beam search
        final_hypothesis, next_top_hypothesis = [], []
        # Incase ctc is disabled
        ctc_state, ctc_prob, candidates, lm_state = None, None, None, None

        # Encode
        encode_len, encode_feature_content, encode_mask_content, \
            encode_feature_speaker, encode_mask_speaker = self.model.audio_encoder(audio_feature, feature_len)

        # CTC decoding
        if self.apply_ctc:
            ctc_output = F.log_softmax(
                self.model.ctc_layer(encode_feature_content), dim=-1)
            ctc_prefix = CTCPrefixScore(ctc_output)
            ctc_state = ctc_prefix.init_state()

        # Start w/ empty hypothesis
        prev_top_hypothesis = [Hypothesis(decoder_state=dec_state, output_seq=[],
                                          output_scores=[], lm_state=None, ctc_prob=0,
                                          ctc_state=ctc_state, att_map=None)]
        # Attention decoding
        for t in range(max_output_len):
            for hypothesis in prev_top_hypothesis:
                # Resume previous step
                prev_seq, prev_token, prev_dec_state, prev_attn, prev_lm_state, prev_ctc_state = hypothesis.get_state(
                    device)
                self.model.set_state(prev_dec_state, prev_attn)

                # Normal asr forward
                if self.model.text_decoder.module in ['LSTM', 'GRU']:
                    attn, context = self.model.attention(
                        self.model.text_decoder.get_query(), encode_feature_content, encode_len)
                    asr_prev_token = self.model.pre_embed(prev_token)
                    text_decoder_input = torch.cat([asr_prev_token, context], dim=-1)
                    cur_prob, d_state = self.model.text_decoder(text_decoder_input)
                elif self.model.text_decoder.module == 'Transformer':
                    txt_input = self.model.pre_embed(prev_seq).unsqueeze(0)
                    txt_mask = make_non_pad_mask([t+1]).to(txt_input.device)
                    txt_mask = target_mask(txt_mask, 0) 
                    d_state = self.model.text_decoder(encode_feature_content, encode_mask_content, txt_input, txt_mask)[:, -1]
                    cur_prob = self.model.text_decoder.char_trans(self.model.text_decoder.final_dropout(d_state))

                # Embedding fusion (output shape 1xV)
                if self.apply_emb:
                    _, cur_prob = self.emb_decoder( d_state, cur_prob, return_loss=False)
                else:
                    cur_prob = F.log_softmax(cur_prob, dim=-1)

                # Perform CTC prefix scoring on limited candidates (else OOM easily)
                if self.apply_ctc:
                    # TODO : Check the performance drop for computing part of candidates only
                    _, ctc_candidates = cur_prob.squeeze(0).topk(self.ctc_beam_size, dim=-1)
                    candidates = ctc_candidates.cpu().tolist()
                    ctc_prob, ctc_state = ctc_prefix.cheap_compute(
                        hypothesis.outIndex, prev_ctc_state, candidates)
                    # TODO : study why ctc_char (slightly) > 0 sometimes
                    ctc_char = torch.FloatTensor(ctc_prob - hypothesis.ctc_prob).to(device)

                    # Combine CTC score and Attention score (HACK: focus on candidates, block others)
                    hack_ctc_char = torch.zeros_like(cur_prob).data.fill_(LOG_ZERO)
                    for idx, char in enumerate(candidates):
                        hack_ctc_char[0, char] = ctc_char[idx]
                    cur_prob = (1-self.ctc_w)*cur_prob + self.ctc_w*hack_ctc_char  # ctc_char
                    cur_prob[0, 0] = LOG_ZERO  # Hack to ignore <sos>

                # Joint RNN-LM decoding
                if self.apply_lm:
                    # assuming batch size always 1, resulting 1x1
                    lm_input = prev_token.unsqueeze(1)

                    asr_inputs = {'audio_feature': None,
                                  'feature_len': None,
                                  'decode_step': None,
                                  'tf_rate': None,
                                  'teacher': None,
                                  'emb_decoder': None,
                                  'get_dec_state': None,
                                  'txt_len': None}
                    lm_inputs = {'x': lm_input,
                                 'lens': torch.ones([batch_size]),
                                 'hidden': prev_lm_state}

                    _, lm_outputs = self.lm(asr_inputs, lm_inputs)
                    lm_output = lm_outputs['pred']
                    lm_state = lm_outputs['hidden']
                    # _, _, _, _, _, lm_output, lm_state = \
                        # self.lm(None, None, None, None, None, None, None, None,
                        # lm_input, torch.ones([batch_size]), hidden=prev_lm_state)

                    # assuming batch size always 1,  resulting 1xV
                    lm_output = lm_output.squeeze(0)
                    cur_prob += self.lm_w*lm_output.log_softmax(dim=-1)

                # Beam search
                # Note: Ignored batch dim.
                topv, topi = cur_prob.squeeze(0).topk(self.beam_size)
                if self.model.text_decoder.module in ['LSTM', 'GRU']:
                    prev_attn = self.model.attention.att_layer.prev_att.cpu() if store_att else None
                else:
                    prev_attn = None
                final, top = hypothesis.addTopk(topi, topv, self.model.text_decoder.get_state(), att_map=prev_attn,
                                                lm_state=lm_state, ctc_state=ctc_state, ctc_prob=ctc_prob,
                                                ctc_candidates=candidates)
                # Move complete hyps. out
                if final is not None and (t >= min_output_len):
                    final_hypothesis.append(final)
                    if self.beam_size == 1:
                        return final_hypothesis
                next_top_hypothesis.extend(top)

            # Sort for top N beams
            next_top_hypothesis.sort(key=lambda o: o.avgScore(), reverse=True)
            prev_top_hypothesis = next_top_hypothesis[:self.beam_size]
            next_top_hypothesis = []

        # Rescore all hyp (finished/unfinished)
        final_hypothesis += prev_top_hypothesis
        final_hypothesis.sort(key=lambda o: o.avgScore(), reverse=True)

        return final_hypothesis[:self.beam_size]


class Hypothesis:
    '''Hypothesis for beam search decoding.
       Stores the history of label sequence & score 
       Stores the previous text_decoder state, ctc state, ctc score, lm state and attention map (if necessary)'''

    def __init__(self, decoder_state, output_seq, output_scores, lm_state, ctc_state, ctc_prob, att_map):
        assert len(output_seq) == len(output_scores)
        # attention decoder
        self.decoder_state = decoder_state
        self.att_map = att_map

        # RNN language model
        if type(lm_state) is tuple:
            self.lm_state = (lm_state[0].cpu(),
                             lm_state[1].cpu())  # LSTM state
        elif lm_state is None:
            self.lm_state = None                                  # Init state
        else:
            self.lm_state = lm_state.cpu()                        # GRU state

        # Previous outputs
        self.output_seq = output_seq        # Prefix, List of list
        self.output_scores = output_scores  # Prefix score, list of float

        # CTC decoding
        self.ctc_state = ctc_state          # List of np
        self.ctc_prob = ctc_prob            # List of float

    def avgScore(self):
        '''Return the averaged log probability of hypothesis'''
        assert len(self.output_scores) != 0
        return sum(self.output_scores) / len(self.output_scores)

    def addTopk(self, topi, topv, decoder_state, att_map=None,
                lm_state=None, ctc_state=None, ctc_prob=0.0, ctc_candidates=[]):
        '''Expand current hypothesis with a given beam size'''
        new_hypothesis = []
        term_score = None
        ctc_s, ctc_p = None, None
        beam_size = topi.shape[-1]

        for i in range(beam_size):
            # Detect <eos>
            if topi[i].item() == 1:
                term_score = topv[i].cpu()
                continue

            idxes = self.output_seq[:]     # pass by value
            scores = self.output_scores[:]  # pass by value
            idxes.append(topi[i].cpu())
            scores.append(topv[i].cpu())
            if ctc_state is not None:
                # ToDo: Handle out-of-candidate case.
                idx = ctc_candidates.index(topi[i].item())
                ctc_s = ctc_state[idx, :, :]
                ctc_p = ctc_prob[idx]
            new_hypothesis.append(Hypothesis(decoder_state,
                                             output_seq=idxes, output_scores=scores, lm_state=lm_state,
                                             ctc_state=ctc_s, ctc_prob=ctc_p, att_map=att_map))
        if term_score is not None:
            self.output_seq.append(torch.tensor(1))
            self.output_scores.append(term_score)
            return self, new_hypothesis
        return None, new_hypothesis

    def get_state(self, device):
        prev_token = self.output_seq[-1] if len(self.output_seq) != 0 else 0
        prev_token = torch.LongTensor([prev_token]).to(device)
        att_map = self.att_map.to(device) if self.att_map is not None else None
        if type(self.lm_state) is tuple:
            lm_state = (self.lm_state[0].to(device),
                        self.lm_state[1].to(device))  # LSTM state
        elif self.lm_state is None:
            lm_state = None                                  # Init state
        else:
            lm_state = self.lm_state.to(
                device)                        # GRU state
        prev_seq = torch.LongTensor([0]).to(device)
        if len(self.output_seq) != 0:
            prev_seq = torch.cat([prev_seq, torch.LongTensor(self.output_seq).to(device)], dim=-1)
        return prev_seq, prev_token, self.decoder_state, att_map, lm_state, self.ctc_state

    @property
    def outIndex(self):
        return [i.item() for i in self.output_seq]
