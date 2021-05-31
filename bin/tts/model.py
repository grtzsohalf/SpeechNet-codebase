import torch
import torch.nn as nn
import numpy as np
from bin.tts.modules import Embedding, LengthRegulator, VariancePredictor
from bin.tts.utils import get_mask_from_lengths, get_sinusoid_encoding_table
class SpeakerIntegrator(nn.Module):
    """ Speaker Integrator """
    def __init__(self, config, spk_embed_table):
        super(SpeakerIntegrator, self).__init__()
        self.spk_embed_table = spk_embed_table
        n_spkers = len(spk_embed_table)
        spk_embed_dim = config['spk_embed_dim']
        embed_std = config['spk_embed_weight_std']
        
        self.embed_speakers = Embedding(n_spkers, spk_embed_dim, padding_idx=None, std=embed_std)
    
    def get_speaker_embedding(self, spk_ids):
        return self.embed_speakers(spk_ids)
    
    def forward(self, spembs, x):
        '''
        spembs shape : (batch, 256)
        x shape :      (batch, 39, 256)
        '''
        spembs = spembs.unsqueeze(1)
        spembs = spembs.repeat(1, x.shape[1] ,1)
        x = x + spembs
            
        return x
    
class VarianceAdaptor(nn.Module):
    """ Variance Adaptor """

    def __init__(self, model_config, datarc, device):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(model_config, datarc)
        self.length_regulator   = LengthRegulator(device)
        #self.pitch_predictor    = VariancePredictor(model_config, datarc)
        #self.energy_predictor   = VariancePredictor(model_config, datarc)
        

        self.model_config, self.datarc = model_config, datarc
        n_bins     = datarc['n_bins']
        f0_min     = datarc['f0_min']
        f0_max     = datarc['f0_max']
        energy_min = datarc['energy_min']
        energy_max = datarc['energy_max']
        
        self.encoder_hidden = model_config['word_dim']
        self.len_max = model_config['max_seq_len']

        
        n_position = self.len_max +1
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, self.encoder_hidden).unsqueeze(0), requires_grad=False)

        #self.pitch_bins = nn.Parameter(torch.exp(torch.linspace(np.log(f0_min), np.log(f0_max), n_bins-1)))
        #self.energy_bins = nn.Parameter(torch.linspace(energy_min, energy_max, n_bins-1))
        #self.pitch_embedding  = nn.Embedding(n_bins, encoder_hidden)
        #self.energy_embedding = nn.Embedding(n_bins, encoder_hidden)
            
    def forward(self, spembs, x, src_mask, mel_mask=None, duration_target=None, pitch_target=None, energy_target=None, max_len=None):
      
        ## Duration Predictor ##
        log_duration_prediction = self.duration_predictor(x+spembs.unsqueeze(1).repeat(1, x.shape[1], 1), src_mask)
        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
        else:
            duration_rounded = torch.clamp(torch.round(torch.exp(log_duration_prediction)-1.), min=0)
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(self.device, mel_len)

        # POSITIONAL ENCODING #
        # -- Forward
        if not self.training and x.shape[1] > self.len_max:
            var_output = x + get_sinusoid_encoding_table(x.shape[1], self.encoder_hidden)[:x.shape[1],:].unsqueeze(0).expand(x.shape[0], -1, -1).to(x.device)
        else:
            var_output = x + self.position_enc[:, :x.shape[1], :].expand(x.shape[0], -1, -1)
        """
        ## Pitch Predictor ##
        pitch_prediction = self.pitch_predictor(x, mel_mask)
        if pitch_target is not None:
            pitch_embedding = self.pitch_embedding(torch.bucketize(pitch_target.detach(), self.pitch_bins.detach()))
        else:
            pitch_embedding = self.pitch_embedding(torch.bucketize(pitch_prediction.detach(), self.pitch_bins.detach()))
        
        ## Energy Predictor ##
        energy_prediction = self.energy_predictor(x, mel_mask)
        if energy_target is not None:
            energy_embedding = self.energy_embedding(torch.bucketize(energy_target.detach(), self.energy_bins.detach()))
        else:
            energy_embedding = self.energy_embedding(torch.bucketize(energy_prediction.detach(), self.energy_bins.detach()))
        
        x = x + pitch_embedding + energy_embedding
        """
        #return x, log_duration_prediction, pitch_prediction, energy_prediction, mel_len, mel_mask
        return var_output, log_duration_prediction, mel_len, mel_mask
