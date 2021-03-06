import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from scipy.io import wavfile
import os
import librosa
import soundfile

from bin.tts import text

# Kaiwei 2020.10.06
# necessary to keep the duration(int) close to the original one

class FastSpeech2Utils():
    def __init__(self, data_config):
        self.data_config = data_config
        self.sampling_rate = self.data_config['sampling_rate']
        self.hop_length = self.data_config['hop_length']
        self.n_mel_channels = self.data_config['n_mel_channels']
        self.energy_min = self.data_config['energy_min']
        self.energy_max = self.data_config['energy_max']
        self.f0_max = self.data_config['f0_max']

    def duration_warp(self, real_d, int_d):
        total_diff = sum(real_d) - sum(int_d)
        drop_diffs = np.array(real_d) - np.array(int_d)
        drop_order = np.argsort(-drop_diffs)
        for i in range(int(total_diff)):
            index = drop_order[i]
            int_d[index] +=1
            
        return int_d 

    def get_alignment(self, tier):
        sil_phones = ['sil', 'sp', 'spn']
        phones = []
        durations_real = []
        durations_int = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trimming leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s
            if p not in sil_phones:
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                phones.append(p)
            
            d = e*self.sampling_rate/self.hop_length - s*self.sampling_rate/self.hop_length
            durations_real.append(d)
            durations_int.append(int(d))

        # Trimming tailing silences
        durations_real = durations_real[:end_idx]
        durations_int = durations_int[:end_idx]
        phones = phones[:end_idx]
        durations = self.duration_warp(durations_real, durations_int)
        
        return phones, durations, start_time, end_time

    def plot_data(self, data, titles=None, filename=None):
        fig, axes = plt.subplots(len(data), 1, squeeze=False)
        if titles is None:
            titles = [None for i in range(len(data))]

        def add_axis(fig, old_ax, offset=0):
            ax = fig.add_axes(old_ax.get_position(), anchor='W')
            ax.set_facecolor("None")
            return ax

        for i in range(len(data)):
            spectrogram, pitch, energy = data[i]
            axes[i][0].imshow(spectrogram, origin='lower')
            axes[i][0].set_aspect(2.5, adjustable='box')
            axes[i][0].set_ylim(0, self.n_mel_channels)
            axes[i][0].set_title(titles[i], fontsize='medium')
            axes[i][0].tick_params(labelsize='x-small', left=False, labelleft=False) 
            axes[i][0].set_anchor('W')
            
            ax1 = add_axis(fig, axes[i][0])
            ax1.plot(pitch, color='tomato')
            ax1.set_xlim(0, spectrogram.shape[1])
            ax1.set_ylim(0, self.f0_max)
            ax1.set_ylabel('F0', color='tomato')
            ax1.tick_params(labelsize='x-small', colors='tomato', bottom=False, labelbottom=False)
            
            ax2 = add_axis(fig, axes[i][0], 1.2)
            ax2.plot(energy, color='darkviolet')
            ax2.set_xlim(0, spectrogram.shape[1])
            ax2.set_ylim(self.energy_min, self.energy_max)
            ax2.set_ylabel('Energy', color='darkviolet')
            ax2.yaxis.set_label_position('right')
            ax2.tick_params(labelsize='x-small', colors='darkviolet', bottom=False, labelbottom=False, left=False, labelleft=False, right=True, labelright=True)
        
        plt.savefig(filename, dpi=200)
        plt.close()

    def get_melgan(self):
        melgan = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
        return melgan

    def melgan_infer(self, mel, melgan, path):
        wav = melgan.inverse(mel).squeeze(0).detach().cpu().numpy()
        soundfile.write(path, wav, self.sampling_rate)
        
    def melgan_infer_batch(self, mel, melgan):
        return melgan.inverse(mel).cpu().numpy()


def process_meta(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        text = []
        name = []
        for line in f.readlines():
            n, t = line.strip('\n').split('|')
            name.append(n)
            text.append(t)
        return name, text

def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

def get_mask_from_lengths(device, lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = (ids >= lengths.unsqueeze(1).expand(-1, max_len))

    return mask

def pad_1D(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = np.pad(x, (0, length - x.shape[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded

def pad_2D(inputs, maxlen=None):

    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(x, (0, max_len - np.shape(x)[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output

def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0)for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len-batch.size(0)), "constant", 0.0)
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len-batch.size(0)), "constant", 0.0)
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i)
                               for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.
    
    return torch.FloatTensor(sinusoid_table)