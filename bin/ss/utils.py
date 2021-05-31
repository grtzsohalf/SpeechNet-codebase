import math
import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import hann as hanning
import torch
import torch.nn as nn
from torch import Tensor
from argparse import Namespace


EPS = np.finfo("float").eps
hann = torch.Tensor(scipy.hanning(258)[1:-1])
PAD_SIZE = 27


def update_args(old, new):
    old_dict = vars(old)
    new_dict = vars(new)
    old_dict.update(new_dict)
    return Namespace(**old_dict)

def remove_self(variables):
    return {k : v for k, v in variables.items() if k != 'self'}

def masked_mean(batch, length_masks, keepdim=False, eps=1e-8):
    # batch: (batch_size, max_time)
    means = (batch * length_masks).sum(dim=-1, keepdim=keepdim) / (length_masks.sum(dim=-1, keepdim=keepdim) + eps)
    return means

def masked_normalize_decibel(audio, target, length_masks, eps=1e-8):
    # audio: (batch_size, max_time)
    # length_masks: (batch_size, max_time)

    if type(target) is float:
        # target: fixed decibel level
        target = torch.ones(len(audio)).to(device=audio.device) * target
    elif type(target) is torch.Tensor and target.dim() > 1:
        # target: reference audio for decibel level
        target = 10.0 * torch.log10(masked_mean(target.pow(2), length_masks, keepdim=False))
    assert type(target) is torch.Tensor and target.dim() == 1
    # target: (batch_size, ), each utterance has a target decibel level

    scalar_square = (10.0 ** (target.unsqueeze(-1) / 10.0)) / (masked_mean(audio.pow(2), length_masks, keepdim=True) + eps)        
    scalar = scalar_square.pow(0.5)
    return audio * scalar

def plot_spectrogram(spec, height=2):
    spec = spec.detach().cpu().squeeze().transpose(0, 1).flip(dims=[0])
    assert spec.dim() == 2
    h, w = spec.size(0), spec.size(1)
    scaling = height / h
    fig = plt.figure(figsize=(round(w * scaling), round(h * scaling)))
    plt.imshow(spec)
    return fig

def plot_spectrograms(specs, height=2):
    assert type(specs) is list
    specs = [spec.detach().cpu().squeeze().transpose(0, 1).flip(dims=[0]) for spec in specs]
    h, w = specs[0].size(0), specs[0].size(1)
    scaling = height / h
    fig, axes = plt.subplots(len(specs), 1, figsize=(round(w * scaling), len(specs) * round(h * scaling)), gridspec_kw = {'wspace':0, 'hspace':0})
    for i in range(len(specs)):
        spec = specs[i]
        assert spec.dim() == 2
        h, w = spec.size(0), spec.size(1)        
        axes[i].imshow(spec)
    return fig


class Silence_Remover(nn.Module):
    def __init__(self, device, use_ref=False):
        super().__init__()
        self.N_FRAME = 256
        self.w = torch.Tensor(scipy.hanning(
            self.N_FRAME + 2)[1:-1]).to(device)
        self.EPS = np.finfo("float").eps
        self.use_ref = use_ref

    def forward(self, x, y, dyn_range=40, framelen=256, hop=128):
        x_frames = self.w * x.unfold(0, framelen, hop)
        y_frames = self.w * y.unfold(0, framelen, hop)

        if(self.use_ref):
            energies = 20 * \
                torch.log10(torch.norm(y_frames, p=2, dim=1) + self.EPS)
        else:
            energies = 20 * \
                torch.log10(torch.norm(x_frames, p=2, dim=1) + self.EPS)

        speech_part = (torch.max(energies) - dyn_range - energies) < 0
        silence_part = (torch.max(energies) - dyn_range - energies) >= 0

        if(silence_part.sum() != 0):
            silence = x_frames[silence_part]
            silence = silence.reshape(
                silence.shape[0]*2, math.floor(silence.shape[1]/2))
            silence = torch.cat([silence[0], torch.flatten(
                silence[::2][1:] + silence[1::2][:-1]), silence[-1]], dim=0)
        else:
            silence = torch.zeros(56)

        x_frames = x_frames[speech_part]
        y_frames = y_frames[speech_part]
        x_frames = x_frames.reshape(
            x_frames.shape[0]*2, math.floor(x_frames.shape[1]/2))
        x_speech = torch.cat([x_frames[0], torch.flatten(
            x_frames[::2][1:] + x_frames[1::2][:-1]), x_frames[-1]], dim=0)

        y_frames = y_frames.reshape(
            y_frames.shape[0]*2, math.floor(y_frames.shape[1]/2))
        y_speech = torch.cat([y_frames[0], torch.flatten(
            y_frames[::2][1:] + y_frames[1::2][:-1]), y_frames[-1]], dim=0)

        return x_speech, y_speech, silence


class Resampler(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def _lcm(self, a, b):
        return abs(a * b) // math.gcd(a, b)

    def _get_num_LR_output_samples(self, input_num_samp, samp_rate_in, samp_rate_out):
        samp_rate_in = int(samp_rate_in)
        samp_rate_out = int(samp_rate_out)

        tick_freq = self._lcm(samp_rate_in, samp_rate_out)
        ticks_per_input_period = tick_freq // samp_rate_in

        interval_length_in_ticks = input_num_samp * ticks_per_input_period
        if interval_length_in_ticks <= 0:
            return 0
        ticks_per_output_period = tick_freq // samp_rate_out

        last_output_samp = interval_length_in_ticks // ticks_per_output_period

        if last_output_samp * ticks_per_output_period == interval_length_in_ticks:
            last_output_samp -= 1

        num_output_samp = last_output_samp + 1
        return num_output_samp

    def _get_LR_indices_and_weights(self, orig_freq, new_freq, output_samples_in_unit, window_width,
                                    lowpass_cutoff, lowpass_filter_width):
        assert lowpass_cutoff < min(orig_freq, new_freq) / 2
        output_t = torch.arange(0, output_samples_in_unit,
                                dtype=torch.get_default_dtype()) / new_freq
        min_t = output_t - window_width
        max_t = output_t + window_width

        min_input_index = torch.ceil(min_t * orig_freq)

        max_input_index = torch.floor(max_t * orig_freq)
        num_indices = max_input_index - min_input_index + 1

        max_weight_width = num_indices.max()

        j = torch.arange(max_weight_width).unsqueeze(0)
        input_index = min_input_index.unsqueeze(1) + j
        delta_t = (input_index / orig_freq) - output_t.unsqueeze(1)

        weights = torch.zeros_like(delta_t)
        inside_window_indices = delta_t.abs().lt(window_width)

        weights[inside_window_indices] = 0.5 * (1 + torch.cos(2 * math.pi * lowpass_cutoff /
                                                              lowpass_filter_width * delta_t[inside_window_indices]))

        t_eq_zero_indices = delta_t.eq(0.0)
        t_not_eq_zero_indices = ~t_eq_zero_indices

        weights[t_not_eq_zero_indices] *= torch.sin(
            2 * math.pi * lowpass_cutoff * delta_t[t_not_eq_zero_indices]) / (math.pi * delta_t[t_not_eq_zero_indices])

        weights[t_eq_zero_indices] *= 2 * lowpass_cutoff

        weights /= orig_freq
        return min_input_index, weights

    def forward(self, waveform, orig_freq, new_freq, lowpass_filter_width=6):
        assert waveform.dim() == 2
        assert orig_freq > 0.0 and new_freq > 0.0

        min_freq = min(orig_freq, new_freq)
        lowpass_cutoff = 0.99 * 0.5 * min_freq

        assert lowpass_cutoff * 2 <= min_freq

        base_freq = math.gcd(int(orig_freq), int(new_freq))
        input_samples_in_unit = int(orig_freq) // base_freq
        output_samples_in_unit = int(new_freq) // base_freq

        window_width = lowpass_filter_width / (2.0 * lowpass_cutoff)
        first_indices, weights = self._get_LR_indices_and_weights(orig_freq, new_freq, output_samples_in_unit,
                                                                  window_width, lowpass_cutoff, lowpass_filter_width)
        weights = weights.to(self.device)

        assert first_indices.dim() == 1
        conv_stride = input_samples_in_unit
        conv_transpose_stride = output_samples_in_unit
        num_channels, wave_len = waveform.size()
        window_size = weights.size(1)
        tot_output_samp = self._get_num_LR_output_samples(
            wave_len, orig_freq, new_freq)
        output = torch.zeros((num_channels, tot_output_samp))

        output = output.to(self.device)

        eye = torch.eye(num_channels).unsqueeze(2)

        eye = eye.to(self.device)

        for i in range(first_indices.size(0)):
            wave_to_conv = waveform
            first_index = int(first_indices[i].item())
            if first_index >= 0:
                wave_to_conv = wave_to_conv[..., first_index:]

            max_unit_index = (tot_output_samp - 1) // output_samples_in_unit
            end_index_of_last_window = max_unit_index * conv_stride + window_size
            current_wave_len = wave_len - first_index
            right_padding = max(
                0, end_index_of_last_window + 1 - current_wave_len)

            left_padding = max(0, -first_index)
            if left_padding != 0 or right_padding != 0:
                wave_to_conv = torch.nn.functional.pad(
                    wave_to_conv, (left_padding, right_padding))

            conv_wave = torch.nn.functional.conv1d(
                wave_to_conv.unsqueeze(0), weights[i].repeat(
                    num_channels, 1, 1),
                stride=conv_stride, groups=num_channels)

            dilated_conv_wave = torch.nn.functional.conv_transpose1d(
                conv_wave, eye, stride=conv_transpose_stride).squeeze(0)

            dialated_conv_wave_len = dilated_conv_wave.size(-1)
            left_padding = i
            right_padding = max(0, tot_output_samp -
                                (left_padding + dialated_conv_wave_len))
            dilated_conv_wave = torch.nn.functional.pad(
                dilated_conv_wave, (left_padding, right_padding))[..., :tot_output_samp]

            output += dilated_conv_wave

        return output
