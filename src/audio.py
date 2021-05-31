import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio import transforms
from collections import namedtuple
from src.nb_SpecAugment import time_warp, time_mask, freq_mask
import numpy as np 
from scipy.signal import savgol_coeffs
import scipy
from apex import amp

import torch.cuda.nvtx as nvtx

# class CMVN(torch.jit.ScriptModule):
class CMVN(nn.Module):

    __constants__ = ["mode", "dim", "eps"]

    def __init__(self, mode="global", dim=2, eps=1e-10):
        # `torchaudio.load()` loads audio with shape [channel, feature_dim, time]
        # so perform normalization on dim=2 by default
        super(CMVN, self).__init__()

        if mode != "global":
            raise NotImplementedError(
                "Only support global mean variance normalization.")

        self.mode = mode
        self.dim = dim
        self.eps = eps

    # @torch.jit.script_method
    def forward(self, x):
        if self.mode == "global":
            nvtx.range_push('CMVN')
            length = x[1]
            input_x = x[0]
            for i in range(len(length)):
                input_x[i] = (input_x[i] - input_x[i, :, :, :length[i]].mean(-1, keepdim=True)) / (self.eps + input_x[i, :, :, :length[i]].std(-1, keepdim=True))

            nvtx.range_pop()
            return input_x, length

    def extra_repr(self):
        return "mode={}, dim={}, eps={}".format(self.mode, self.dim, self.eps)


# class Delta(torch.jit.ScriptModule):
class Delta(nn.Module):

    __constants__ = ["order", "window_size", "padding"]

    def __init__(self, order=1, window_size=2):
        # Reference:
        # https://kaldi-asr.org/doc/feature-functions_8cc_source.html
        # https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_audio.py
        super(Delta, self).__init__()

        self.order = order
        self.window_size = window_size

        self.filters = self._create_filters(order, window_size)
        # self.register_buffer("filters", filters)
        # print(filters.shape)
        self.padding = (0, (self.filters.shape[-1] - 1) // 2)
        # (out_channels, in_channels, kernel_size[0]. kernel_size[1]): (3, 1, 1, 9)
        self.conv = nn.Conv2d(1, 3, (1, 9), padding=self.padding, bias=False)
        self.conv.weight = nn.Parameter(self.filters)

    # @torch.jit.script_method
    def forward(self, x):
        # Unsqueeze batch dim
        nvtx.range_push('Delta')
        length =x[1]
        x = x[0].unsqueeze(1)
        nvtx.range_pop()

        return self.conv(x), length

    # TODO(WindQAQ): find more elegant way to create `scales`
    def _create_filters(self, order, window_size):
        scales = [[1.0]]
        for i in range(1, order + 1):
            prev_offset = (len(scales[i-1]) - 1) // 2
            curr_offset = prev_offset + window_size

            curr = [0] * (len(scales[i-1]) + 2 * window_size)
            normalizer = 0.0
            for j in range(-window_size, window_size + 1):
                normalizer += j * j
                for k in range(-prev_offset, prev_offset + 1):
                    curr[j+k+curr_offset] += (j * scales[i-1][k+prev_offset])
            curr = [x / normalizer for x in curr]
            scales.append(curr)

        max_len = len(scales[-1])
        for i, scale in enumerate(scales[:-1]):
            padding = (max_len - len(scale)) // 2
            scales[i] = [0] * padding + scale + [0] * padding

        return torch.tensor(scales).unsqueeze(1).unsqueeze(1)

    def extra_repr(self):
        return "order={}, window_size={}".format(self.order, self.window_size)


# class Postprocess(torch.jit.ScriptModule):
class Postprocess(nn.Module):
    # @torch.jit.script_method
    def forward(self, inputs):
        nvtx.range_push('Post process')
        # [batch_size, channel, feature_dim, time] -> [batch_size, time, channel, feature_dim]
        x, length = inputs
        x = x.permute(0, 3, 1, 2) 
        # [time, channel, feature_dim] -> [time, feature_dim * channel]
        nvtx.range_pop()
        return x.reshape(x.size(0), x.size(1), -1).detach(), length
# class SpecAug(torch.jit.ScriptModule):
class SpecAug(nn.Module):
    #@torch.jit.script_method
    def forward(self, inputs):
        x, length = inputs
        nvtx.range_push('SpecAug')
        # import ipdb; ipdb.set_trace()
        # [time, feature_dim * channel]
        mask_spec = []
        for i in range(x.shape[0]):
            # output = time_mask(freq_mask(time_warp(x[i].permute(1, 0).unsqueeze(0)), num_masks=2), num_masks=2)
            # warped_output = time_warp(x[i].permute(1, 0).unsqueeze(0))
            output = time_mask(freq_mask(x[i].permute(1, 0).unsqueeze(0), num_masks=2), num_masks=2)
            mask_spec.append(output.squeeze(0).permute(1, 0))
        
        #x = time_mask(freq_mask(time_warp(x), num_masks=2), num_masks=2)
        nvtx.range_pop()
        return torch.stack(mask_spec), length

# TODO(Windqaq): make this scriptable
class ExtractAudioFeature(nn.Module):
    def __init__(self, mode="fbank", num_mel_bins=40, sample_rate=16000, n_linear=1025, **kwargs):
        super(ExtractAudioFeature, self).__init__()
        self.mode = mode
        
        self.n_fft, self.hop_length, self.win_length = _stft_parameters(
            n_linear, kwargs['frame_shift'], kwargs['frame_length'], sample_rate=sample_rate)
        self.linear_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length, power=2)
        self.mel_transform = torchaudio.transforms.MelScale(
            n_mels=num_mel_bins, n_stft=n_linear)
        self.mel_transform.fb = create_fb_matrix(
            n_linear, 0, 8000, num_mel_bins, sample_rate, norm="slaney")
        
        # self.extract_fn = torchaudio.compliance.kaldi.fbank if mode == "fbank" else torchaudio.compliance.kaldi.mfcc
        self.num_mel_bins = num_mel_bins
        self.sample_rate = sample_rate
        self.kwargs = kwargs

    def forward(self, input_object):
        nvtx.range_push('Extract audio feature')
        waveform = input_object[0]
        wave_length = input_object[1]

        linear_spectrogram = self.linear_transform(waveform)
        mel_spectrogram = (self.mel_transform(linear_spectrogram) + 1e-6).log()
        length = wave_length//self.hop_length+1

        nvtx.range_pop()

        return mel_spectrogram.detach(), length

    def extra_repr(self):
        return "mode={}, num_mel_bins={}".format(self.mode, self.num_mel_bins)


def create_transform(audio_config):
    audio_config = copy.deepcopy(audio_config)
    feat_type = audio_config.pop("feat_type")
    feat_dim = audio_config.pop("feat_dim")
    sample_rate = audio_config.pop("sample_rate")

    delta_order = audio_config.pop("delta_order", 0)
    delta_window_size = audio_config.pop("delta_window_size", 2)
    apply_cmvn = audio_config.pop("apply_cmvn")

    transforms = [ExtractAudioFeature(feat_type, feat_dim, sample_rate, **audio_config)]

    if delta_order >= 1:
        transforms.append(Delta(delta_order, delta_window_size))

    if apply_cmvn:
        transforms.append(CMVN())

    transforms.append(Postprocess())
    transforms.append(SpecAug())

    return nn.Sequential(*transforms), feat_dim * (delta_order + 1), sample_rate


def mel_to_hz(mels, htk=False):
    """Convert mel bin numbers to frequencies

    Examples
    --------
    >>> librosa.mel_to_hz(3)
    200.

    >>> librosa.mel_to_hz([1,2,3,4,5])
    array([  66.667,  133.333,  200.   ,  266.667,  333.333])

    Parameters
    ----------
    mels          : np.ndarray [shape=(n,)], float
        mel bins to convert
    htk           : bool
        use HTK formula instead of Slaney

    Returns
    -------
    frequencies   : np.ndarray [shape=(n,)]
        input mels in Hz

    See Also
    --------
    hz_to_mel
    """

    mels = np.asanyarray(mels)

    if htk:
        return 700.0 * (10.0**(mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0                         # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep = np.log(6.4) / 27.0                # step size for log region

    if mels.ndim:
        # If we have vector data, vectorize
        log_t = (mels >= min_log_mel)
        freqs[log_t] = min_log_hz * \
            np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        # If we have scalar data, check directly
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))

    return freqs


def hz_to_mel(frequencies, htk=False):
    """Convert Hz to Mels

    Examples
    --------
    >>> librosa.hz_to_mel(60)
    0.9
    >>> librosa.hz_to_mel([110, 220, 440])
    array([ 1.65,  3.3 ,  6.6 ])

    Parameters
    ----------
    frequencies   : number or np.ndarray [shape=(n,)] , float
        scalar or array of frequencies
    htk           : bool
        use HTK formula instead of Slaney

    Returns
    -------
    mels        : number or np.ndarray [shape=(n,)]
        input frequencies in Mels

    See Also
    --------
    mel_to_hz
    """

    frequencies = np.asanyarray(frequencies)

    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0                         # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep = np.log(6.4) / 27.0                # step size for log region

    if frequencies.ndim:
        # If we have array data, vectorize
        log_t = (frequencies >= min_log_hz)
        mels[log_t] = min_log_mel + np.log(frequencies[log_t]/min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        # If we have scalar data, heck directly
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep

    return mels


def create_fb_matrix(n_freqs, f_min, f_max, n_mels, sample_rate, htk=False, norm=None):
    if norm is not None and norm != "slaney":
        raise ValueError("norm must be one of None or 'slaney'")

    # freq bins
    # Equivalent filterbank construction by Librosa
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)

    # calculate mel freq bins
    # hertz to mel(f) is 2595. * math.log10(1. + (f / 700.))
    # m_min = 2595.0 * math.log10(1.0 + (f_min / 700.0))
    # m_max = 2595.0 * math.log10(1.0 + (f_max / 700.0))
    m_min = hz_to_mel(f_min, htk=False)
    m_max = hz_to_mel(f_max, htk=False)

    # print("min vs max: ", m_min, m_max)
    # print("min2 vs max2: ", m_min1, m_max1)

    m_pts = torch.linspace(m_min, m_max, n_mels+2)

    f_pts = torch.from_numpy(mel_to_hz(m_pts, htk=htk))
    # mel to hertz(mel) is 700. * (10**(mel / 2595.) - 1.)
    # f_pts = 700.0 * (10 ** (m_pts / 2595.0) - 1.0)
    # calculate the difference between each mel point and each stft freq point in hertz
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_mels + 1)
    # (n_freqs, n_mels + 2)
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)
    # create overlapping triangles
    zero = torch.zeros(1)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_mels)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_mels)
    fb = torch.max(zero, torch.min(down_slopes, up_slopes))

    if norm is not None and norm == "slaney":
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (f_pts[2:n_mels + 2] - f_pts[:n_mels])
        fb *= enorm.unsqueeze(0)

    return fb

def compute_deltas(
        specgram,
        win_length: int = 9,
        mode: str = "interp"):
    
    r"""Compute delta coefficients of a tensor, usually a spectrogram:

    .. math::
       d_t = \frac{\sum_{n=1}^{\text{N}} n (c_{t+n} - c_{t-n})}{2 \sum_{n=1}^{\text{N}} n^2}

    where :math:`d_t` is the deltas at time :math:`t`,
    :math:`c_t` is the spectrogram coeffcients at time :math:`t`,
    :math:`N` is ``(win_length-1)//2``.

    Args:
        specgram (Tensor): Tensor of audio of dimension (..., freq, time)
        win_length (int, optional): The window length used for computing delta (Default: ``5``)
        mode (str, optional): Mode parameter passed to padding (Default: ``"replicate"``)

    Returns:
        Tensor: Tensor of deltas of dimension (..., freq, time)

    Example
        >>> specgram = torch.randn(1, 40, 1000)
        >>> delta = compute_deltas(specgram)
        >>> delta2 = compute_deltas(delta)
    """
    from scipy.signal import savgol_coeffs

    output = scipy.signal.savgol_filter(specgram, 9,
                                      deriv=1,
                                      axis=-1,
                                      polyorder=1,
                                      mode=mode)
    return torch.tensor(output)


def _stft_parameters(num_freq, frame_shift_ms, frame_length_ms, sample_rate):
    n_fft = (num_freq - 1) * 2
    hop_length = int(frame_shift_ms / 1000 * sample_rate)
    win_length = int(frame_length_ms / 1000 * sample_rate)
    return n_fft, hop_length, win_length
