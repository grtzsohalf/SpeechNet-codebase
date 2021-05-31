from torch import nn
import torch
from asteroid.masknn import norms
from asteroid.masknn import activations
from asteroid.utils import has_arg


class Conv1DBlock(nn.Module):

    def __init__(self, in_chan, hid_chan, kernel_size, padding,
                 dilation, norm_type="gLN"):
        super(Conv1DBlock, self).__init__()

        conv_norm = norms.get(norm_type)
        depth_conv1d = nn.Conv1d(in_chan, hid_chan, kernel_size,
                                 padding=padding, dilation=dilation)

        self.out = nn.Sequential(depth_conv1d, nn.PReLU(), conv_norm(hid_chan))

    def forward(self, x):
        """ Input shape [batch, feats, seq]"""

        return self.out(x)


class SepConv1DBlock(nn.Module):

    def __init__(self, in_chan, hid_chan, spk_vec_chan, kernel_size, padding,
                 dilation, norm_type="gLN", use_FiLM=True):
        super(SepConv1DBlock, self).__init__()

        self.use_FiLM = use_FiLM
        conv_norm = norms.get(norm_type)
        self.depth_conv1d = nn.Conv1d(in_chan, hid_chan, kernel_size,
                                 padding=padding, dilation=dilation)
        self.out = nn.Sequential(nn.PReLU(),
                                          conv_norm(hid_chan))

        # FiLM conditioning
        if self.use_FiLM:
            self.mul_lin = nn.Linear(spk_vec_chan, hid_chan)
        self.add_lin = nn.Linear(spk_vec_chan, hid_chan)

    def apply_conditioning(self, spk_vec, squeezed):
        bias = self.add_lin(spk_vec)
        if self.use_FiLM:
            mul = self.mul_lin(spk_vec)
            return mul.unsqueeze(-1)*squeezed + bias.unsqueeze(-1)
        else:
            return squeezed + bias.unsqueeze(-1)

    def forward(self, x, spk_vec):
        """ Input shape [batch, feats, seq]"""

        conditioned = self.apply_conditioning(spk_vec, self.depth_conv1d(x))

        return self.out(conditioned)


class SpeakerStack(nn.Module):
    # basically this is plain conv-tasnet remove this in future releases

    def __init__(self, n_src, in_dim, embed_dim, n_blocks=14, n_repeats=1,
                 kernel_size=3,
                 norm_type="gLN"):
        
        super(SpeakerStack, self).__init__()
        self.embed_dim = embed_dim
        self.n_src = n_src
        self.in_dim = in_dim
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        
        # Succession of Conv1DBlock with exponentially increasing dilation.
        in_channel = self.in_dim
        self.TCN = nn.ModuleList()
        for r in range(n_repeats):
            for x in range(n_blocks):
                padding = (kernel_size - 1) * 2 ** x // 2
                
                self.TCN.append(
                    Conv1DBlock(
                        in_channel, embed_dim, \
                        kernel_size, padding=padding, \
                        dilation=2 ** x, norm_type=norm_type
                    )
                )
                
                in_channel = embed_dim
                                            
        mask_conv = nn.Conv1d(embed_dim, n_src * embed_dim, 1)
        self.mask_net = nn.Sequential(mask_conv)

    def forward(self, mixture_w):
        """
            Args:
                mixture_w (:class:`torch.Tensor`): Tensor of shape
                    [batch, n_filters, n_frames]

            Returns:
                :class:`torch.Tensor`:
                    # estimated mask of shape [batch, n_src, n_filters, n_frames]
                    estimated mask of shape [n_src * batch, n_filters, n_frames]
        """
        batch, _, n_frames = mixture_w.size()
        output = mixture_w
        # output = mixture_w.unsqueeze(1)
        for i in range(len(self.TCN)):
            if i == 0:
                output = self.TCN[i](output)
            else:
                residual = self.TCN[i](output)
                output = output + residual
        emb = self.mask_net(output)

        emb = emb.view(self.n_src * batch, self.embed_dim, n_frames)
        emb = emb / torch.sqrt(torch.sum(emb**2, 2, keepdim=True))

        return emb

class Linear(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(Linear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, features, **kwargs):
        predicted = self.linear(features)
        return predicted, {}
