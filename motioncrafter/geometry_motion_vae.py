# Geometry Motion VAE Module
# Provides unified autoencoder for encoding/decoding point maps and scene flow

from typing import Dict, Tuple, Union, Optional

import torch
import torch.nn as nn
from diffusers.models import AutoencoderKL
from copy import deepcopy
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.autoencoders.vae import (
    DiagonalGaussianDistribution,
    Encoder
)


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

        
class UnifyAutoencoderKL(AutoencoderKL):
    # Unified AutoencoderKL for joint encoding/decoding of point maps and flow
    # Supports separate encode/decode branches for geometric and motion data
    @register_to_config
    def __init__(self):
        # Initialize unified VAE with dual encoder/decoder paths
        # Loads pretrained weights from Stable Diffusion VAE
        # Load pretrained VAE from Stable Diffusion
        pretrained_vae = AutoencoderKL.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            cache_dir="cache",
            subfolder="vae",
            revision=None,
            variant="fp16"
        )
        super().__init__(**pretrained_vae.config)

        # Copy encoder and decoder for second branch
        self.encoder_2 = deepcopy(pretrained_vae.encoder)
        self.quant_conv_2 = deepcopy(pretrained_vae.quant_conv)
        self.post_quant_conv_2 = deepcopy(pretrained_vae.post_quant_conv)
        self.decoder_2 = deepcopy(pretrained_vae.decoder)

        # Expand decoder input channels from 4 to 8 for concatenated input
        # Replicate weights for new channels
        in_channels = self.decoder_2.conv_in.in_channels
        new_conv_in = nn.Conv2d(
            in_channels * 2,
            self.decoder_2.conv_in.out_channels,
            kernel_size=self.decoder_2.conv_in.kernel_size,
            stride=self.decoder_2.conv_in.stride,
            padding=self.decoder_2.conv_in.padding,
        )
        with torch.no_grad():
            # Initialize new conv layer by duplicating original weights
            new_conv_in.weight[:, :in_channels, :, :] = (
                self.decoder_2.conv_in.weight
            )
            new_conv_in.weight[:, in_channels:, :, :] = (
                self.decoder_2.conv_in.weight
            )
            new_conv_in.bias = self.decoder_2.conv_in.bias
        self.decoder_2.conv_in = new_conv_in

        # Release memory from temporary pretrained model
        del pretrained_vae

    @apply_forward_hook
    def encode(
        self,
        x: torch.Tensor,
    ) -> Union[DiagonalGaussianDistribution, DiagonalGaussianDistribution]:
        # Encode first branch (point map) to latent distribution
        with torch.no_grad():
            h_1 = self.encoder(x)
            moments_1 = self.quant_conv(h_1)
            posterior_1 = DiagonalGaussianDistribution(moments_1)

        return posterior_1

    @apply_forward_hook
    def encode_2(
        self,
        x: torch.Tensor,
    ) -> Union[DiagonalGaussianDistribution, DiagonalGaussianDistribution]:
        # Encode second branch (scene flow) to latent distribution
        # Scale input by 10 for better numerical stability
        x = x * 10
        h_2 = self.encoder_2(x)
        moments_2 = self.quant_conv_2(h_2)
        posterior_2 = DiagonalGaussianDistribution(moments_2)
        return posterior_2

    @apply_forward_hook
    def decode(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        # Decode first branch latent to point map
        with torch.no_grad():
            z = self.post_quant_conv(z)
            decoded_1 = self.decoder(z)
        
        return decoded_1
    
    @apply_forward_hook
    def decode_2(
        self,
        z: torch.Tensor,
        latent_1: torch.Tensor
    ) -> torch.Tensor:
        # Decode second branch latent to scene flow
        # Concatenates with first branch latent for combined decoding
        z = self.post_quant_conv_2(z)
        # Concatenate along channel dimension for joint decoding
        z = torch.cat([z, latent_1], dim=1)
        decoded_2 = self.decoder_2(z)
        # Inverse scaling from encode_2
        decoded_2 = decoded_2 / 10
        return decoded_2
