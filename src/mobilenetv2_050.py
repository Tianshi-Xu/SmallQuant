import sys
import torch.nn as nn

sys.path.append("/home/mengli/third-party/pytorch-image-models")

from functools import partial

from timm.models.layers import create_conv2d
from timm.models.registry import register_model
from timm.models.efficientnet import _create_effnet
from timm.models.efficientnet_builder import (
    decode_arch_def,
    resolve_bn_args,
    resolve_act_layer,
    round_channels,
)


__all__ = ['mobilenetv2_050']


def _gen_mobilenet_v2(
        variant, 
        channel_multiplier=1.0, depth_multiplier=1.0, fix_stem_head=False, pretrained=False, **kwargs):
    """ Generate MobileNet-V2 network
    Ref impl: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py
    Paper: https://arxiv.org/abs/1801.04381
    """
    arch_def = [
        ['ds_r1_k3_s1_c16'],
        ['ir_r2_k3_s2_e6_c24'],
        ['ir_r3_k3_s2_e6_c32'],
        ['ir_r4_k3_s2_e6_c64'],
        ['ir_r3_k3_s1_e6_c96'],
        ['ir_r3_k3_s2_e6_c160'],
        ['ir_r1_k3_s1_e6_c320'],
    ]
    round_chs_fn = partial(round_channels, multiplier=channel_multiplier)
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier=depth_multiplier, fix_first_last=fix_stem_head),
        num_features=1280 if fix_stem_head else max(1280, round_chs_fn(1280)),
        stem_size=32,
        fix_stem=fix_stem_head,
        round_chs_fn=round_chs_fn,
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'prelu'),
        **kwargs
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model

@register_model
def mobilenetv2_050(pretrained=False, **kwargs):
    """ MobileNet V2 w/ 0.5 channel multiplier """
    model = _gen_mobilenet_v2('mobilenetv2_050', 0.5, pretrained=pretrained, **kwargs)
    return model