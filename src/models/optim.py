from typing import Dict, Iterable, List, Tuple

import torch


def weight_decay_param_groups(named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
                              no_weight_decay: Iterable[str] = ()) -> List[Dict]:
    """Split the trainable parameters into a weight-decay and a no-weight-decay group.

    Biases, 1-D parameters (LayerNorm, BatchNorm) and the parameters named in `no_weight_decay`
    (matched on the full name or as a suffix, e.g. ``pos_embed`` matches
    ``image_encoder.pos_embed``) are not decayed; all other trainable parameters (weight matrices,
    conv kernels, tokens) are. Every trainable parameter ends up in exactly one group; frozen
    parameters are left out.
    """
    no_weight_decay = set(no_weight_decay)
    decay, no_decay = [], []
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        name = name.replace('_orig_mod.', '')  # names inside torch.compile'd modules
        skip = name in no_weight_decay or any(name.endswith('.' + s) for s in no_weight_decay)
        if name.endswith('bias') or param.ndim <= 1 or skip:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': decay}, {'params': no_decay, 'weight_decay': 0.}]
