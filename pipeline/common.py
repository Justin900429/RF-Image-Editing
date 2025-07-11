from typing import Any, Callable, Dict, Optional, Union

import torch
from diffusers.models.attention_processor import FluxAttnProcessor2_0


def get_module_having_attn_processor(
    module: torch.nn.Module,
    target_processor: Callable,
    after_layer: Optional[int] = None,
    **target_processor_kwargs,
) -> Dict[str, Union[FluxAttnProcessor2_0, Callable]]:
    def _get_module_having_attn_processor_driver(
        name: str, module: torch.nn.Module, res: Dict[str, Any], after_layer: Optional[int] = None
    ):
        if hasattr(module, "set_processor"):
            added = True
            if after_layer is not None:
                block_idx = name.rsplit(".", 2)[-2]
                if int(block_idx) <= after_layer:
                    added = False
            if added:
                res[f"{name}.processor"] = target_processor(**target_processor_kwargs)
            else:
                res[f"{name}.processor"] = FluxAttnProcessor2_0()

        for sub_name, child in module.named_children():
            _get_module_having_attn_processor_driver(f"{name}.{sub_name}", child, res, after_layer)

    res = {}
    for sub_name, child in module.named_children():
        _get_module_having_attn_processor_driver(f"{sub_name}", child, res, after_layer)
    return res
