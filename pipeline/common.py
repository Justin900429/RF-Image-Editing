from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, FluxPipeline
from diffusers.models import AutoencoderKL, FluxTransformer2DModel
from diffusers.models.transformers.transformer_flux import FluxAttnProcessor
from PIL import Image
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
    T5TokenizerFast,
)

from processor import FluxAttnProcessorWithMemory


def get_module_having_attn_processor(
    transformer: torch.nn.Module,
    target_processor: Callable,
    before_layer: Optional[int] = None,
    after_layer: Optional[int] = None,
    filter_name: list[str] | None = None,
    **target_processor_kwargs,
) -> Dict[str, Union[FluxAttnProcessor, Callable]]:
    def _get_module_having_attn_processor_driver(
        name: str, module: torch.nn.Module, res: Dict[str, Any], after_layer: Optional[int] = None
    ):
        if hasattr(module, "set_processor"):
            block_type = name.split(".", 1)[0]
            added = any(block_type == filter_name for filter_name in filter_name)
            block_idx = int(name.rsplit(".", 2)[-2])
            target_processor_kwargs["block_idx"] = block_idx
            if added and after_layer is not None:
                if block_idx < after_layer:
                    added = False
            if added and before_layer is not None:
                if block_idx > before_layer:
                    added = False
            if added:
                res[f"{name}.processor"] = target_processor(**target_processor_kwargs)
            else:
                res[f"{name}.processor"] = FluxAttnProcessor()

        for sub_name, child in module.named_children():
            _get_module_having_attn_processor_driver(f"{name}.{sub_name}", child, res, after_layer)

    res = {}
    for sub_name, child in transformer.named_children():
        _get_module_having_attn_processor_driver(f"{sub_name}", child, res, after_layer)
    return res


class ProcessorMixin:
    def add_processor(
        self,
        after_layer: Optional[int] = None,
        before_layer: Optional[int] = None,
        filter_name: str | list[str] | None = None,
        target_processor: Callable = FluxAttnProcessorWithMemory,
        **kwargs,
    ):
        if isinstance(filter_name, str):
            filter_name = [filter_name]

        self.transformer.set_attn_processor(
            get_module_having_attn_processor(
                self.transformer,
                target_processor,
                before_layer=before_layer,
                after_layer=after_layer,
                filter_name=filter_name,
                **kwargs,
            )
        )
        self.initialize_processor = True
        self.processors = self.transformer.attn_processors


class RFEditingFluxPipeline(FluxPipeline, ProcessorMixin):
    _callback_tensor_inputs = ["latents", "inverse"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
    ):
        super().__init__(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        self.initialize_processor = False
        self.processors: dict[str, Any] = self.transformer.attn_processors

    @torch.inference_mode()
    def encode_img(
        self,
        img: Union[torch.Tensor, np.ndarray, Image.Image, str],
        dtype: torch.dtype,
        target_size: tuple[int, int] | None = None,  # (width, height)
    ):
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        if isinstance(img, Image.Image):
            img = np.array(img)

        ori_height, ori_width = img.shape[:2]
        if target_size is not None:
            img = self.image_processor.resize(img[None], height=target_size[1], width=target_size[0])[0]

        shape = img.shape
        new_h = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
        new_w = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16

        img = img[:new_h, :new_w, :]
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
        img = img.to(device=self._execution_device, dtype=dtype)
        latents = self.vae.encode(img).latent_dist.mode()

        batch_size, channels, height, width = latents.shape
        latents = self._pack_latents(latents, batch_size, channels, height, width)
        image_ids = self._prepare_latent_image_ids(
            batch_size, height // 2, width // 2, self._execution_device, dtype
        )
        latents = self.vae.config.scaling_factor * (latents - self.vae.config.shift_factor)

        return latents, image_ids, new_h, new_w, ori_height, ori_width
