from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from diffusers.pipelines.flux.pipeline_flux import (
    FluxPipelineOutput,
    calculate_shift,
    retrieve_timesteps,
)
from PIL import Image
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
    T5TokenizerFast,
)

from pipeline.common import get_module_having_attn_processor
from processor import FluxAttnProcessor2_0WithMemory


class RFImageEditingFluxPipeline(FluxPipeline):
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

    @torch.inference_mode()
    def encode_img(self, img: Union[torch.Tensor, np.ndarray, Image.Image, str], dtype):
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        if isinstance(img, Image.Image):
            img = np.array(img)

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

        return latents, image_ids, new_h, new_w

    def add_processor(self, after_layer: Optional[int] = None):
        self.transformer.set_attn_processor(
            get_module_having_attn_processor(
                self.transformer, FluxAttnProcessor2_0WithMemory, after_layer=after_layer
            )
        )
        self.initialize_processor = True
        return self.transformer.attn_processors

    def denoise(
        self,
        latents: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        latent_image_ids: torch.Tensor,
        guidance_scale: float,
        inject_step: int,
        num_inference_steps: int,
        device: torch.device,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        with_second_order: bool = False,
        inverse: bool = False,
    ):
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, _ = retrieve_timesteps(
            self.scheduler,
            num_inference_steps + 1,
            device,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - 1 - num_inference_steps * self.scheduler.order, 0)
        using_inject_list = [True] * inject_step + [False] * (len(timesteps[:-1]) - inject_step)
        if inverse:
            timesteps = torch.flip(timesteps, [0])
            using_inject_list = using_inject_list[::-1]

        dtype = latents.dtype

        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for step_idx, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:], strict=True)):
                sigma_curr = t_curr / self.scheduler.config.num_train_timesteps
                sigma_prev = t_prev / self.scheduler.config.num_train_timesteps

                joint_attention_kwargs["second_order"] = False
                joint_attention_kwargs["inject"] = using_inject_list[step_idx]
                joint_attention_kwargs["timestep"] = int(t_prev.item()) if inverse else int(t_curr.item())

                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=sigma_curr.expand(latents.shape[0]).to(dtype),
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=False,
                )[0].float()

                if (step_idx == len(timesteps) - 2) or not with_second_order:
                    latents = latents + (sigma_prev - sigma_curr) * noise_pred
                    latents = latents.to(dtype)
                else:
                    mid_sample = latents + (sigma_prev - sigma_curr) / 2 * noise_pred
                    mid_sample = mid_sample.to(dtype)

                    sigma_mid = torch.full(
                        (mid_sample.shape[0],),
                        (sigma_curr + (sigma_prev - sigma_curr) / 2),
                        dtype=mid_sample.dtype,
                        device=mid_sample.device,
                    )
                    joint_attention_kwargs["second_order"] = True
                    mid_noise_pred = self.transformer(
                        hidden_states=mid_sample,
                        timestep=sigma_mid.expand(latents.shape[0]).to(latents.dtype),
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=joint_attention_kwargs,
                        return_dict=False,
                    )[0].float()

                    first_order = (mid_noise_pred - noise_pred) / ((sigma_prev - sigma_curr) / 2)
                    latents = (
                        latents
                        + (sigma_prev - sigma_curr) * noise_pred
                        + 0.5 * (sigma_prev - sigma_curr) ** 2 * first_order
                    )
                    latents = latents.to(dtype)

                # call the callback, if provided
                if step_idx == len(timesteps) - 2 or (
                    (step_idx + 1) > num_warmup_steps and (step_idx + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        return latents

    @torch.inference_mode()
    def __call__(
        self,
        source_img: Union[np.ndarray, torch.Tensor],
        source_prompt: Union[str, List[str]],
        target_prompt: Union[str, List[str]],
        source_prompt_2: Optional[Union[str, List[str]]] = None,
        target_prompt_2: Optional[Union[str, List[str]]] = None,
        inject_step: int = 6,
        num_inference_steps: int = 28,
        guidance_scale: float = 2,
        num_images_per_prompt: Optional[int] = 1,
        latents: Optional[torch.FloatTensor] = None,
        source_prompt_embeds: Optional[torch.FloatTensor] = None,
        source_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        target_prompt_embeds: Optional[torch.FloatTensor] = None,
        target_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        max_sequence_length: int = 512,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        with_second_order: bool = True,
    ):
        if not self.initialize_processor:
            raise ValueError("Please call `add_processor` before running the pipeline.")

        if joint_attention_kwargs is None:
            joint_attention_kwargs = {}

        device = self._execution_device
        (
            source_prompt_embeds,
            source_pooled_prompt_embeds,
            source_text_ids,
        ) = self.encode_prompt(
            prompt=source_prompt,
            prompt_2=source_prompt_2,
            prompt_embeds=source_prompt_embeds,
            pooled_prompt_embeds=source_pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        source_img_latents, source_latent_image_ids, height, width = self.encode_img(
            source_img, source_prompt_embeds.dtype
        )
        inverse_latents = self.denoise(
            latents=source_img_latents,
            pooled_prompt_embeds=source_pooled_prompt_embeds,
            prompt_embeds=source_prompt_embeds,
            text_ids=source_text_ids,
            latent_image_ids=source_latent_image_ids,
            guidance_scale=1,
            inject_step=inject_step,
            num_inference_steps=num_inference_steps,
            device=device,
            joint_attention_kwargs=joint_attention_kwargs,
            inverse=True,
            with_second_order=with_second_order,
        )

        (
            target_prompt_embeds,
            target_pooled_prompt_embeds,
            target_text_ids,
        ) = self.encode_prompt(
            prompt=target_prompt,
            prompt_2=target_prompt_2,
            prompt_embeds=target_prompt_embeds,
            pooled_prompt_embeds=target_pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        latents = self.denoise(
            latents=inverse_latents,
            pooled_prompt_embeds=target_pooled_prompt_embeds,
            prompt_embeds=target_prompt_embeds,
            text_ids=target_text_ids,
            latent_image_ids=source_latent_image_ids,  # reuse the same image ids
            guidance_scale=guidance_scale,
            inject_step=inject_step,
            num_inference_steps=num_inference_steps,
            device=device,
            joint_attention_kwargs=joint_attention_kwargs,
            inverse=False,
            with_second_order=with_second_order,
        )

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)
