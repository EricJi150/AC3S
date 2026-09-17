import torch
import torch.nn as nn

from diffusers.schedulers import DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, ControlNetModel

from stable_diffusion.unet import UNet2DConditionModelAC3S

class SDControlNetModel(nn.Module):
    def __init__(
            self, 
            config_path =  "runwayml/stable-diffusion-v1-5",
            edge_controlnet_path = "lllyasviel/sd-controlnet-canny",
            device = "cuda"
    ):
        super().__init__()

        # Initialize pre-trained models
        self.controlnet = ControlNetModel.from_pretrained(edge_controlnet_path)
        self.vae = AutoencoderKL.from_pretrained(config_path, subfolder='vae')
        self.unet = UNet2DConditionModelAC3S.from_pretrained(config_path, subfolder='unet')
        self.scheduler = DDIMScheduler.from_pretrained(config_path, subfolder='scheduler')
        self.tokenizer = CLIPTokenizer.from_pretrained(config_path, subfolder='tokenizer')
        self.text_encoder = CLIPTextModel.from_pretrained(config_path, subfolder='text_encoder')
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)

        for model in [self.controlnet, self.vae, self.unet, self.text_encoder]:
            model.requires_grad_(False)

        self.device = device
        self.to(device)

    def forward(
            self,
            timestep,
            latent_model_input,
            visual_prompt,
            text_prompt,
            conditioning_scale,
            collect_feats,
            visual_modulator_output
        ):
        batch_size = visual_prompt.shape[0]

        # Convert prompts -> input_ids -> embeddings
        if isinstance(text_prompt, str):
            text_prompt = [text_prompt] * batch_size
        prompt_input_ids = self.tokenizer(text_prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids
        prompt_input_embeds = self.text_encoder(prompt_input_ids.to(self.device))[0]
        prompt_embeds = prompt_input_embeds

        # Produce the controlnet features
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            timestep=timestep,
            sample=latent_model_input,
            controlnet_cond=visual_prompt,
            conditioning_scale=conditioning_scale,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
            )

        # Scale ControlNet features by visual modulator output if provided (conditioning_scale = 1.0 at inference time)
        if visual_modulator_output is not None:
            down_block_res_samples = [sample * visual_modulator_output[:, :, None, None] for sample in down_block_res_samples]
            mid_block_res_sample = mid_block_res_sample * visual_modulator_output[:, :, None, None]

        # Collect ControlNet features for modulator
        CN_feats = []
        if collect_feats:
            for sample in down_block_res_samples:
                CN_feats.append(sample.detach().cpu())
            CN_feats.append(mid_block_res_sample.detach().cpu())

        # Predict the noise to be removed
        noise_pred, SD_feats = self.unet(
            timestep=timestep,
            sample=latent_model_input,
            encoder_hidden_states=prompt_embeds,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            collect_feats=collect_feats
        )
        noise_pred = noise_pred[0]

        return noise_pred, (CN_feats, SD_feats)