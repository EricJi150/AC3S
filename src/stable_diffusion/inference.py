import torch

def generate(
            model,
            visual_prompt,
            text_prompt,
            negative_text_prompt,
            conditioning_scale,
            guidance_scale,
            height,
            width,
            timesteps,
            collect_feats,
            visual_modulator_output
        ):      
    # Initialize the scheduler
    scheduler = model.scheduler
    scheduler.set_timesteps(timesteps, device=model.device)
    timesteps = scheduler.timesteps

    # Sample a random latent representation
    batch_size = visual_prompt.shape[0]
    shape = (batch_size, model.unet.config.in_channels, height//model.vae_scale_factor, width//model.vae_scale_factor)
    latents = torch.randn(shape, device=model.device) * scheduler.init_noise_sigma

    # Prepare text prompts
    if negative_text_prompt:
        uncond_text_prompt = negative_text_prompt
    else:
        uncond_text_prompt = [""] * batch_size
    cond_text_prompt = text_prompt

    # Auto-regressive denoising loop
    feats = None
    for idx, t in enumerate(timesteps):
        latent_model_input = model.scheduler.scale_model_input(latents, t)

        uncond_noise_pred, _ = model(
                timestep=t.unsqueeze(0), 
                latent_model_input=latent_model_input, 
                visual_prompt=visual_prompt,
                text_prompt=uncond_text_prompt,
                conditioning_scale=conditioning_scale,
                collect_feats=collect_feats,
                visual_modulator_output=visual_modulator_output
        )
        
        cond_noise_pred, cond_feats = model(
                timestep=t.unsqueeze(0), 
                latent_model_input=latent_model_input, 
                visual_prompt=visual_prompt,
                text_prompt=cond_text_prompt,
                conditioning_scale=conditioning_scale,
                collect_feats=collect_feats,
                visual_modulator_output=visual_modulator_output
        )
        
        # Collect modulator input features
        if collect_feats and idx == 0:
            feats = cond_feats

        # CFG implementation
        noise_pred = uncond_noise_pred + guidance_scale * (cond_noise_pred - uncond_noise_pred)   

        # Remove noise from latent representation
        latents = model.scheduler.step(noise_pred, t, latents).prev_sample

    # Decode the latent representation to an image
    image = model.vae.decode(latents / model.vae.config.scaling_factor, return_dict=False)[0]
    image = model.image_processor.postprocess(image, output_type="pt")

    if collect_feats:
        return image, feats
    else:
        return image
