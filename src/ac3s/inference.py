import torch

def get_conditioning_scale(
            model,
            modulator,
            visual_prompt,
            text_prompt,
            negative_text_prompt,
            mean,
            std,
            height,
            width,
            num_inference_steps,
            padding,
            min_scale,
            max_scale,
            device,

        ):
    # Set up the scheduler
    scheduler = model.scheduler
    scheduler.set_timesteps(num_inference_steps, device=model.device)
    timesteps = scheduler.timesteps

    # Sample a random latent representation
    batch_size = visual_prompt.shape[0]
    shape = (batch_size, model.unet.config.in_channels, height//model.vae_scale_factor, width//model.vae_scale_factor)
    latents = torch.randn(shape, device=model.device) * scheduler.init_noise_sigma

    # Prepare prompts
    if negative_text_prompt:
      uncond_prompt = negative_text_prompt
    else:
      uncond_prompt = ""
    cond_prompt = text_prompt

    # Single denoising pass
    t = timesteps[0]
    latent_model_input = model.scheduler.scale_model_input(latents, t)

    uncond_noise_pred, _ = model(
        timestep=t.unsqueeze(0),
        latent_model_input=latent_model_input,
        visual_prompt=visual_prompt,
        text_prompt=uncond_prompt,
        conditioning_scale=1.0,
        collect_feats=False,
        visual_modulator_output=None,
    )

    cond_noise_pred, cond_feats = model(
        timestep=t.unsqueeze(0),
        latent_model_input=latent_model_input,
        visual_prompt=visual_prompt,
        text_prompt=cond_prompt,
        conditioning_scale=1.0,
        collect_feats=True,
        visual_modulator_output=None,
    )

    # Extract modulator input features
    CN_features = cond_feats[0]
    SD_features = cond_feats[1]

    batch_size = CN_features[0].shape[0]
    modulator_input = []
    for idx in range(batch_size):
        features = []
        for layer in CN_features:
            norm = layer[idx, :, :, :].norm().unsqueeze(0)
            features.append(norm.cpu())
        for layer in SD_features:
            norm = layer[idx, :, :, :].norm().unsqueeze(0)
            features.append(norm.cpu())
        modulator_input.append(torch.cat(features))
    
    modulator_input = torch.stack(modulator_input)

    # Normalize features
    modulator_input = (modulator_input-mean) / std
    
    # Forward pass of modulator
    with torch.no_grad():
        modulator_input = modulator_input.to(device)
        conditioning_scale = modulator(modulator_input)

    #Apply constraints
    conditioning_scale = conditioning_scale + padding
    conditioning_scale = torch.clamp(conditioning_scale, min_scale, max_scale)
        
    return conditioning_scale