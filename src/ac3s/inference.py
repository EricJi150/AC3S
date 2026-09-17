def get_conditioning_scale(
            model,
            mlp,
            text_prompt,
            visual_prompt,
            negative_prompt = None,
            height = 512,
            width = 512,
            num_inference_steps = 20,
            padding = 0.2,
            min_scale = 0.3,
            max_scale = 1.0,
            
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
    if negative_prompt:
      uncond_prompt = negative_prompt
    else:
      uncond_prompt = ""
    cond_prompt = text_prompt

    # Auto-regressive denoising loop
    t = timesteps[0]
    latent_model_input = model.scheduler.scale_model_input(latents, t)


    uncond_noise_pred, _ = model(
        t.unsqueeze(0), 
        latent_model_input, 
        visual_prompt,
        1.0,
        uncond_prompt,
        False
    )
    cond_noise_pred, intermediate_features = model(
        t.unsqueeze(0), 
        latent_model_input, 
        visual_prompt,
        1.0,
        cond_prompt,
        True
    )

    # Extract modulator input features
    CN_features = intermediate_features[0]
    SD_features = intermediate_features[1]
    edge_map = visual_prompt

    assert CN_features[0].shape[0] == SD_features[0].shape[0] == len(edge_map)
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
        norm = edge_map[idx].norm().unsqueeze(0)
        features.append(norm.cpu())
        modulator_input.append(torch.cat(features))
    
    modulator_input = torch.stack(modulator_input)

    # Normalize features
    mean = torch.tensor([95.3262, 314.6048, 664.1411, 523.2632, 547.3312, 603.0335, 428.0241, 440.5691, 532.0027, 460.5557, 502.3954, 393.0252, 899.3889, 
            409.6641, 686.8041, 474.8352, 443.4046, 529.2507, 511.6783, 393.0137, 602.4334, 544.5546, 529.7780, 534.2770, 579.5325, 880.6142, 
            95.6728])
    std = torch.tensor([7.2452,  22.9817, 132.4456,  92.5421,  89.7678, 132.8370, 87.0174, 63.7730,  79.4488,  54.4137,  70.4102,  53.2295, 108.5898, 
            2.1652, 4.8736, 3.8539, 4.1850, 4.0411, 64.3496, 8.2243, 9.1612, 19.9286,  37.4086,  33.6463,  30.9399, 51.0437,  
            37.5874])

    modulator_input = (modulator_input-mean) / std
    
    # Forward pass of modulator
    with torch.no_grad():
        modulator_input = modulator_input.to("cuda")
        conditioning_scale = mlp(modulator_input)

    #Apply constraints
    conditioning_scale = conditioning_scale + padding
    conditioning_scale = torch.clamp(conditioning_scale, 0.3, 1.0)
        
    return conditioning_scale