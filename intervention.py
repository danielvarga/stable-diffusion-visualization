import numpy as np
import torch
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, LMSDiscreteScheduler
from torch import autocast
from tqdm.auto import tqdm
from PIL import Image

from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline



torch_device = "cuda" if torch.cuda.is_available() else "cpu"

def simple_generation():
    # pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    # pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)

    pipe = pipe.to(torch_device)

    prompt = "a photograph of an astronaut riding a horse"

    generator = torch.Generator(torch_device).manual_seed(1024)

    image = pipe(prompt, generator=generator).images[0]

    # Now to display an image you can either save it such as:
    image.save(f"astronaut_rides_horse.png")


# simple_generation() ; exit()



# 1. Load the autoencoder model which will be used to decode the latents into image space.
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

# 2. Load the tokenizer and text encoder to tokenize and encode the text.
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# 3. The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

scheduler = LMSDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")


vae = vae.to(torch_device)
text_encoder = text_encoder.to(torch_device)
unet = unet.to(torch_device)




intervention_output = None

def generation(text_embeddings, unet, output_filename):
    batch_size = 1

    height = 512                        # default height of Stable Diffusion
    width = 512                         # default width of Stable Diffusion

    num_inference_steps = 100            # Number of denoising steps

    guidance_scale = 7.5                # Scale for classifier-free guidance

    generator = torch.manual_seed(32)   # Seed generator to create the inital latent noise

    scheduler.set_timesteps(num_inference_steps)

    latents = torch.randn(
      (batch_size, unet.config.in_channels, height // 8, width // 8),
      generator=generator,
    )
    latents = latents.to(torch_device)

    latents = latents * scheduler.init_noise_sigma

    for iteration_step, t in enumerate(tqdm(scheduler.timesteps)):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)

        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        if iteration_step in (10, num_inference_steps - 10):
            print("iteration_step", iteration_step)
            print(intervention_output.shape)
            for text_i in (0, 1):
                for token_i in (0, 1):
                    vec = intervention_output[text_i, token_i, :].cpu().numpy()
                    h = np.histogram(np.log(np.abs(vec)))
                    # print("uncond" if text_i == 0 else "cond", f"neuron #{token_i}", h[0], np.exp(h[1]))
                    print("uncond" if text_i == 0 else "cond", f"neuron #{token_i}", vec.mean(), vec.std())

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents

    with torch.no_grad():
        image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")

    pil_images = [Image.fromarray(image) for image in images]
    pil_images[0].save(output_filename)


def hook_fn(module, input, output):
    global intervention_output
    intervention_output = output


# intervention
def intervention(prefix, tensor, positions, bias_shifts):
    for position in positions:
        for bias_shift in bias_shifts:
            tensor[..., position] += bias_shift
            filename = f"{prefix}_{position}_{bias_shift}.png"
            generation(text_embeddings, unet, filename)
            print(filename, "saved")
            tensor[..., position] -= bias_shift


def prompt_to_tensor(prompt):
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

    batch_size = 1

    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    return text_embeddings



'''
hook = unet._modules['mid_block'].attentions[0].transformer_blocks[0].attn1.register_forward_hook(hook_fn)

# tensor = unet._modules['down_blocks'][2].resnets[0].conv2.bias.data
tensor = unet._modules['mid_block'].attentions[0].transformer_blocks[0].attn1.to_v.weight
'''

layer = unet._modules['mid_block'].attentions[0].transformer_blocks[0].ff.net[2]
layer = unet._modules['mid_block'].attentions[0].proj_out

hook = layer.register_forward_hook(hook_fn)

tensor = layer.bias.data


prompts = [
    "inspiring beautiful girl walking among crowd of celebrating people in a beautiful futuristic city on 1st of May by Edward Hopper and Dan Mumford and WLOP, Unreal Engine 5, Lumen, Nanite",
    "highly detailed portrait of emily blunt, in the walking dead, stephen bliss, unreal engine, fantasy art by greg rutkowski, loish, rhads, ferdinand knab, makoto shinkai and lois van baarle, ilya kuvshinov, rossdraws, tom bagshaw, global illumination, radiant light, detailed and intricate environment",
    "a portrait of donald trump with gorgeous pastel balayage hairstyle, contemplating deep philosophical quandaries, as seen on artgerm, octane render, in the style of alphonse mucha, ultra realistic, highly detailed, 8 k, ",
    "juicy vegan hamburger topped with guacamole and fried onion and a vegan fried egg, crispy buns, 8 k resolution, professional food photography, studio lighting, sharp focus, hyper - detailed ",
    "portrait, 30 years old man :: red hair ponytail :: burned face, grimy, rough, shirtless :: high detail, digital art, RPG, concept art, illustration",
    "cute cartoon little tractor dragging the russian tank on the sunflower field by goro fujita and simon stalenhag and wes anderson and alex andreev and chiho aoshima and beeple and banksy and kandinsky and magritte and basquiat and picasso, 8 k, trending on artstation, hyper detailed, cinematic ",
    "a beautiful watercolour on 3 0 0 gsm paper of a school of mackerel, 8 k, frostbite 3 engine, cryengine, dof, trending on artstation, digital art, crepuscular ray ",
    "rosie jetson, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by artgerm and greg rutkowski and alphonse mucha and william - adolphe bouguereau ",
    "cute calico kitten looking out of the window on a [ [ [ [ beautiful ] ] ] ] summer day, storybook art, detailed, cute, profile shot, featured on artstationg, gorgeous!!! ",
    "portrait shot of a steampunk robot bug, unreal engine realistic render, 8 k, micro detail, intricate, elegant, highly detailed, centered, digital painting, artstation, smooth, sharp focus, illustration, artgerm, tomasz alen kopera, peter mohrbacher, donato giancola, joseph christian leyendecker, wlop, boris vallejo ",
]

for prompt_index, prompt in enumerate(prompts):
    text_embeddings = prompt_to_tensor(prompt)

    with torch.no_grad():
        prefix = f"j_boost_{prompt_index}"
        intervention(prefix, tensor, positions=range(30), bias_shifts=np.around(np.linspace(-1000, 1000, 11)).astype(int))

exit()




# hook = unet._modules['down_blocks'][0].resnets[0].conv1.register_forward_hook(hook_fn)
hook = unet._modules['conv_in'].register_forward_hook(hook_fn)

with torch.no_grad():
    noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
print(hooked_output3.shape)

for i in range(10):
    plt.imshow(hooked_output3[0, i].cpu().detach().numpy())
    plt.savefig(f"neuron_{i:03}.png")



input_data = torch.randn(
  (2, 4, height // 8, width // 8),
  generator=generator, requires_grad=True
)

# input_data = latent_model_input.clone()
print("input_data.shape", input_data.shape)

optimizer = torch.optim.SGD([input_data], lr=0.01)
hooked_output_x = None
for i in range(100):
    print(f"iteration {i} start")
    optimizer.zero_grad()
    def hook_fn(module, input, output):
        global hooked_output_x
        hooked_output_x = output
        print("BEEN HERE", hooked_output_x.shape)
    hook = unet._modules['conv_in'].register_forward_hook(hook_fn)
    noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
    NEURON_INDEX = 35
    loss = - hooked_output_x[0][NEURON_INDEX].mean(axis=(-2, -1))
    print(loss.shape, loss)
    loss.backward()
    optimizer.step()
    hook.remove()
    print(f"iteration {i} end")


print("input_data", input_data.shape)

input_data = input_data.to(torch_device)


with torch.no_grad():
  image = vae.decode(input_data).sample

print("image.shape", image.shape)


image = (image / 2 + 0.5).clamp(0, 1)
image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
images = (image * 255).round().astype("uint8")

from PIL import Image

pil_images = [Image.fromarray(image) for image in images]
for i, pil_image in enumerate(pil_images):
     pil_image.save(f"backprop_{i:03}.png")
