import torch
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline


torch_device = "cuda" if torch.cuda.is_available() else "cpu"

'''
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)

pipe = pipe.to("cuda")

prompt = "a photograph of an astronaut riding a horse"

generator = torch.Generator("cuda").manual_seed(1024)

image = pipe(prompt, generator=generator).images[0]

# Now to display an image you can either save it such as:
image.save(f"astronaut_rides_horse.png")
'''



from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler

# 1. Load the autoencoder model which will be used to decode the latents into image space.
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

# 2. Load the tokenizer and text encoder to tokenize and encode the text.
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# 3. The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")


from diffusers import LMSDiscreteScheduler

scheduler = LMSDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")


vae = vae.to(torch_device)
text_encoder = text_encoder.to(torch_device)
unet = unet.to(torch_device)


unet._modules['down_blocks'][2].resnets[0].conv2.bias.data[0] -= 70


prompt = ["Portrait of a beautiful girl. national geographic cover photo."]

height = 512                        # default height of Stable Diffusion
width = 512                         # default width of Stable Diffusion

num_inference_steps = 100            # Number of denoising steps

guidance_scale = 7.5                # Scale for classifier-free guidance

generator = torch.manual_seed(32)   # Seed generator to create the inital latent noise

batch_size = 1

text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")


with torch.no_grad():
  text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]


max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer(
    [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
)
with torch.no_grad():
  uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

latents = torch.randn(
  (batch_size, unet.config.in_channels, height // 8, width // 8),
  generator=generator,
)
latents = latents.to(torch_device)

scheduler.set_timesteps(num_inference_steps)

latents = latents * scheduler.init_noise_sigma

from torch import autocast
from tqdm.auto import tqdm


for t in tqdm(scheduler.timesteps):
  # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
  latent_model_input = torch.cat([latents] * 2)

  latent_model_input = scheduler.scale_model_input(latent_model_input, t)

  # predict the noise residual
  with torch.no_grad():
    noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

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
# pil_images = [Image.fromarray(image) for image in images]

from PIL import Image

pil_images = [Image.fromarray(image) for image in images]
for i, pil_image in enumerate(pil_images):
     pil_image.save(f"boost_{i:03}.png")


exit()





def hook_fn(module, input, output):
    global hooked_output3
    hooked_output3 = output

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

input_data = input_data.to("cuda")


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
