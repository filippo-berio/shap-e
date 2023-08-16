from urllib.request import urlopen
from gradio_client import Client

import torch
import urllib.request
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.image_util import load_image
from shap_e.util.notebooks import decode_latent_mesh


def make_model(images):
    client = Client('https://hysts-shap-e.hf.space/')
    result = client.predict(
        # images[0],
        'https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png',
        # str (filepath or URL to image) in 'Input image' Image component
        0,  # int | float (numeric value between 0 and 2147483647) in 'Seed' Slider component
        3,  # int | float (numeric value between 1 and 20) in 'Guidance scale' Slider component
        64,  # int | float (numeric value between 1 and 100) in 'Number of inference steps' Slider component
        api_name="/image-to-3d"
    )
    print(result)
    # result = '/tmp/gradio/595a9135732bc9c1a348306470042afc4f835a2b/tmplp8qv9n7.glb'
    file = urlopen(f'https://hysts-shap-e.hf.space/file={result}')
    return file

def make_model_the_right_way(images):
    # %%
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # %%
    xm = load_model('transmitter', device=device)
    model = load_model('image300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))

    # %%
    batch_size = 4
    guidance_scale = 3.0

    # To get the best result, you should remove the background and show only the object of interest to the model.
    urllib.request.urlretrieve(
        images[0],
        "tmp.png"
    )
    image = load_image("tmp.png")

    print(image)

    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(images=[image] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )
    # %%
    render_mode = 'nerf'  # you can change this to 'stf' for mesh rendering
    size = 64  # this is the size of the renders; higher values take longer to render.

    # Example of saving the latents as meshes.

    for i, latent in enumerate(latents):
        t = decode_latent_mesh(xm, latent).tri_mesh()
        with open(f'example_mesh_{i}.ply', 'wb') as f:
            t.write_ply(f)
        with open(f'example_mesh_{i}.obj', 'w') as f:
            t.write_obj(f)


if __name__ == '__main__':
    make_model_the_right_way(['https://upload.wikimedia.org/wikipedia/commons/1/14/Vans_womans_shoe.JPG'])