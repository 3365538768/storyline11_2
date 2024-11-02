import random
import numpy as np
MAX_SEED = np.iinfo(np.int32).max
import soundfile as sf
import torch
from pathlib import Path
import sys
import urllib.request
import yaml
import os
from transformers import T5Tokenizer, T5EncoderModel
from src.modules.dac import DAC
from src.modules.stable_vae import load_vae
import torch.nn as nn
from src.models.conditioners import MaskDiT
from accelerate import Accelerator
from diffusers import DDIMScheduler

configs = {'s3_xl': {'path': 'ckpts/s3/ezaudio_s3_xl.pt',
                     'url': 'https://huggingface.co/OpenSound/EzAudio/resolve/main/ckpts/s3/ezaudio_s3_xl.pt',
                     'config': 'ckpts/ezaudio-xl.yml'},
          'vae': {'path': 'ckpts/vae/1m.pt',
                  'url': 'https://huggingface.co/OpenSound/EzAudio/resolve/main/ckpts/vae/1m.pt'}
          }

class EzAudio:
    def __init__(self, model_name, ckpt_path=None, vae_path=None, device='cuda'):
        self.device = device
        config_name = configs[model_name]['config']
        if ckpt_path is None:
            ckpt_path = self.download_ckpt(configs[model_name])

        if vae_path is None:
            vae_path = self.download_ckpt(configs['vae'])

        (self.autoencoder, self.unet, self.tokenizer,
         self.text_encoder, self.noise_scheduler, self.params) = self.load_models(config_name, ckpt_path, vae_path, device)

    def download_ckpt(self, model_dict):
        local_path = Path(model_dict['path'])
        url = model_dict['url']
        # Create directories if they don't exist
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if not local_path.exists() and url:
            print(f"Downloading from {url} to {local_path}...")

            def progress_bar(block_num, block_size, total_size):
                downloaded = block_num * block_size
                progress = downloaded / total_size * 100
                sys.stdout.write(f"\rProgress: {progress:.2f}%")
                sys.stdout.flush()

            try:
                urllib.request.urlretrieve(url, local_path, reporthook=progress_bar)
                print(f"Downloaded checkpoint to {local_path}")
            except Exception as e:
                print(f"Error downloading checkpoint: {e}")
        else:
            print(f"Checkpoint already exists at {local_path}")
        return local_path
    def load_models(self, config_name, ckpt_path, vae_path, device):
        params = load_yaml_with_includes(config_name)

        # Load codec model
        autoencoder = Autoencoder(ckpt_path=vae_path,
                                  model_type=params['autoencoder']['name'],
                                  quantization_first=params['autoencoder']['q_first'],
                                  ).to(device)
        autoencoder.eval()

        # Load text encoder
        tokenizer = T5Tokenizer.from_pretrained(params['text_encoder']['model'], cache_dir="models")
        text_encoder = T5EncoderModel.from_pretrained(params['text_encoder']['model'], cache_dir="models").to(device)
        text_encoder.eval()

        # Load main U-Net model
        unet = MaskDiT(**params['model']).to(device)
        unet.load_state_dict(torch.load(ckpt_path, map_location='cpu')['model'])
        unet.eval()

        if device == 'cuda':
            accelerator = Accelerator(mixed_precision="fp16")
            unet = accelerator.prepare(unet)

        # Load noise scheduler
        noise_scheduler = DDIMScheduler(**params['diff'])

        latents = torch.randn((1, 128, 128), device=device)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=device)
        _ = noise_scheduler.add_noise(latents, noise, timesteps)

        return autoencoder, unet, tokenizer, text_encoder, noise_scheduler, params

    def generate_audio(self, text, length=3,
                       guidance_scale=5, guidance_rescale=0.75, ddim_steps=100, eta=1,
                       random_seed=None, randomize_seed=False):
        print(f"ddim_steps={ddim_steps}")
        neg_text = None
        length = length * self.params['autoencoder']['latent_sr']

        gt, gt_mask = None, None

        if text == '':
            guidance_scale = None
            print('empyt input')

        if randomize_seed:
            random_seed = random.randint(0, MAX_SEED)

        pred = inference(self.autoencoder, self.unet,
                         gt, gt_mask,
                         self.tokenizer, self.text_encoder,
                         self.params, self.noise_scheduler,
                         text, neg_text,
                         length,
                         guidance_scale, guidance_rescale,
                         ddim_steps, eta, random_seed,
                         self.device)

        pred = pred.cpu().numpy().squeeze(0).squeeze(0)
        # output_file = f"{save_path}/{text}.wav"
        # sf.write(output_file, pred, samplerate=params['autoencoder']['sr'])

        return self.params['autoencoder']['sr'], pred


def inference(autoencoder, unet, gt, gt_mask,
              tokenizer, text_encoder,
              params, noise_scheduler,
              text_raw, neg_text=None,
              audio_frames=500,
              guidance_scale=3, guidance_rescale=0.0,
              ddim_steps=50, eta=1, random_seed=2024,
              device='cuda',
              ):
    print(f"ddim_steps={ddim_steps}")
    if neg_text is None:
        neg_text = [""]
    if tokenizer is not None:
        text_batch = tokenizer(text_raw,
                               max_length=params['text_encoder']['max_length'],
                               padding="max_length", truncation=True, return_tensors="pt")
        text, text_mask = text_batch.input_ids.to(device), text_batch.attention_mask.to(device).bool()
        text = text_encoder(input_ids=text, attention_mask=text_mask).last_hidden_state

        uncond_text_batch = tokenizer(neg_text,
                                      max_length=params['text_encoder']['max_length'],
                                      padding="max_length", truncation=True, return_tensors="pt")
        uncond_text, uncond_text_mask = uncond_text_batch.input_ids.to(device), uncond_text_batch.attention_mask.to(
            device).bool()
        uncond_text = text_encoder(input_ids=uncond_text,
                                   attention_mask=uncond_text_mask).last_hidden_state
    else:
        text, text_mask = None, None
        guidance_scale = None

    codec_dim = params['model']['out_chans']
    unet.eval()

    if random_seed is not None:
        generator = torch.Generator(device=device).manual_seed(random_seed)
    else:
        generator = torch.Generator(device=device)
        generator.seed()
    print(f"ddim_steps={ddim_steps}")
    noise_scheduler.set_timesteps(ddim_steps)

    # init noise
    noise = torch.randn((1, codec_dim, audio_frames), generator=generator, device=device)
    latents = noise

    for t in noise_scheduler.timesteps:
        latents = noise_scheduler.scale_model_input(latents, t)

        if guidance_scale:

            latents_combined = torch.cat([latents, latents], dim=0)
            text_combined = torch.cat([text, uncond_text], dim=0)
            text_mask_combined = torch.cat([text_mask, uncond_text_mask], dim=0)

            if gt is not None:
                gt_combined = torch.cat([gt, gt], dim=0)
                gt_mask_combined = torch.cat([gt_mask, gt_mask], dim=0)
            else:
                gt_combined = None
                gt_mask_combined = None

            output_combined, _ = unet(latents_combined, t, text_combined, context_mask=text_mask_combined,
                                      cls_token=None, gt=gt_combined, mae_mask_infer=gt_mask_combined)
            output_text, output_uncond = torch.chunk(output_combined, 2, dim=0)

            output_pred = output_uncond + guidance_scale * (output_text - output_uncond)
            if guidance_rescale > 0.0:
                output_pred = rescale_noise_cfg(output_pred, output_text,
                                                guidance_rescale=guidance_rescale)
        else:
            output_pred, mae_mask = unet(latents, t, text, context_mask=text_mask,
                                         cls_token=None, gt=gt, mae_mask_infer=gt_mask)

        latents = noise_scheduler.step(model_output=output_pred, timestep=t,
                                       sample=latents,
                                       eta=eta, generator=generator).prev_sample
        print(f"进行到第{t}次noise_scheduler")
        print(f"latents:{latents},{latents.shape}")
    pred = scale_shift_re(latents, params['autoencoder']['scale'],
                          params['autoencoder']['shift'])
    print(f"pred={pred}")
    if gt is not None:
        pred[~gt_mask] = gt[~gt_mask]
    pred_wav = autoencoder(embedding=pred)
    print(f"pred_wav:{pred_wav}")
    return pred_wav

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

def scale_shift_re(x, scale, shift):
    return (x/scale) - shift

def load_yaml_with_includes(yaml_file):
    def loader_with_include(loader, node):
        # Load the included file
        include_path = os.path.join(os.path.dirname(yaml_file), loader.construct_scalar(node))
        with open(include_path, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    yaml.add_constructor('!include', loader_with_include, Loader=yaml.FullLoader)

    with open(yaml_file, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)
class Autoencoder(nn.Module):
    def __init__(self, ckpt_path, model_type='dac', quantization_first=False):
        super(Autoencoder, self).__init__()
        self.model_type = model_type
        if self.model_type == 'dac':
            model = DAC.load(ckpt_path)
        elif self.model_type == 'stable_vae':
            model = load_vae(ckpt_path)
        else:
            raise NotImplementedError(f"Model type not implemented: {self.model_type}")
        self.ae = model.eval()
        self.quantization_first = quantization_first
        print(f'Autoencoder quantization first mode: {quantization_first}')

    @torch.no_grad()
    def forward(self, audio=None, embedding=None):
        if self.model_type == 'dac':
            return self.process_dac(audio, embedding)
        elif self.model_type == 'encodec':
            return self.process_encodec(audio, embedding)
        elif self.model_type == 'stable_vae':
            return self.process_stable_vae(audio, embedding)
        else:
            raise NotImplementedError(f"Model type not implemented: {self.model_type}")

    def process_dac(self, audio=None, embedding=None):
        if audio is not None:
            z = self.ae.encoder(audio)
            if self.quantization_first:
                z, *_ = self.ae.quantizer(z, None)
            return z
        elif embedding is not None:
            z = embedding
            if self.quantization_first:
                audio = self.ae.decoder(z)
            else:
                z, *_ = self.ae.quantizer(z, None)
                audio = self.ae.decoder(z)
            return audio
        else:
            raise ValueError("Either audio or embedding must be provided.")

    def process_encodec(self, audio=None, embedding=None):
        if audio is not None:
            z = self.ae.encoder(audio)
            if self.quantization_first:
                code = self.ae.quantizer.encode(z)
                z = self.ae.quantizer.decode(code)
            return z
        elif embedding is not None:
            z = embedding
            if self.quantization_first:
                audio = self.ae.decoder(z)
            else:
                code = self.ae.quantizer.encode(z)
                z = self.ae.quantizer.decode(code)
                audio = self.ae.decoder(z)
            return audio
        else:
            raise ValueError("Either audio or embedding must be provided.")

    def process_stable_vae(self, audio=None, embedding=None):
        if audio is not None:
            z = self.ae.encoder(audio)
            if self.quantization_first:
                z = self.ae.bottleneck.encode(z)
            return z
        if embedding is not None:
            z = embedding
            if self.quantization_first:
                audio = self.ae.decoder(z)
            else:
                z = self.ae.bottleneck.encode(z)
                audio = self.ae.decoder(z)
            return audio
        else:
            raise ValueError("Either audio or embedding must be provided.")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ezaudio = EzAudio(model_name='s3_xl', device='cuda')
    prompt = "a dog barking"
    sr, audio = ezaudio.generate_audio(prompt, ddim_steps=100)
    sf.write(f'{prompt}.wav', audio, sr)