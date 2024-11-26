from typing import Union, Dict, List, Optional
import torch
from torch import nn
from diffusers import UNet2DConditionModel, StableDiffusionPipeline
from huggingface_hub import hf_hub_download
from fineface.au_attention import hack_unet_attn_layers, AUAttnProcessor


class AUEncoder(torch.nn.Module):
    def __init__(self, number_of_aus: int = 12, hidden_dim: int = 64, clip_dim: int = 1024, pad_zeros: bool = True):
        super().__init__()
        self.au_encoder = nn.Linear(number_of_aus, hidden_dim)
        self.n_aus = number_of_aus
        self.hidden_dim = hidden_dim
        self.clip_dim = clip_dim
        self.pad_zeros = pad_zeros
        
    def forward(self, x):
        x = x.clone()
        x = torch.cat([x, self.au_encoder(x)], dim=1)
        if self.pad_zeros:
            x = torch.cat([
                x,
                torch.zeros(x.shape[0], self.clip_dim - self.n_aus - self.hidden_dim).float().to(x.device)
            ], dim=1)
        x = x.reshape(-1, 1, self.clip_dim)
        return x


class FineFacePipeline:
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        sd_repo_id = "stabilityai/stable-diffusion-2-1-base"
        fineface_repo_id = "Tvaranka/fineface"
        AUS = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
        self.AU_DICT = {f"AU{au}": 0 for au in AUS}

        unet = UNet2DConditionModel.from_pretrained(sd_repo_id, subfolder="unet", cache_dir=cache_dir)
        unet.set_default_attn_processor()
        hack_unet_attn_layers(unet, AUAttnProcessor)
        unet.load_state_dict(
            torch.load(hf_hub_download(fineface_repo_id, "attn_processors.ckpt"), map_location=self.device),
            strict=False
        )
        unet.load_attn_procs(
            fineface_repo_id, weight_name="pytorch_lora_weights.safetensors", adapter_name="unet"
        )
        self.pipe = StableDiffusionPipeline.from_pretrained(sd_repo_id, unet=unet, cache_dir=cache_dir).to(self.device)

        self.au_encoder = AUEncoder().to(self.device)
        self.au_encoder.load_state_dict(
            torch.load(hf_hub_download(fineface_repo_id, "au_encoder.ckpt"), map_location=self.device)
        )

    def encode_aus(self, aus: Union[Dict, List[Dict]] = None):
        if isinstance(aus, Dict):
            aus = [aus]
        # Create empty dict with all AUs
        new_aus_dict = [self.AU_DICT.copy() for _ in aus]
        # Update created dicts with AU prompts
        [new_aus_dict[i].update(aus[i]) for i in range(len(aus))]
        new_aus = [list(au_dict.values()) for au_dict in new_aus_dict]
        new_aus = torch.tensor(new_aus).float().to(self.device)
        # Encode AUs
        au_embeds = self.au_encoder(new_aus)
        uncond_aus = torch.zeros_like(new_aus)
        negative_au_embeds = self.au_encoder(uncond_aus)
        return torch.cat([negative_au_embeds, au_embeds], dim=0)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        aus: Union[Dict, List[Dict]] = None,
        au_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ):

        au_embeds = self.encode_aus(aus).to(self.device)
        output = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            cross_attention_kwargs={"au_embedding": au_embeds, "au_scale": au_scale},
        )
        return output
        

        