from typing import Union, Dict, List, Optional
import torch
from torch import nn
from diffusers import UNet2DConditionModel, StableDiffusionPipeline
from huggingface_hub import hf_hub_download
from fineface.au_attention import hack_unet_attn_layers, AUAttnProcessor, AUIPAttnProcessor2_0
from fineface.fineface_pipeline import AUEncoder, FineFacePipeline

class MLPProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, num_tokens=4):
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim*2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim*2, cross_attention_dim*num_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        
    def forward(self, id_embeds):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x

class FineFaceIPAdapterPipeline(FineFacePipeline):
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        sd_repo_id = "SG161222/Realistic_Vision_V4.0_noVAE"
        vae_repo_id = "stabilityai/sd-vae-ft-mse"
        fineface_repo_id = "Tvaranka/fineface"
        ip_adapter_repo_id = "h94/IP-Adapter-FaceID"
        AUS = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
        self.AU_DICT = {f"AU{au}": 0 for au in AUS}

        unet = UNet2DConditionModel.from_pretrained(sd_repo_id, subfolder="unet", cache_dir=cache_dir)
        unet.set_default_attn_processor()
        hack_unet_attn_layers(unet, AUIPAttnProcessor2_0)
        unet.load_state_dict(
            torch.load(hf_hub_download(fineface_repo_id, "sd15_attn_processors.ckpt"), map_location=self.device),
            strict=False
        )
        self.pipe = StableDiffusionPipeline.from_pretrained(sd_repo_id, unet=unet, cache_dir=cache_dir).to(self.device)

        self.au_encoder = AUEncoder(hidden_dim=756, clip_dim=768, pad_zeros=False).to(self.device)
        self.au_encoder.load_state_dict(
            torch.load(hf_hub_download(fineface_repo_id, "sd15_au_encoder.ckpt"), map_location=self.device)
        )
        ip_adapter_state_dict = torch.load(
            hf_hub_download(ip_adapter_repo_id, "ip-adapter-faceid-portrait-v11_sd15.bin"), map_location=self.device
        )
        # unet.load_state_dict(
        #     ip_adapter_state_dict["ip_adapter"],
        #     strict=False
        # )
        ip_layers = torch.nn.ModuleList(unet.attn_processors.values())
        ip_layers.load_state_dict(ip_adapter_state_dict["ip_adapter"], strict=False)
        self.ip_proj_model = MLPProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            id_embeddings_dim=512,
            num_tokens=16,
        ).to(self.device)
        self.ip_proj_model.load_state_dict(ip_adapter_state_dict["image_proj"])

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

    def encode_images(self, faceid_embeds):
        multi_face = False
        if faceid_embeds.dim() == 3:
            multi_face = True
            b, n, c = faceid_embeds.shape
            faceid_embeds = faceid_embeds.reshape(b*n, c)
        faceid_embeds = faceid_embeds.to(self.device)
        image_embeds = self.ip_proj_model(faceid_embeds)
        uncond_image_embeds = self.ip_proj_model(torch.zeros_like(faceid_embeds))
        if multi_face:
            c = image_embeds.size(-1)
            image_embeds = image_embeds.reshape(b, -1, c)
            uncond_image_embeds = uncond_image_embeds.reshape(b, -1, c)
        return torch.cat([uncond_image_embeds, image_embeds])

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        aus: Union[Dict, List[Dict]] = None,
        au_scale: float = 1.0,
        faceid_embeds: torch.Tensor = None,
        ip_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ):

        au_embeds = self.encode_aus(aus).to(self.device)
        image_embedding = self.encode_images(faceid_embeds).to(self.device)
        output = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            cross_attention_kwargs={
                "au_embedding": au_embeds,
                "au_scale": au_scale,
                "image_embedding": image_embedding,
                "ip_scale": ip_scale
            }
        )
        return output