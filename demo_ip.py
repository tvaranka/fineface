import torch
from fineface import FineFaceIPAdapterPipeline
from huggingface_hub import hf_hub_download

#REMOVE CACHE
pipe = FineFaceIPAdapterPipeline("/scratch/project_462000442/tvaranka/.cache")

torch.manual_seed(2)

# Example case. Use https://huggingface.co/h94/IP-Adapter-FaceID to create your own
# Save with torch.save to shape (1, n_faces, 512)
faceid_embeds = torch.load(
    hf_hub_download("tvaranka/fineface", "elon_id_embeds.pt"), map_location="cpu"
)

prompt = "a man as dark hooded emperor"
# Set AUs
# Possible AUs: [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
aus = {"AU4": 5, "AU6": 3}
torch.manual_seed(3)
image = pipe(
    prompt=prompt,
    aus=aus,
    au_scale=1.0,
    faceid_embeds=faceid_embeds,
    ip_scale=0.9
).images[0]

image.save(f"results/{prompt} {str(aus)}.jpg")