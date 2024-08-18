import torch
from fineface import FineFacePipeline

pipe = FineFacePipeline()

torch.manual_seed(2)

prompt = "a closeup of a boy in a park"
# Set AUs
# Possible AUs: [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
aus = {"AU1": 2.5, "AU6": 2.5, "AU12": 5}

image = pipe(prompt, aus).images[0]

image.save(f"results/{prompt} {str(aus)}.jpg")