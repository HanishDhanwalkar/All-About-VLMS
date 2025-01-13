import PIL
from transformers import CLIPProcessor, CLIPModel
import argparse
import torch

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, default="openai/clip-vit-base-patch32")
parser.add_argument("--image_path", type=str, required=True)
parser.add_argument("--guess", type=str, default="guess.txt", help="guess.txt containing the options from which VIT will choose")

args = parser.parse_args()

model_checkpoint = args.model
model = CLIPModel.from_pretrained(model_checkpoint)
processor = CLIPProcessor.from_pretrained(model_checkpoint)

image = PIL.Image.open(args.image_path)
with open(args.guess, 'r') as f:
    options = f.read().splitlines()

inputs = processor(text=options, images=image, return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = model(**inputs)

# print(outputs)
