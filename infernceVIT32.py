import PIL
from transformers import CLIPProcessor, CLIPModel
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, default="openai/clip-vit-base-patch32")
parser.add_argument("--image_path", type=str, required=True)
parser.add_argument("--guess", type=str, required=True, help="guess.txt containing the options from which VIT will choose")

args = parser.parse_args()

model_checkpoint = args.model
model = CLIPModel.from_pretrained(model_checkpoint)
processor = CLIPProcessor.from_pretrained(model_checkpoint)

image = PIL.Image.open(args.image_path)
with open(args.guess, 'r') as f:
    options = f.read().splitlines()

inputs = processor(text=options, images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

dic = dict(zip(options, probs.detach().numpy()[0]))

dic = dict(sorted(dic.items(), key = lambda item: item[1], reverse=True))

print("Rank \tOption\t\tProbability")
for i, (key, val) in enumerate(dic.items()):
    print(i+1," | \t",  key ," | \t ", val)