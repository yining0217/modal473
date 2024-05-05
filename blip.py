import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

def image_to_text(directory):
    raw_image = Image.open(directory).convert('RGB')
    # unconditional image captioning
    wordlist = directory.split('/')
    text = wordlist[2] + ':'
    print(text)
    inputs = processor(raw_image, text, return_tensors="pt")
    out = model.generate(**inputs)
    result = processor.decode(out[0], skip_special_tokens=True)
    return result