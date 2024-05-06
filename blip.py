import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
import json

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

cheese_list_directory = './list_of_cheese.txt'
f = open(cheese_list_directory)
cheese_list = []
for name in f:
    cheese_list.append(name.replace('\n',''))

# files = {'cheese_name':[images_directory]}
cheese_directories = ['./datas/'+name for name in cheese_list]
files = {}
for directory in cheese_directories:
    cheese_images = []
    for dirpath, dirnames, filenames in os.walk(directory): 
        for file in filenames:
            cheese_images.append(directory+'/'+file)
        files[cheese_list[cheese_directories.index(directory)]] = cheese_images

text = {}

def image_to_text(directory):
    raw_image = Image.open(directory).convert('RGB')
    # unconditional image captioning
    wordlist = directory.split('/')
    #text = wordlist[2]+ ':'
    #inputs = processor(raw_image, text, return_tensors="pt")
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)   
    result = processor.decode(out[0], skip_special_tokens= True, max_length = 100)
    return result

def write_json(dictionnaire,json_name):
     dictionnaire_json= json.dumps(dictionnaire,sort_keys=False, indent=1, separators=(',', ': '))
     file_name = json_name+'.json'
     f = open(file_name, 'w')
     f.write(dictionnaire_json)
     f.close()

for name in cheese_list:
    for image in files[name]:
        if(image.endswith(('jpg','png','jpeg','bmp'))):
            wordlist = image.split('/')
            text[image] = '<'+wordlist[2]+'>' + ' '+':'+image_to_text(image)
            print(text[image])

json_name = 'captionning_validation'
write_json(text,json_name)