import fine_tune_model
import os
import wandb
import json

import torch
import torchvision
import torchvision.transforms as transforms
import random
from PIL import Image
import numpy as np

from diffusers import PNDMScheduler, StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor

from diffusers import DDPMScheduler

from transformers import CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTokenizer

def jpg_to_tensor(image_path):
    """
    Function to transform a colorful JPG image to a PyTorch tensor.

    Parameters:
    image_path (str): Path to the JPG image file.

    Returns:
    torch.Tensor: Tensor representing the image.
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Resize the image to 512x512
        transforms.ToTensor(),           # Convert image to tensor
    ])
    
    # Load the image
    image = Image.open(image_path)
    
    # Apply transformations
    tensor = transform(image)
    
    return tensor


noise_scheduler = DDPMScheduler(beta_start=0.00085,
                                beta_end=0.012,
                                beta_schedule='scaled_linear',
                                num_train_timesteps=1000)
                                ##tensor_format='pt')


#加载tokenizer
tokenizer = CLIPTokenizer.from_pretrained(
    'models/cheese_chellenge',
    subfolder='tokenizer',
)
#cheese_name导入
cheese_list_directory = './list_of_cheese.txt'
f = open(cheese_list_directory)
cheese_list = []
for name in f:
    cheese_list.append(name.replace('\n',''))

# files = {''cheese_name':[images_directory]}
cheese_directories = ['./datas/'+name for name in cheese_list]
files = {}
for directory in cheese_directories:
    cheese_images = []
    for dirpath, dirnames, filenames in os.walk(directory): 
        for file in filenames:
            cheese_images.append(directory+'/'+file)
        files[cheese_list[cheese_directories.index(directory)]] = cheese_images

all_images = []
for name in cheese_list:
    for image in files[name]:
        if(image.endswith(('jpg','png','jpeg','bmp'))):
            all_images.append(image)
 
#captionning_validation loading
f2 = open('captionning_validation' + '.json', 'r')
captionning_validation = json.load(f2)
f2.close()


#定义数据集
class Dataset(torch.utils.data.Dataset):

    def __init__(self):
        self.flip_transform = torchvision.transforms.RandomHorizontalFlip(
            p=0.5)

    def __len__(self):
        return len(all_images)

    def __getitem__(self, i):
        data = {}
        text = captionning_validation[all_images[i]]
        #选择选择一段文字,并编码
        data['input_ids'] = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=77,
            return_tensors='pt',
        )['input_ids'][0]

        #加载图片
        data['pixel_values'] = jpg_to_tensor(all_images[i])

        return data



def forward(data):
    device = data['pixel_values'].device

    #使用vae压缩原图像
    #[1, 3, 512, 512] -> [1, 4, 64, 64]
    latents = vae.encode(data['pixel_values']).latent_dist.sample().detach()
    latents = latents * 0.18215

    #随机b张噪声图
    #[1, 4, 64, 64]
    noise = torch.randn(latents.shape).to(device)

    #随机采样0-1000之间的b个数字,为每张图片随机一个步数
    #[1]
    timesteps = torch.randint(0, 1000, (1, ), device=device).long()

    #把噪声添加到压缩图中,维度不变
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    #编码文字
    #[1, 77, 768]
    encoder_hidden_states = text_encoder(data['input_ids'])[0]

    #根据文字信息,从混合图中把噪声图给抽取出来
    #[1, 4, 64, 64]
    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

    #求mse loss即可
    #[1, 4, 64, 64]
    loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction='none')
    #[1, 4, 64, 64] -> [1]
    loss = loss.mean(dim=[1, 2, 3])

    return loss



def save():
    #保存模型
    pipeline = StableDiffusionPipeline(
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        scheduler=PNDMScheduler(beta_start=0.00085,
                                beta_end=0.012,
                                beta_schedule='scaled_linear',
                                skip_prk_steps=True),
        safety_checker=StableDiffusionSafetyChecker.from_pretrained(
            'CompVis/stable-diffusion-safety-checker'),
        feature_extractor=CLIPFeatureExtractor.from_pretrained(
            'openai/clip-vit-base-patch32'),
    )
    pipeline.save_pretrained('models/cheese_chellenge')
    #保存新词的映射
    learned_embeds = {}
    for i in range(49408,49408+37):
        learned_embeds[i] = text_encoder.get_input_embeddings().weight[i].detach(
    ).cpu()
    torch.save(learned_embeds, 'models/cheese_chellenge/learned_embeds.bin')
    

def train(cfg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    global text_encoder
    global vae
    global unet
    
    dataset = Dataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    text_encoder, vae, unet = fine_tune_model.model2('models/cheese_chellenge')

    text_encoder = text_encoder.to(device)
    vae = vae.to(device)
    unet = unet.to(device)
    
    pipeline = StableDiffusionPipeline(
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        scheduler=PNDMScheduler(beta_start=0.00085,
                                beta_end=0.012,
                                beta_schedule='scaled_linear',
                                skip_prk_steps=True),
        safety_checker=StableDiffusionSafetyChecker.from_pretrained(
            'CompVis/stable-diffusion-safety-checker'),
        feature_extractor=CLIPFeatureExtractor.from_pretrained(
            'openai/clip-vit-base-patch32'),
    )
    
    optimizer = torch.optim.AdamW(
        pipeline.parameters(),
        lr=2e-3,
    )
    loss_mean = []
    for epoch in range(cfg['epoch']):
        for i, data in enumerate(loader):
            data['pixel_values'] = data['pixel_values'].to(device)
            data['input_ids'] = data['input_ids'].to(device)
            loss = forward(data)
            loss.backward()
            
            #把除了新词以外,其他词的梯度置为0
            grads = text_encoder.get_input_embeddings().weight.grad
            for j in range(grads.shape[0]):
                if not(j>=49408 and j<49408+37):
                    grads[j, :] = 0
            optimizer.step()
            optimizer.zero_grad()
            
            loss_mean.append(loss.item())
        print(epoch)
        save()
    
    print('save successful')
    
train({'experiment_name':'finetune_epoch_100_allunfreeze','epoch':100})
