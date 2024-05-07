from matplotlib import pyplot as plt
from diffusers import PNDMScheduler, StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
import torch;

def test(prompt):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #节省显存
    device = 'cpu'

    #加载
    pipe = StableDiffusionPipeline.from_pretrained('models/cheese_chellenge',
                                                   torch_dtype=torch.float32)
    pipe = pipe.to(device)
    
    #运算
    images = pipe([prompt] * 4, num_inference_steps=50,
                  guidance_scale=7.5).images

    #画图
    def show(image, idx):
        plt.subplot(1, 4, idx)
        plt.imshow(image)
        plt.axis('off')

    plt.figure(figsize=[8, 3])
    for i in range(len(images)):
        show(images[i], i + 1)
    plt.show()

test('<Emmental> : a cheese')