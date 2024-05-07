from matplotlib import pyplot as plt
from diffusers import PNDMScheduler, StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
import torch;

def test(prompt):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #加载
    pipe = StableDiffusionPipeline.from_pretrained('models/cheese_chellenge',
                                                   torch_dtype=torch.float32)
    pipe = pipe.to(device)
    
    #运算
    image = pipe(prompt).images[0]
    image.save('test.jpg')

test("<POULIGNY SAINT- PIERRE> :a close up of a piece of cheese on a rock")