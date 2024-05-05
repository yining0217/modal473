from transformers import CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel

def model1(checkpoint): ## 除text_encoder外都冻结
    #加载模型
    text_encoder = CLIPTextModel.from_pretrained(checkpoint,
                                                 subfolder='text_encoder')
    vae = AutoencoderKL.from_pretrained(checkpoint, subfolder='vae')
    unet = UNet2DConditionModel.from_pretrained(checkpoint, subfolder='unet')

    text_encoder.train()
    vae.eval()
    unet.eval()

    #添加新词
    text_encoder.resize_token_embeddings(tokenizer.vocab_size + 37)
    
    #初始化新词的参数 toy -> <cat-toy>
    token_embeds = text_encoder.get_input_embeddings().weight.data
    
    for i in range(49408,49408+ 37):
        token_embeds[i] = token_embeds[10738]

    #冻结参数
    for param in vae.parameters():
        param.requires_grad = False

    for param in unet.parameters():
        param.requires_grad = False

    for name, param in text_encoder.named_parameters():
        #除了这一层,其他全部冻结
        if name != 'text_model.embeddings.token_embedding.weight':
            param.requires_grad = False

    return text_encoder, vae, unet


