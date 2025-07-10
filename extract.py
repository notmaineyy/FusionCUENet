import clip
import torch
from collections import OrderedDict

def extract_and_save_weights(model_name, filename):
    model, _ = clip.load(model_name, device='cpu')
    new_state_dict = OrderedDict()
    for k, v in model.state_dict().items():
        if 'visual.' in k and k[7:] not in ["proj", "ln_post.weight", "ln_post.bias"]:
            new_state_dict[k[7:]] = v
    torch.save(new_state_dict, filename)

extract_and_save_weights("ViT-B/16", '/vol/bitbucket/sna21/checkpoints/extract/vit_b16.pth')
#extract_and_save_weights("ViT-L/14", '/vol/bitbucket/sna21/checkpoints/extract/vit_l14.pth')
extract_and_save_weights("ViT-L/14@336px", '/vol/bitbucket/sna21/checkpoints/extract/vit_l14_336.pth')
