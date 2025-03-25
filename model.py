from utils.uniformer import uniformer_base
from huggingface_hub import hf_hub_download
import torch

def get_model():
    model = uniformer_base()
    # load state uniformer_base_k600_32x4.pth  uniformer_small_k600_16x8.pth
    model_path = hf_hub_download(repo_id="Sense-X/uniformer_video", filename="uniformer_base_k600_32x4.pth")
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)

    model.reset_classifier(num_classes=1)

    for param in model.patch_embed1.parameters():
        param.requires_grad = False
    for param in model.patch_embed2.parameters():
        param.requires_grad = False
    for param in model.patch_embed3.parameters():
        param.requires_grad = False
    for param in model.patch_embed4.parameters():
        param.requires_grad = False

    for blk in model.blocks1:
        for param in blk.parameters():
            param.requires_grad = False
    for blk in model.blocks2:
        for param in blk.parameters():
            param.requires_grad = False
    for blk in model.blocks3:
        for param in blk.parameters():
            param.requires_grad = False
    i = 0
    n = len(model.blocks4) // 2
    for blk in model.blocks4:
        if i > n:
            break
        i += 1
        for param in blk.parameters():
            param.requires_grad = False
    # trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
    # print("Các tham số sẽ được train:", trainable_params)
    return model

def get_model():
    model = uniformer_base()
    # load state uniformer_base_k600_32x4.pth  uniformer_small_k600_16x8.pth
    model_path = hf_hub_download(repo_id="Sense-X/uniformer_video", filename="uniformer_base_k600_32x4.pth")
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)

    model.reset_classifier(num_classes=1)

    for param in model.patch_embed1.parameters():
        param.requires_grad = False
    for param in model.patch_embed2.parameters():
        param.requires_grad = False
    for param in model.patch_embed3.parameters():
        param.requires_grad = False
    for param in model.patch_embed4.parameters():
        param.requires_grad = False

    for blk in model.blocks1:
        for param in blk.parameters():
            param.requires_grad = False
    for blk in model.blocks2:
        for param in blk.parameters():
            param.requires_grad = False
    for blk in model.blocks3:
        for param in blk.parameters():
            param.requires_grad = False

    # trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
    # print("Các tham số sẽ được train:", trainable_params)
    return model