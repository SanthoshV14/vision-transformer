import cv2
import torch
from vision_transformer import VisionTransformer

if __name__ == '__main__':
    image_size = 224
    patch_size = 16
    n_channels = 3
    d_model = 1024
    dropout = 0.1
    n_patches = int(image_size**2/patch_size**2) + 1
    n_dim = (patch_size**2)*n_channels
    n_heads = 64
    n_x = 6
    num_class = 10

    image = cv2.imread('./image.JPEG')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = torch.tensor(image/255., dtype=torch.float)
    input = torch.cat([image.unsqueeze(0), image.unsqueeze(0), image.unsqueeze(0)])
    
    vit = VisionTransformer(patch_size, n_patches, n_dim, num_class, n_heads, d_model, n_x, dropout)
    output = vit(input)
    print(input.shape, output.shape)