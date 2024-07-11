import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, 
                 patch_size=16, 
                 stride=16, 
                 in_chans=3, 
                 embed_dim=768, 
                 norm_layer=None, 
                 flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
    

if __name__ == "__main__":    
    # Create a random input tensor with the shape (batch_size, channels, height, width)
    batch_size = 1
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    
    # Instantiate the PatchEmbed class
    patch_embed = PatchEmbed(patch_size=16, stride=16)
    
    # Forward pass through the PatchEmbed module
    output = patch_embed(input_tensor)
    
    # Print the shape of the output
    print("Output shape:", output.shape)