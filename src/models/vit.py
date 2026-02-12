"""ViT From scratch"""
import torch
from torch import nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


device = "cuda" if torch.cuda.is_available() else "cpu"


### ViT patch embedding layer
class PatchEmbedding(nn.Module):
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dim=768,
                 ):
        
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # Conv2d layer

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x): # (B, C, H, W)
        x = self.proj(x) # (B, D, H/P, W/P)
        x = x.flatten(2)
        x = x.transpose(1, 2) # Swap dim 1 & 2
        return x


### Creating class token embedding and postion embedding
class ViTEmbedding(nn.Module):
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dim=768,
                 embed_dropout=0.1):
        
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            image_size, patch_size, in_channels, embed_dim
        )

        num_patches = self.patch_embed.num_patches

        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Postitional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        self.dropout = nn.Dropout(p=embed_dropout)  

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # Adding the class token
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Adding the pos embedding
        x = x + self.pos_embed

        x = self.dropout(x)
        
        return x
    

class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self,
                 embed_dim=768,
                 num_heads=12):
        
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=embed_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, 
                                                    num_heads,
                                                    batch_first=True)
       

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(x, x, x) # (B, N, D)
        x = attn_output + residual
        return x
    


class MLPBlock(nn.Module):
    def __init__(self,
                 embed_dim= 768,
                 MLP_size=3072,
                 dropout=0.1):
        super().__init__()

        self.layer_norm = nn.LayerNorm(embed_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=MLP_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=MLP_size, out_features=embed_dim),
            nn.Dropout(p=dropout))
        
    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x + residual
    


## We can also use a pytorch layer to replace this...
class TransformerEncoderBlock(nn.Module):
    def __init__(self,
               embed_dim=768,
               nums_head=12,
               MLP_size=3072,
               dropout=0.1):
        super().__init__()

        self.msa = MultiHeadSelfAttentionBlock(embed_dim,nums_head)

        self.mlp = MLPBlock(embed_dim,MLP_size,dropout)


    def forward(self, x):
        x = self.msa(x)
        x = self.mlp(x)
        
        return x



class ClassifierHead(nn.Module):
    def __init__(self, 
                 embed_dim=768,
                 num_classes=10):
        super().__init__()
        
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        return self.classifier(x)
    


class TransformerEncoder(nn.Module):
    def __init__(self,
                 num_layers=12,
                 embed_dim=768,
                 nums_head=12,
                 MLP_size=3072,
                 dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList([TransformerEncoderBlock(embed_dim, nums_head,MLP_size,dropout) 
                                     for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



class ViT(nn.Module):
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dim=768,
                 num_layers=12,
                 nums_head=12,
                 MLP_size=3072,
                 dropout=0.1,
                 embed_dropout=0.1,
                 num_classes=10
                 ):
        super().__init__()

        self.embedding_layer = ViTEmbedding(image_size, patch_size, in_channels, embed_dim,embed_dropout)

        self.transformer_encoder = TransformerEncoder(num_layers, embed_dim, nums_head, MLP_size, dropout)
        
        self.classifier = ClassifierHead(embed_dim, num_classes)

        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x): # (B, C, H, W)
        x = self.embedding_layer(x) # (B, N, D)
        x = self.transformer_encoder(x) # (B, N, D)
        x = self.ln(x) # (B, N, D)
        x = x[:, 0] # (B, D)
        x = self.classifier(x) # (B, num_classes)
        return x
    


def create_ViT_model():
    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)

    for params in model.parameters():
        params.requires_grad = False
    
    model.heads.head = nn.Linear(in_features=768, out_features=10)
    model.to(device)

    print(f"ViT transfer learning model initialised")
    return model
