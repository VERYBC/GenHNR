import torch
import timm
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

args_do_not_overide = ['verbose', 'resume_from']
os.environ["TORCH_CUDA_DSA_DISABLE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class TimmModel_mix(nn.Module):

    def __init__(self,
                 model_name,
                 pretrained=True,
                 sz0=1,
                 sz1=1,
                 ):  

        super(TimmModel_mix, self).__init__()

        self.conv_pool  = torch.nn.AdaptiveAvgPool2d(1)
        self.dino_pool  = torch.nn.AdaptiveAvgPool1d(1)
        self.model_name = model_name
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        if "Tiny" in model_name:
            self.model1 = timm.create_model('convnextv2_nano.fcmae_ft_in22k_in1k_384', pretrained=pretrained, num_classes=0)
            self.model2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
            # local loading
            # self.model2 = torch.hub.load(current_dir + '/model_lib/dinov2/facebookresearch_dinov2_main/','dinov2_vits14_reg', trust_repo=True, source='local')
            # self.model2.load_state_dict(torch.load(current_dir + '/model_lib/dinov2/dinov2_vits14_reg4_pretrain.pth'))
        elif "Base" in model_name:
            self.model1 = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k_384', pretrained=pretrained, num_classes=0)
            self.model2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
            # local loading
            # self.model2 = torch.hub.load(current_dir + '/model_lib/dinov2/facebookresearch_dinov2_main/','dinov2_vitb14_reg', trust_repo=True, source='local')
            # self.model2.load_state_dict(torch.load(current_dir + '/model_lib/dinov2/dinov2_vitb14_reg4_pretrain.pth'))

        self.region_size = 10
        
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.sz0 = sz0
        self.sz1 = sz1
  
    def get_config(self, ):
        data_config1 = timm.data.resolve_model_data_config(self.model1)
        return data_config1

    def set_grad_checkpointing(self, enable=True):

        self.model1.set_grad_checkpointing(enable)

    def forward(self, img1, img2=None, center_feature=False, fmap=False):
        
        if img2 is not None:

            #------------------------------------------------------------img1
            # convnextv2
            x = self.model1.stem(img1)
            feat1_1 = self.model1.stages(x)
            local_feat1 = self.conv_pool(feat1_1).squeeze(-2, -1)

            # dino
            x = self.model2.prepare_tokens_with_masks(img1)
            for blk in self.model2.blocks:
                x = blk(x)
            x_norm = self.model2.norm(x)
            ret = {
                "x_norm_clstoken": x_norm[:, 0],
                "x_norm_regtokens": x_norm[:, 1 : self.model2.num_register_tokens + 1],
                "x_norm_patchtokens": x_norm[:, self.model2.num_register_tokens + 1 :],
                "x_prenorm": x,
            }
            feat1_2 = self.model2.head(ret["x_norm_patchtokens"])

            global_feat1 = self.dino_pool(feat1_2.permute(0, 2, 1)).squeeze(-1)

            # Conv
            c1 = local_feat1

            # Dino
            d1 = global_feat1

            # ------------------------------------------------------------img2
            # convnext
            x = self.model1.stem(img2)
            feat2_1 = self.model1.stages(x)
            local_feat2 = self.conv_pool(feat2_1).squeeze(-2, -1)

            # dino
            x = self.model2.prepare_tokens_with_masks(img2)
            for blk in self.model2.blocks:
                x = blk(x)
            x_norm = self.model2.norm(x)
            ret = {
                "x_norm_clstoken": x_norm[:, 0],
                "x_norm_regtokens": x_norm[:, 1 : self.model2.num_register_tokens + 1],
                "x_norm_patchtokens": x_norm[:, self.model2.num_register_tokens + 1 :],
                "x_prenorm": x,
            }
            feat2_2 = self.model2.head(ret["x_norm_patchtokens"])

            global_feat2 = self.dino_pool(feat2_2.permute(0, 2, 1)).squeeze(-1)

            # Conv
            c2 = local_feat2

            # Dino
            d2 = global_feat2

            return  c1, c2, d1, d2

        else:
            # convnext
            x = self.model1.stem(img1)
            feat1 = self.model1.stages(x)
            local_feat = self.conv_pool(feat1).squeeze(-2, -1)

            # dino
            x = self.model2.prepare_tokens_with_masks(img1)
            for blk in self.model2.blocks:
                x = blk(x)
            x_norm = self.model2.norm(x)
            ret = {
                "x_norm_clstoken": x_norm[:, 0],
                "x_norm_regtokens": x_norm[:, 1 : self.model2.num_register_tokens + 1],
                "x_norm_patchtokens": x_norm[:, self.model2.num_register_tokens + 1 :],
                "x_prenorm": x,
            }
            feat2 = self.model2.head(ret["x_norm_patchtokens"])
            global_feat = self.dino_pool(feat2.permute(0, 2, 1)).squeeze(-1)

            # fusion
            local_feat = F.normalize(local_feat, p=2, dim=-1)
            global_feat = F.normalize(global_feat, p=2, dim=-1)

            feat = torch.cat((local_feat, global_feat), dim=-1)

            if center_feature:

                # center feature
                width = feat2.shape[1] // 784

                feat2_ = feat2.permute(0, 2, 1)
                feat2_ = feat2_.view(feat2_.shape[0], feat2_.shape[1], 28, 28 * width)

                feat1_ = F.interpolate(feat1, size=(28, 28 * width), mode='bilinear', align_corners=False) 

                feat1_ = F.normalize(feat1_, p=2, dim=1)
                feat2_ = F.normalize(feat2_, p=2, dim=1)

                center_feat =  torch.cat([feat1_, feat2_], dim=1)

                return feat, center_feat

            elif fmap:
                feat1 = F.interpolate(feat1, size=(self.sz1, self.sz1), mode='bilinear', align_corners=False)

                B2, N2, C2 = feat2.shape
                feat2 = feat2.transpose(1, 2)
                feat2 = feat2.reshape(B2, C2, 28, 28)
                feat2 = F.interpolate(feat2, size=(self.sz1, self.sz1), mode='bilinear', align_corners=False)

                return torch.cat((feat1, feat2), dim=1)
            else:
                return feat


class Encoder(nn.Module):
    def __init__(self, input_dim=1024, bottleneck_dim=64):
        super().__init__()

        self.encoder_base = nn.Sequential(
            nn.Conv2d(input_dim, bottleneck_dim, kernel_size=1, bias=False),
            nn.GroupNorm(bottleneck_dim // 8, bottleneck_dim)
        )

        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, input_dim // 4, kernel_size=1, bias=False),
            nn.GroupNorm(input_dim // 4 // 8, input_dim // 4),
            nn.ReLU(),

            nn.Conv2d(input_dim // 4, bottleneck_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(bottleneck_dim // 8, bottleneck_dim),
            nn.ReLU()
        )

        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):

        base = self.encoder_base(x)
        delta = self.encoder(x)

        x = base + self.beta * delta

        return x

class Adapter(nn.Module):
    def __init__(self, input_dim=128, output_dim=512):
        super().__init__()

        hidden_dim = input_dim * 2 # 2
        low_dim    = input_dim * 4 # 4

        self.decoder_base = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False),
            nn.GroupNorm(output_dim // 8, output_dim)
        )

        self.decoder_high = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(hidden_dim // 8, hidden_dim),
            nn.ReLU(),
        )

        self.decoder_low = nn.Sequential(
            nn.Conv2d(hidden_dim, low_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(low_dim // 8, low_dim),
            nn.ReLU(),
        )

        self.lateral = nn.Sequential(
            nn.Conv2d(low_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(hidden_dim // 8, hidden_dim)
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, output_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(output_dim // 8, output_dim),
        )

        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):

        base = self.decoder_base(x)

        x_high = self.decoder_high(x)      # B × 128 × 12 × 12

        x_low = self.decoder_low(x_high)   # B × 256 × 6 × 6
        x_low = self.lateral(x_low)        # B × 128 × 6 × 6

        x_low = F.interpolate(x_low, size=x_high.shape[-2:], mode="bilinear", align_corners=False)

        delta = x_high + x_low
        delta = self.out_proj(delta)    

        x = base + self.alpha * delta

        return x
