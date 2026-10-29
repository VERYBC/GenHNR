import torch
import time 
import torch.nn as nn
import torch.nn.functional as F
from .update import GMA_ori, GMA_mov
from .corr import CorrBlock
from .utils.utils import *
from .utils.torch_geometry import get_perspective_transform
from sample4geo.model import TimmModel_mix, Adapter, Encoder

autocast = torch.amp.autocast

class HCNet(nn.Module):
    def __init__(self, args, model_name, dataset_name):
        super().__init__()
        self.args = args

        self.radius = [3, 3]

        ckpt_pth = None
        if 'BEMN-Base' in model_name:
            sz0 = 16
            sz1 = 12
            deinput_dim = 1536
            input_dim   = deinput_dim // sz0
            output_dim  = input_dim * 4
            if 'Xihua' in dataset_name:
                ckpt_pth   =  "./checkpoints/XIHUA/BEMN/BEMN-B.pth"
        elif 'BEMN-Tiny' in model_name:
            sz0 = 8
            sz1 = 12
            deinput_dim = 1024 # 12-1024
            input_dim   = deinput_dim // sz0
            output_dim  = input_dim * 4
            if 'Xihua' in dataset_name:
                ckpt_pth = "./checkpoints/XIHUA/BEMN/BEMN-T.pth"
           
        self.bemn = TimmModel_mix(model_name, sz0=sz0, sz1=sz1)

        if ckpt_pth is not None:
            model_state_dict = torch.load(ckpt_pth)
            self.bemn.load_state_dict(model_state_dict, strict=False)
            print("Model read weights from:", ckpt_pth)

        self.sat_encoder = Encoder(input_dim=deinput_dim, bottleneck_dim=input_dim)
        self.grd_encoder = Encoder(input_dim=deinput_dim, bottleneck_dim=input_dim) if args.p_siamese else None

        self.sat_ori = Adapter(input_dim=input_dim, output_dim=output_dim)
        self.grd_ori = Adapter(input_dim=input_dim, output_dim=output_dim) if args.p_siamese else None

        self.sat_model = Adapter(input_dim=input_dim, output_dim=output_dim)
        self.grd_model = Adapter(input_dim=input_dim, output_dim=output_dim) #  if args.p_siamese else None

        in_dim = 100  # 2-52 | 3-100 | 4-164 | 5-244 | 3+2-76 4+2-108
        sz = sz1 # sz1

        self.update_block_mov = GMA_mov(self.args, sz, in_dim, self.args.orien)
        self.update_block_ori = GMA_ori(self.args, sz, sz**2) # sz1
    
    def get_flow_now_k(self, four_point, k=4, kf=4):
        N, _, h, w = self.sz
        h, w = h//k, w//k
        four_point = four_point / kf # four_point is at original size， coordinate is at feature map size

        four_point_org = torch.zeros((2, 2, 2)).to(four_point.device)
        four_point_org[:, 0, 0] = torch.Tensor([0, 0])
        four_point_org[:, 0, 1] = torch.Tensor([w-1, 0])
        four_point_org[:, 1, 0] = torch.Tensor([0, h-1])
        four_point_org[:, 1, 1] = torch.Tensor([w -1, h-1])

        four_point_org = four_point_org.unsqueeze(0)
        four_point_org = four_point_org.repeat(N, 1, 1, 1)
        four_point_new = torch.autograd.Variable(four_point_org) + four_point
        four_point_org = four_point_org.flatten(2).permute(0, 2, 1)
        four_point_new = four_point_new.flatten(2).permute(0, 2, 1)
        # H = tgm.get_perspective_transform(four_point_org, four_point_new)
        H = get_perspective_transform(four_point_org, four_point_new)
        gridy, gridx = torch.meshgrid(torch.linspace(0, w-1, steps=w), torch.linspace(0,h-1, steps=h),indexing='ij')
        points = torch.cat((gridx.flatten().unsqueeze(0), gridy.flatten().unsqueeze(0), torch.ones((1, w * h))),
                           dim=0).unsqueeze(0).repeat(N, 1, 1).to(four_point.device)
        points_new = H.bmm(points) # (N,3,3) (N,3,w*h)
        points_new = points_new / points_new[:, 2, :].unsqueeze(1)
        points_new = points_new[:, 0:2, :]
        flow = torch.cat((points_new[:, 0, :].reshape(N, w, h).unsqueeze(1),
                          points_new[:, 1, :].reshape(N, w, h).unsqueeze(1)), dim=1)
        return flow, H

    def initialize_flow_k(self, img, k= 4):
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//k, W//k).to(img.device) # [batch,2, H, W]
        coords1 = coords_grid(N, H//k, W//k).to(img.device)

        return coords0, coords1 # [x,y]

    def forward(self, image1, image2, sat_gps = [], iters_lev0 = 6, test_mode=False, refine=False, refine_mode=False):
        # 0. Normalize input data
        image1  = image1.contiguous()    
        image2  = image2.contiguous()    
        self.sz = image1.shape # [N, 3, h, w]
        
        # 1. Using Backbone to Obtain Feature Maps
        grd_fmap_ = self.bemn(image1, fmap=True)
        sat_fmap_ = self.bemn(image2, fmap=True)

        sat_fmap = self.sat_encoder(sat_fmap_)

        grd_fmap = self.sat_encoder(grd_fmap_) if self.grd_encoder is None else self.grd_encoder(grd_fmap_)

        # Quantification for test
        if refine_mode:
            # int 8
            sat_fmap_, sat_scale = self.quantize_int8_per_channel(sat_fmap)
            sat_fmap = self.dequantize_int8(sat_fmap_, sat_scale)

        fmap1_ori = self.sat_ori(grd_fmap) if self.grd_ori is None else self.grd_ori(grd_fmap)
        fmap2_ori = self.sat_ori(sat_fmap)

        fmap1_ori = fmap1_ori.float()
        fmap2_ori = fmap2_ori.float() 

        fmap1 = self.sat_model(grd_fmap) if self.grd_model is None else self.grd_model(grd_fmap)
        fmap2 = self.sat_model(sat_fmap)

        fmap1 = fmap1.float()
        fmap2 = fmap2.float() 

        sz = fmap1.shape

        # 2. Calculate Correlation Matrix      
        corr_fn_ori = CorrBlock(fmap1_ori, fmap2_ori)
        B, h1, w1, dim, h2, w2 = corr_fn_ori.corr_map.shape
        global_corr = corr_fn_ori.corr_map.contiguous().view(B, h1*w1*dim, h2, w2)
        delta_theta = self.update_block_ori(global_corr)

        if not torch.isfinite(global_corr).all():
            print("[Invalid] global_corr") 
        if not torch.isfinite(delta_theta).all():
            print("[Invalid] delta_theta")

        corr_fn = CorrBlock(fmap1, fmap2, num_levels=2, radius=self.radius)
        coords0, coords1 = self.initialize_flow_k(image1, k=self.sz[-1]//sz[-1]) 
        four_point_disp = torch.zeros((sz[0], 2, 2, 2)).to(fmap1.device) # displacement cube D
        theta = torch.zeros((sz[0], 1)).to(fmap1.device)

        # global rotation initialization
        delta_four_point_rot = self.theta_to_four_point_disp(delta_theta, four_point_disp, k=self.sz[-1]//sz[-1], kf=self.sz[-1]/sz[-1])
        four_point_disp =  four_point_disp + delta_four_point_rot
        coords1, _ = self.get_flow_now_k(four_point_disp, k=self.sz[-1]//sz[-1], kf=self.sz[-1]/sz[-1])
        theta = theta + delta_theta

        # 3. Recurrent Homography Estimation
        flow_predictions = [four_point_disp]  # four_point_disp
        thetas = [theta]

        for itr in range(iters_lev0):
            corr = corr_fn(coords1) # batch,channel,H,W  
            flow = coords1 - coords0 if self.args.flow else torch.cat((coords1, coords0), dim=1) # [batch,2, H, W] mean x, y

            with autocast('cuda', enabled=self.args.mixed_precision): 
                if self.args.orien:

                    delta_four_point, delta_theta = self.update_block_mov(corr, flow) 
                    delta_four_point_rot = self.theta_to_four_point_disp(delta_theta, four_point_disp, k=self.sz[-1]//sz[-1], kf=self.sz[-1]/sz[-1])
                    four_point_disp =  four_point_disp  + delta_four_point + delta_four_point_rot

                    if refine:
                        theta = theta + delta_theta
                        thetas.append(theta)
                else:
                    delta_four_point = self.update_block_mov(corr, flow) # input shape: [b,2+channel,H,W] , output shape [b,2,2,2]
                    four_point_disp =  four_point_disp + delta_four_point

            coords1, H = self.get_flow_now_k(four_point_disp, k=self.sz[-1]//sz[-1], kf=self.sz[-1]/sz[-1]) 
            flow_predictions.append(four_point_disp) # for train 

        # 4. Feature Projection
        flow_predictions_refine = []  
        warp_fmap1 = self.warp_feature(fmap1, H)
        corr_fn_refine = CorrBlock(warp_fmap1, fmap2, num_levels=2, radius=self.radius)
        
        if test_mode:
            return flow_predictions[-1], thetas 
        elif refine_mode:
            return flow_predictions, corr_fn_refine.corr_map, thetas
        else:
            return flow_predictions[1:], corr_fn.corr_map, thetas, flow_predictions_refine
        
    def theta_to_four_point_disp(self, delta_theta, four_point_disp, k=4, kf=4):
        N, _, h, w = self.sz
        h, w = h // k, w // k

        device = four_point_disp.device
        dtype = four_point_disp.dtype
        B = four_point_disp.shape[0]

        delta_theta = delta_theta.view(B)

        four_point_org = torch.zeros((2, 2, 2), device=device, dtype=dtype)

        four_point_org[:, 0, 0] = torch.tensor([0.0    , 0.0], device=device, dtype=dtype)
        four_point_org[:, 0, 1] = torch.tensor([w - 1.0, 0.0], device=device, dtype=dtype)
        four_point_org[:, 1, 0] = torch.tensor([0.0    , h - 1.0], device=device, dtype=dtype)
        four_point_org[:, 1, 1] = torch.tensor([w - 1.0, h - 1.0], device=device, dtype=dtype)

        four_point_org = four_point_org.unsqueeze(0).repeat(B, 1, 1, 1)  # [B,2,2,2]

        four_point_disp_feat = four_point_disp / kf

        current_four_point = four_point_org + four_point_disp_feat  # [B,2,2,2]

        current_corners = current_four_point.flatten(2).permute(0, 2, 1)  # [B,4,2]

        cx = (w - 1.0) / 2.0
        cy = (h - 1.0) / 2.0
        center = torch.tensor([cx, cy], device=device, dtype=dtype).view(1, 1, 2)

        x = current_corners[:, :, 0] - center[:, :, 0]
        y = current_corners[:, :, 1] - center[:, :, 1]

        cos_t = torch.cos(delta_theta).view(B, 1)
        sin_t = torch.sin(delta_theta).view(B, 1)

        x_new = cos_t * x - sin_t * y
        y_new = sin_t * x + cos_t * y

        rotated_corners = torch.stack([
            x_new + center[:, :, 0],
            y_new + center[:, :, 1],
        ], dim=-1)  # [B,4,2]

        delta_rot_feat = rotated_corners - current_corners  # [B,4,2]

        delta_rot_feat = delta_rot_feat.permute(0, 2, 1).view(B, 2, 2, 2)

        delta_rot = delta_rot_feat * kf

        return delta_rot

    def warp_feature(self, fmap, H):
        B, C, h, w = fmap.shape

        yy, xx = torch.meshgrid(
            torch.arange(h, device=fmap.device, dtype=fmap.dtype),
            torch.arange(w, device=fmap.device, dtype=fmap.dtype),
            indexing='ij'
        )

        points = torch.stack([
            xx.reshape(-1),
            yy.reshape(-1),
            torch.ones(h * w, device=fmap.device, dtype=fmap.dtype)
        ], dim=0)

        points = points.unsqueeze(0).repeat(B, 1, 1)

        H_inv = torch.linalg.inv(H)

        points_src = H_inv.bmm(points)
        points_src = points_src / (points_src[:, 2:3, :] + 1e-8)

        x = points_src[:, 0, :]
        y = points_src[:, 1, :]

        x = 2.0 * x / (w - 1) - 1.0
        y = 2.0 * y / (h - 1) - 1.0

        grid = torch.stack([x, y], dim=-1)
        grid = grid.view(B, h, w, 2)

        warped_fmap = F.grid_sample(fmap, grid, mode='bilinear', padding_mode='zeros',align_corners=True)

        return warped_fmap

    def quantize_int8_per_channel(self, x):
        """
        x: [B, C, H, W]
        return:
            x_q: int8
            scale: [B, C, 1, 1]
        """
        scale = x.abs().amax(dim=(2, 3), keepdim=True) / 127.0
        scale = scale.clamp_min(1e-8)

        x_q = torch.round(x / scale)
        x_q = torch.clamp(x_q, -127, 127).to(torch.int8)

        return x_q, scale


    def dequantize_int8(self, x_q, scale):
        return x_q.float() * scale.float()