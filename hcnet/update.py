import torch
import torch.nn as nn
from .utils.utils import *

IN_DIM = 164 # 162 + 2

    
class CNN12(nn.Module):
    def __init__(self, input_dim=256, orien=False):
        super().__init__()

        outputdim = input_dim

        self.layer1 = nn.Sequential(
            nn.Conv2d(IN_DIM, outputdim, 3, padding=1),
            nn.GroupNorm(outputdim // 8, outputdim),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(outputdim, outputdim, 3, padding=1),
            nn.GroupNorm(outputdim // 8, outputdim),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(outputdim, outputdim, 3, padding=1),
            nn.GroupNorm(outputdim // 8, outputdim),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # 10|6 注释
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(outputdim, outputdim, 3, padding=1),
            nn.GroupNorm(outputdim // 8, outputdim),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((2, 2))

        self.layer10 = nn.Sequential(
            nn.Conv2d(outputdim, outputdim, 3, padding=1, stride=1),
            nn.GroupNorm(num_groups=outputdim // 8, num_channels=outputdim),
            nn.ReLU()
        )

        self.layer11 = nn.Sequential(
            nn.Conv2d(outputdim, outputdim, 3, padding=1, stride=1),
            nn.GroupNorm(num_groups=outputdim // 8, num_channels=outputdim),
            nn.ReLU()
        )

        self.theta_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(outputdim * 2 * 2, outputdim),
            nn.ReLU(),
            nn.Linear(outputdim, 1)
        )

        nn.init.zeros_(self.theta_head[-1].weight)
        nn.init.zeros_(self.theta_head[-1].bias)

    def forward(self, x):

        x = self.layer1(x)   # 12→12
        x = self.layer2(x)   # 12→6
        x = self.layer3(x)   # 6→3
        x = self.pool(x)     # 3→2

        x = self.layer10(x)   # 3×3
        x = self.layer11(x)   # 3×3
        x = self.theta_head(x)

        return x


class CNN12m_weight(nn.Module):
    def __init__(self, input_dim=256, orien=False, sz=12):
        super(CNN12m_weight, self).__init__()
        
        self.orien = orien 
        outputdim = input_dim

        self.layer0 = nn.Sequential(nn.Conv2d(IN_DIM, 256, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=256//8, num_channels=256), 
                                    nn.ReLU(), 
                                    nn.Conv2d(256, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=outputdim//8, num_channels=outputdim), 
                                    nn.ReLU())
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(outputdim, outputdim, 3, padding=1, stride=1),
            nn.GroupNorm(num_groups=outputdim // 8, num_channels=outputdim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 12 -> 6
        )
        outputdim_1 = outputdim

        self.layer3 = nn.Sequential(
            nn.Conv2d(outputdim, outputdim, 3, padding=1, stride=1),
            nn.GroupNorm(num_groups=outputdim // 8, num_channels=outputdim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 6 -> 3
        )
        outputdim_3 = outputdim

        self.layer4 = nn.Sequential(
            nn.Conv2d(outputdim, outputdim, 3, padding=1, stride=1),
            nn.GroupNorm(num_groups=outputdim // 8, num_channels=outputdim),
            nn.ReLU()
        )

        self.layer4_global = nn.Sequential(
            nn.Conv2d(input_dim, outputdim - 64, 3, padding=1, stride=1),
            nn.GroupNorm(num_groups=(outputdim - 64) // 8, num_channels=outputdim - 64),
            nn.ReLU()
        )

        outputdim_final = outputdim
        outputdim = outputdim - 64

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        self.layer5 = nn.Sequential(
            nn.Conv2d(outputdim + outputdim_3, outputdim, 3, padding=1, stride=1),
            nn.GroupNorm(num_groups=outputdim // 8, num_channels=outputdim),
            nn.ReLU()
        )

        self.layer8 = nn.Sequential(
            nn.Conv2d(outputdim + outputdim_1, outputdim, 3, padding=1, stride=1),
            nn.GroupNorm(num_groups=outputdim // 8, num_channels=outputdim),
            nn.ReLU()
        )

        self.layer9 = nn.Sequential(
            nn.Conv2d(outputdim, 1, 3, padding=1, stride=1),
            nn.Sigmoid()
        )

        self.pool_to_corners = nn.AdaptiveAvgPool2d((2, 2))

        self.layer10 = nn.Sequential(
            nn.Conv2d(outputdim_final, outputdim_final, 3, padding=1, stride=1),
            nn.GroupNorm(num_groups=outputdim_final // 8, num_channels=outputdim_final),
            nn.ReLU()
        )
        
        self.layer11 = nn.Sequential(
            nn.Conv2d(outputdim_final, outputdim_final, 3, padding=1, stride=1),
            nn.GroupNorm(num_groups=outputdim_final // 8, num_channels=outputdim_final),
            nn.ReLU()
        )

        self.disp_head = nn.Conv2d(outputdim_final, 2, 1)

        if self.orien:
            self.layer10_ = nn.Sequential(
                nn.Conv2d(outputdim_final, outputdim_final, 3, padding=1, stride=1),
                nn.GroupNorm(num_groups=outputdim_final // 8, num_channels=outputdim_final),
                nn.ReLU()
            )

            self.layer11_ = nn.Sequential(
                nn.Conv2d(outputdim_final, outputdim_final, 3, padding=1, stride=1),
                nn.GroupNorm(num_groups=outputdim_final // 8, num_channels=outputdim_final),
                nn.ReLU()
            )

            self.theta_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(outputdim_final * 2 * 2, outputdim_final), 
                nn.ReLU(),
                nn.Linear(outputdim_final, 1)
            )

            nn.init.zeros_(self.theta_head[-1].weight)
            nn.init.zeros_(self.theta_head[-1].bias)

    def forward(self, x):
        x0 = self.layer0(x)

        x1 = self.layer1(x0)      # 12 -> 6
        x3 = self.layer3(x1)      # 6  -> 3
        x4 = self.layer4(x3)      # 3  -> 3

        x4 = self.layer4_global(x4)
        x4 = self.up(x4)          # 3 -> 6 
        x5 = self.layer5(torch.cat([x4, x1], dim=1))
        x5 = self.up(x5)          # 6 -> 12
        x8 = self.layer8(torch.cat([x5, x0], dim=1))

        weight = self.layer9(x8)
        x0 = weight * x0

        x1 = self.layer1(x0)           # 12 -> 6
        x3 = self.layer3(x1)           # 6  -> 3
        x4 = self.layer4(x3)           # 3  -> 3
        x_ = self.pool_to_corners(x4)  # 3  -> 2

        x = self.layer10(x_)
        x = self.layer11(x)
        delta_flow = self.disp_head(x)

        if self.orien:
            x_ = self.layer10_(x_)
            x_ = self.layer11_(x_)
            delta_theta = self.theta_head(x_)

            return delta_flow, delta_theta
        else:
            return delta_flow


class GMA_ori(nn.Module):
    def __init__(self, args, sz, in_dim = 164):
        super().__init__()
        self.args = args
        global IN_DIM
        IN_DIM = in_dim

        self.cnn = CNN12(144, args.orien)

    def forward(self, corr):
     
        delta_theta = self.cnn(corr)

        return delta_theta


class GMA_mov(nn.Module):
    def __init__(self, args, sz=12, in_dim = 164, orien=False):
        super().__init__()
        self.args = args
        global IN_DIM
        IN_DIM = in_dim
        self.orien = orien

        self.cnn = CNN12m_weight(128, self.orien, sz)
            
    def forward(self, corr, flow):
        if self.orien:      
            delta_flow, delta_theta = self.cnn(torch.cat((corr, flow), dim=1))
            return delta_flow, delta_theta
        else:
            delta_flow = self.cnn(torch.cat((corr, flow), dim=1))
            return delta_flow