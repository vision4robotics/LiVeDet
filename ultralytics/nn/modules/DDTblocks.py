import os
import torch
import torch.nn as nn
import numpy as np
from functools import partial
import math
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
import torch.nn.functional as F
import cv2
import time
import matplotlib.pyplot as plt
from einops import rearrange
from ultralytics.nn.modules.conv import *
from ultralytics.nn.modules.block import *
from ultralytics.nn.modules.transformer import *
from math import sqrt 
__all__ = ('Reconstructor', 'dynamic_filter', 'DistributionET', 'DDTBlock')

class BasicConv(nn.Module):
    def __init__(self, dim, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(dim, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(dim, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


    
class Reconstructor(nn.Module):
    def __init__(self, dim, drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma_1 = nn.Parameter(torch.ones((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, C, H, W = x.shape
        norm_x = x.flatten(2).transpose(1, 2)
        norm_x = self.norm1(norm_x)
        norm_x = norm_x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = x + self.drop_path(self.gamma_1*self.conv2(self.attn(self.conv1(norm_x))))
        return x


class dynamic_filter(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, group=8):
        super(dynamic_filter, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.conv = nn.Conv2d(dim, group*kernel_size**2, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group*kernel_size**2)
        self.act = nn.Softmax(dim=-2)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        self.lamb_l = nn.Parameter(torch.zeros(dim), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(dim), requires_grad=True)
        self.pad = nn.ReflectionPad2d(kernel_size//2)

        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.modulate = Distiller(dim)

    def forward(self, x):
        dy_filter = self.ap(x)
        dy_filter = self.conv(dy_filter)
        dy_filter = self.bn(dy_filter)     

        n, c, h, w = x.shape  
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(n, self.group, c//self.group, self.kernel_size**2, h*w)

        n,c1,p,q = dy_filter.shape
        dy_filter = dy_filter.reshape(n, c1//self.kernel_size**2, self.kernel_size**2, p*q).unsqueeze(2)
       
        dy_filter = self.act(dy_filter)
    
        out = torch.sum(x * dy_filter, dim=3).reshape(n, c, h, w)

        out = self.modulate(out)
        return out

class Distiller(nn.Module):
    def __init__(self, features):
        super().__init__()
        
        self.features = features
        self.act = nn.SiLU()
        self.fc = nn.Conv2d(features, features, 1, 1, 0)
        self.softmax = nn.Softmax(dim=1)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.out = nn.Conv2d(features, features, 1, 1, 0)
    def forward(self, x):
        y = self.gap(x)
        y = self.fc(y)
        y = self.act(y)
        y = self.fc(y)
        y = self.softmax(y)
        z = x * y
        out = self.out(z) 
        return out

class DistributionET(nn.Module):
    def __init__(self,
                 dim, imgsz=(40,40),
                 num_classes=12):
        super(DistributionET, self).__init__()
        self.window_size = 5
        h, w = imgsz
        self.win_hn = h // self.window_size
        self.win_wn = w // self.window_size
        self.num_classes = num_classes
        self.distribution_head =  nn.Sequential(
            nn.Conv2d(
                in_channels=dim,
                out_channels=self.num_classes*4,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(self.num_classes*4, momentum=0.01),
            nn.ReLU(inplace=False),
            nn.Conv2d(num_classes*4, num_classes, kernel_size=1, groups=self.num_classes) 
        )
        self.deformation_head = nn.Sequential(
                nn.AvgPool2d(kernel_size=self.window_size, stride=self.window_size),
                nn.BatchNorm2d(dim, momentum=0.01),
                nn.ReLU(inplace=False),
                nn.Conv2d(dim, 8, kernel_size=1, stride=1)
        )
        
        base_coords_h = torch.arange(self.window_size).cuda() * 2 / (h-1)
        base_coords_h = (base_coords_h - base_coords_h.mean())
        base_coords_w = torch.arange(self.window_size).cuda() * 2 / (w-1)
        base_coords_w = (base_coords_w - base_coords_w.mean())

        expanded_base_coords_h = base_coords_h.unsqueeze(dim=0).repeat(self.win_hn, 1)
        expanded_base_coords_w = base_coords_w.unsqueeze(dim=0).repeat(self.win_wn, 1)
        expanded_base_coords_h = expanded_base_coords_h.reshape(-1)
        expanded_base_coords_w = expanded_base_coords_w.reshape(-1)
        self.window_coords = torch.stack(torch.meshgrid(expanded_base_coords_w, expanded_base_coords_h), 0).permute(0, 2, 1).reshape(1, 2, self.win_hn, self.window_size, self.win_wn, self.window_size).permute(0, 2, 4, 1, 3, 5).cuda()

    def point_distribution(self, center_heatmap_preds):
        center_heatmap_pred_y = center_heatmap_preds[0]
        center_heatmap_pred = torch.sigmoid(center_heatmap_pred_y)
        heatmap = torch.max(center_heatmap_pred, dim=0, keepdim=True)[0]
        heatmap = 1 - heatmap
        scale_factor = 255 / heatmap.max()
        heatmap = heatmap * scale_factor
        heatmap = heatmap[0]
        Thresh = (9.0/11.0) * 255.0
        pointmap = (heatmap > Thresh).float()
        return pointmap
    
    def point_mask(self, pointmap, feats, k=1):
        '''
            4 : 4
        '''
        binmap = pointmap.clone()
        binmap = (binmap == 255).float()
        density_map = torch.zeros((4, 4))
        w_stride = pointmap.size(1)//4
        h_stride = pointmap.size(0)//4
        for i in range(4):
            for j in range(4):
                x1 = w_stride*i
                y1 = h_stride*j
                x2 = min(x1+w_stride, pointmap.size(1))
                y2 = min(y1+h_stride, pointmap.size(0))
                density_map[i][j] = binmap[y1:y2,x1:x2 == 1].sum()

        topk = 10
        _, idx = torch.topk(density_map.view(-1), topk, largest=True)
        grid_idx = idx.clone()
        idx_x = torch.div(idx, 4) * w_stride
        idx_x = idx_x.view((-1, 1))
        idx_y = idx % 4 * h_stride
        idx_y = idx_y.view((-1, 1))
        idx = torch.cat((idx_x, idx_y), dim=1)
        idx_2 = idx.clone()
        idx_2_column_0 = idx_2[:, 0] + w_stride
        idx_2[:, 0] = torch.clamp(idx_2_column_0, min=0, max=pointmap.shape[1], out=None)
        idx_2_column_1 = idx_2[:, 1] + h_stride
        idx_2[:, 1] = torch.clamp(idx_2_column_1, min=0, max=pointmap.shape[0], out=None)

        grid = torch.zeros((4, 4))
        grid = grid.to(torch.uint8)
        for item in grid_idx:
            x1 = torch.div(item, 4)
            y1 = item % 4
            x1_long = x1.long()
            y1_long = y1.long()
            grid[x1_long, y1_long] = 255
        result = split_overlay_map(grid)
        result = torch.tensor(result)
        result[:, 0::2] = torch.clamp(result[:, 0::2] * w_stride, min=0, max=pointmap.size(1), out=None)
        result[:, 1::2] = torch.clamp(result[:, 1::2] * h_stride, min=0, max=pointmap.size(0), out=None)

        result[:, 2] = result[:, 2] - result[:, 0]
        result[:, 3] = result[:, 3] - result[:, 1]
        
        mask = torch.zeros_like(feats).cuda()
        for x, y, w, h in result:
            mask[:, :, y:y+h, x:x+w] = 1
        feats *= mask  
        feats += (1 - mask) * 10e-6
        return feats
        
    
    
    def region_deformer(self, feats):
        B, C, H, W = feats.shape #
        co_matrix = self.deformation_head(feats).reshape(B, 8, self.win_hn, self.win_wn).permute(0, 2, 3, 1)    #(B,8,nw,nh)
        x_rotation = co_matrix[..., 0:1]
        y_rotation = co_matrix[..., 1:2]
        co_scale_x = co_matrix[..., 2:3]
        co_scale_y = co_matrix[..., 3:4]
        co_scale = co_scale_x * co_scale_y
        co_translation_x = co_matrix[..., 4:6].reshape(-1, self.win_hn, self.win_wn, 2, 1)
        co_translation_y = co_matrix[..., 6:8].reshape(-1, self.win_hn, self.win_wn, 2, 1)
        zero_vector = torch.zeros(B, self.win_hn, self.win_wn).cuda()
        co_add = torch.cat([
                torch.zeros_like(co_translation_x).reshape(-1, self.win_hn, self.win_wn, 1, 2).cuda(),
                torch.ones_like(zero_vector).cuda().reshape(-1, self.win_hn, self.win_wn, 1, 1)
                ], dim=-1)
        
        rotation_matrix_x = torch.stack([
            x_rotation.cos(),
            x_rotation.sin(),
            -x_rotation.sin(),
            x_rotation.cos()
        ], dim=-1).reshape(-1, self.win_hn, self.win_wn, 2, 2)
        rotation_matrix_y = torch.stack([
            y_rotation.cos(),
            y_rotation.sin(),
            -y_rotation.sin(),
            y_rotation.cos()
        ], dim=-1).reshape(-1, self.win_hn, self.win_wn, 2, 2)
        scale_matrix_x = torch.stack([
            co_scale_x[..., 0],
            torch.zeros_like(zero_vector).cuda(),
            torch.zeros_like(zero_vector).cuda(),
            co_scale_x[..., 0]
        ], dim=-1).reshape(-1, self.win_hn, self.win_wn, 2, 2)

        scale_matrix = torch.stack([
            co_scale[..., 0],
            torch.zeros_like(zero_vector).cuda(),
            torch.zeros_like(zero_vector).cuda(),
            co_scale[..., 0]
        ], dim=-1).reshape(-1, self.win_hn, self.win_wn, 2, 2)
        
        transform_matrix_0 = rotation_matrix_x @ rotation_matrix_y @ scale_matrix
        transform_matrix_1 = rotation_matrix_x @ scale_matrix_x @ co_translation_y + co_translation_x
        transform_matrix_co = torch.cat(
                (torch.cat((transform_matrix_0, transform_matrix_1), dim=-1), co_add), dim=-2)

        
        
        window_coords_pers = torch.cat([
                self.window_coords.flatten(-2, -1).cuda(), torch.ones(1, self.win_hn, self.win_wn, 1, self.window_size*self.window_size).cuda()
            ], dim=-2)
        transform_window_coords = transform_matrix_co @ window_coords_pers
        
        _transform_window_coords3 = transform_window_coords[..., -1, :]
        _transform_window_coords3[_transform_window_coords3==0] = 1e-6
        transform_window_coords = transform_window_coords[..., :2, :] / _transform_window_coords3.unsqueeze(dim=-2)

        transform_window_coords = transform_window_coords.reshape(-1,  self.win_hn,  self.win_wn, 2, self.window_size, self.window_size).permute(0, 3, 1, 4, 2, 5)
        coords = transform_window_coords.cuda()

        sample_coords = coords.permute(0, 2, 3, 4, 5, 1).reshape(B, H, W, 2)

        feats = F.grid_sample(feats, grid=sample_coords, padding_mode='zeros', align_corners=True)

        return feats
  
        
    def forward(self, feats):
        feats_y = self.distribution_head(feats)
        pointmap = self.point_distribution(feats_y)

        feats = self.point_mask(pointmap, feats) + feats

        feats = self.region_deformer(feats)
        return feats


class DDTBlock(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.mga = MGA(dim)
        self.attention = selfAttention(2 ,dim ,dim)
        self.drop_path = DropPath(0.2)
        self.norm = nn.LayerNorm(dim)
        self.FFN = nn.Linear(dim, dim)
        self.block =  nn.Sequential(
            nn.BatchNorm2d(dim*2, momentum=0.01),
            nn.ReLU(inplace=False),
            nn.Conv2d(dim*2, dim, kernel_size=1, stride=1)
        )
        self.conv_out = nn.Conv2d(dim, dim*2, kernel_size=1, stride=1)

    def forward(self, feats):
        feats =self.block(feats)
        x = self.mga(feats)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        shortcut = x
        x = self.norm(x)
        x = self.attention(x)
        x = x.view(B, H * W, C)   
        x = shortcut + self.drop_path(x)
 
        x = x + self.drop_path(self.FFN(x))
        feats = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        feats = self.conv_out(feats)
        return feats


class selfAttention(nn.Module) :
    def __init__(self, num_attention_heads, input_size, hidden_size):
        super(selfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0 :
            raise ValueError(
                "the hidden size %d is not a multiple of the number of attention heads"
                "%d" % (hidden_size, num_attention_heads)
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.key_layer = nn.Linear(input_size, hidden_size)
        self.query_layer = nn.Linear(input_size, hidden_size)
        self.value_layer = nn.Linear(input_size, hidden_size)

    def trans_to_multiple_heads(self, x):
        new_size = x.size()[ : -1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        key = self.key_layer(x)
        query = self.query_layer(x)
        value = self.value_layer(x)

        key_heads = self.trans_to_multiple_heads(key)
        query_heads = self.trans_to_multiple_heads(query)
        value_heads = self.trans_to_multiple_heads(value)

        attention_scores = torch.matmul(query_heads, key_heads.permute(0, 1, 3, 2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = F.softmax(attention_scores, dim = -1)

        context = torch.matmul(attention_probs, value_heads)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_size = context.size()[ : -2] + (self.all_head_size , )
        context = context.view(*new_size)
        return context





def split_overlay_map(grid):
        if grid is None or grid.numel() == 0:
            return torch.tensor([])
        # Assume overlap_map is a 2d feature map
        m, n = grid.shape
        visit = torch.zeros((m, n), dtype=torch.bool) 
        count, queue, result = 0, [], []
        for i in range(m):
            for j in range(n):
                if not visit[i][j]:
                    if grid[i][j] == 0:
                        visit[i][j] = 1
                        continue
                    queue.append([i, j])
                    top, left = float("inf"), float("inf")
                    bot, right = float("-inf"), float("-inf")
                    while queue:
                        i_cp, j_cp = queue.pop(0)
                        if 0 <= i_cp < m and 0 <= j_cp < n and grid[i_cp][j_cp] == 255:
                            top = min(i_cp, top)
                            left = min(j_cp, left)
                            bot = max(i_cp, bot)
                            right = max(j_cp, right)
                        if 0 <= i_cp < m and 0 <= j_cp < n and not visit[i_cp][j_cp]:
                            visit[i_cp][j_cp] = 1
                            if grid[i_cp][j_cp] == 255:
                                queue.append([i_cp, j_cp + 1])
                                queue.append([i_cp + 1, j_cp])
                                queue.append([i_cp, j_cp - 1])
                                queue.append([i_cp - 1, j_cp])

                                queue.append([i_cp - 1, j_cp - 1])
                                queue.append([i_cp - 1, j_cp + 1])
                                queue.append([i_cp + 1, j_cp - 1])
                                queue.append([i_cp + 1, j_cp + 1])
                    count += 1
                    result.append([max(0, top), max(0, left), min(bot+1, m), min(right+1, n)])

        return result

class MGA(nn.Module):
    def __init__(self, dim, order=3, s=1.0, kernel=3, n=1):
        super().__init__()
        self.squeeze_dim = dim // 4
        self.squeeze_in = nn.Conv2d(dim, self.squeeze_dim, 1)
        self.squeeze_out = nn.Conv2d(self.squeeze_dim, dim, 1)

        self.order = order
        self.dims = [self.squeeze_dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(self.squeeze_dim, 2 * self.squeeze_dim, 1)
        self.dwconv = nn.Conv2d(sum(self.dims), sum(self.dims), kernel_size=kernel, padding=(kernel-1)//2 ,bias=True, groups=sum(self.dims))
        self.projection = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i+1], 1) for i in range(order-1)]
        )
        self.scale = s
        self.act = nn.SiLU()
        self.softmax = nn.Softmax(dim=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.repconv_blocks = nn.Sequential(*[RepConv(self.squeeze_dim, self.squeeze_dim) for _ in range(n)])
    

    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.squeeze_in(x)
        proj_x = self.proj_in(x)
        projection_0, projection_1 = torch.split(proj_x, (self.dims[0], sum(self.dims)), dim=1)
        pj_split = self.dwconv(projection_1) * self.scale
        pj_list = torch.split(pj_split, self.dims, dim=1)
        proj_x = projection_0 * pj_list[0]

        for i in range(self.order - 1):
            proj_x = self.projection[i](proj_x) * pj_list[i+1]
        
        y = pj_list[self.order-1]

        th = self.gap(y)
        th = self.act(th)
        th = self.softmax(th)
        y = th * y
        x = self.act(proj_x * y) + x
        x = self.repconv_blocks(x)
        x = self.squeeze_out(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x