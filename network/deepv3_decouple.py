"""
Implementation of Decoupled network architectures including FCN, PSPNet, DeeplabV3
Author: Xiangtai Li (lxtpku@pku.edu.cn)
"""

import logging
import cv2
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from network import resnet_d as Resnet_Deep
from network.wider_resnet import wider_resnet38_a2
from network.deepv3 import _AtrousSpatialPyramidPoolingModule
from network.nn.mynn import initialize_weights, Norm2d, Upsample
from network.nn.operators import PSPModule
from network.nn.att_module import ChannelAttention, SpatialAttention
# from network.nn.dran import DranHead
from PIL import Image
import random
from torchvision import transforms
from torchvision import models
import time

"""
特征图可视化
"""
def featuremap_visual(feature,
                      out_dir=None,  # 特征图保存路径文件
                      save_feature=True,  # 是否以图片形式保存特征图
                      show_feature=False,  # 是否使用plt显示特征图
                      feature_title=None,  # 特征图名字，默认以shape作为title
                      num_ch=-1,  # 显示特征图前几个通道，-1 or None 都显示
                      nrow=16,  # 每行显示多少个特征图通道
                      padding=1,  # 特征图之间间隔多少像素值
                      pad_value=1  # 特征图之间的间隔像素
                      ):
    import matplotlib.pylab as plt
    import torchvision
    import os
    # feature = feature.detach().cpu()
    b, c, h, w = feature.shape
    feature = feature[0]
    feature = feature.unsqueeze(1)

    if c > num_ch > 0:
        feature = feature[:num_ch]

    img = torchvision.utils.make_grid(feature, nrow=nrow, padding=padding, pad_value=pad_value)
    img = img.detach().cpu()
    img = img.numpy()
    images = img.transpose((1, 2, 0))

    # title = str(images.shape) if feature_title is None else str(feature_title)
    title = str('hwc-') + str(h) + '-' + str(w) + '-' + str(c) if feature_title is None else str(feature_title)

    plt.title(title)
    plt.imshow(images)
    plt.hsv()
    if save_feature:
        # root=r'C:\Users\Administrator\Desktop\CODE_TJ\123'
        # plt.savefig(os.path.join(root,'1.jpg'))
        out_root = title + '.pmg' if out_dir == '' or out_dir is None else os.path.join(out_dir, title + '.png')
        plt.savefig(out_root, dpi=3200)

    if show_feature:
        plt.show()


class DeepWV3PlusDecoupleEdgeBody(nn.Module):
    """
    WideResNet38 version of DeepLabV3
    mod1
    pool2
    mod2 bot_fine
    pool3
    mod3-7
    bot_aspp

    structure: [3, 3, 6, 3, 1, 1]
    channels = [(128, 128), (256, 256), (512, 512), (512, 1024), (512, 1024, 2048),
              (1024, 2048, 4096)]
    """

    def __init__(self, num_classes, trunk='WideResnet38', criterion=None):

        super(DeepWV3PlusDecoupleEdgeBody, self).__init__()
        self.criterion = criterion
        logging.info("Trunk: %s", trunk)

        wide_resnet = wider_resnet38_a2(classes=1000, dilation=True)
        wide_resnet = torch.nn.DataParallel(wide_resnet)
        if criterion is not None:
            try:
                checkpoint = torch.load('./pretrained_models/wider_resnet38.pth.tar', map_location='cpu')
                wide_resnet.load_state_dict(checkpoint)
                del checkpoint
            except:
                print(
                    "Please download the ImageNet weights of WideResNet38 in our repo to ./pretrained_models/wider_resnet38.pth.tar.")
                raise RuntimeError(
                    "=====================Could not load ImageNet weights of WideResNet38 network.=======================")

        wide_resnet = wide_resnet.module

        self.mod1 = wide_resnet.mod1
        self.mod2 = wide_resnet.mod2
        self.mod3 = wide_resnet.mod3
        self.mod4 = wide_resnet.mod4
        self.mod5 = wide_resnet.mod5
        self.mod6 = wide_resnet.mod6
        self.mod7 = wide_resnet.mod7
        self.pool2 = wide_resnet.pool2
        self.pool3 = wide_resnet.pool3
        del wide_resnet

        self.aspp = _AtrousSpatialPyramidPoolingModule(4096, 256,
                                                       output_stride=8)

        self.bot_aspp = nn.Conv2d(1280, 256, kernel_size=1, bias=False)

        self.bot_fine = nn.Conv2d(128, 48, kernel_size=1, bias=False)

        edge_dim = 256
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, edge_dim, kernel_size=1, bias=False),
            Norm2d(edge_dim), nn.ReLU(inplace=True))

        self.squeeze_body_edge = SqueezeBodyEdge(256, Norm2d)
        # fusion different edges
        self.edge_fusion = nn.Conv2d(256 + 48, 256, 1, bias=False)
        self.sigmoid_edge = nn.Sigmoid()

        self.edge_out = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=3, padding=1, bias=False),
            Norm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 1, kernel_size=1, bias=False)
        )

        self.dsn_seg_body = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False)
        )

        self.final_seg = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False))

        initialize_weights(self.final_seg, self.dsn_seg_body)

    def forward(self, inp, gts=None):

        x_size = inp.size()
        x = self.mod1(inp)
        m2 = self.mod2(self.pool2(x))
        fine_size = m2.size()
        x = self.mod3(self.pool3(m2))
        x = self.mod4(x)
        x = self.mod5(x)
        x = self.mod6(x)
        x = self.mod7(x)
        x = self.aspp(x)
        aspp = self.bot_aspp(x)

        seg_body, seg_edge = self.squeeze_body_edge(aspp)

        # may add canny edge
        # canny_edge = self.edge_canny(inp, x_size)
        # add low-level feature
        dec0_fine = self.bot_fine(m2)
        seg_edge = self.edge_fusion(torch.cat([Upsample(seg_edge, fine_size[2:]), dec0_fine], dim=1))
        seg_edge_out = self.edge_out(seg_edge)

        seg_out = seg_edge + Upsample(seg_body, fine_size[2:])
        aspp = Upsample(aspp, fine_size[2:])

        seg_out = torch.cat([aspp, seg_out], dim=1)
        seg_final = self.final_seg(seg_out)

        seg_edge_out = Upsample(seg_edge_out, x_size[2:])
        seg_edge_out = self.sigmoid_edge(seg_edge_out)

        seg_final_out = Upsample(seg_final, x_size[2:])

        seg_body_out = Upsample(self.dsn_seg_body(seg_body), x_size[2:])

        # dec0_up = Upsample(dec0_up, m2.size()[2:])
        # dec0 = [dec0_fine, dec0_up]
        # dec0 = torch.cat(dec0, 1)
        # dec1 = self.final(dec0)
        # seg_out = Upsample(dec1, x_size[2:])

        if self.training:
            return self.criterion((seg_final_out, seg_body_out, seg_edge_out), gts)

        return seg_final_out

    def edge_canny(self, inp, x_size):
        im_arr = inp.cpu().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
        canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        for i in range(x_size[0]):
            canny[i] = cv2.Canny(im_arr[i], 10, 100)
        canny = torch.from_numpy(canny).cuda().float()
        return canny


class SqueezeBodyEdge(nn.Module):
    def __init__(self, inplane, norm_layer):
        """
        implementation of body generation part
        :param inplane:
        :param norm_layer:
        """
        super(SqueezeBodyEdge, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            norm_layer(inplane),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            norm_layer(inplane),
            nn.ReLU(inplace=True)
        )

        self.flow_make = nn.Conv2d(inplane * 2, 2, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        size = x.size()[2:]
        seg_down = self.down(x)
        seg_down = F.upsample(seg_down, size=size, mode="bilinear", align_corners=True)
        flow = self.flow_make(torch.cat([x, seg_down], dim=1))
        seg_flow_warp = self.flow_warp(x, flow, size)
        seg_edge = x - seg_flow_warp
        return seg_flow_warp, seg_edge

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        # new
        h_grid = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w_gird = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w_gird.unsqueeze(2), h_grid.unsqueeze(2)), 2)

        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output


class DeepV3Plus(nn.Module):
    """
    Implement DeepLabV3 model
    A: stride8
    B: stride16
    with skip connections
    """

    def __init__(self, num_classes, trunk='seresnext-50', criterion=None, variant='D',
                 skip='m1', skip_num=48):
        super(DeepV3Plus, self).__init__()
        self.criterion = criterion
        self.variant = variant
        self.skip = skip
        self.skip_num = skip_num

        if trunk == 'resnet-50-deep':
            resnet = Resnet_Deep.resnet50()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnet-50-deep-4channel':
            resnet = Resnet_Deep.resnet50_4channel()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnet-101-deep':
            resnet = Resnet_Deep.resnet101()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        else:
            raise ValueError("Not a valid network arch")

        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        if self.variant == 'D':
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        elif self.variant == 'D16':
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        else:
            print("Not using Dilation ")

        self.aspp = _AtrousSpatialPyramidPoolingModule(2048, 256,
                                                       output_stride=8)

        if self.skip == 'm1':
            self.bot_fine = nn.Conv2d(256, self.skip_num, kernel_size=1, bias=False)
        elif self.skip == 'm2':
            self.bot_fine = nn.Conv2d(512, self.skip_num, kernel_size=1, bias=False)
        else:
            raise Exception('Not a valid skip')

        self.bot_aspp = nn.Conv2d(1280, 256, kernel_size=1, bias=False)

        # body_edge module
        self.squeeze_body_edge = SqueezeBodyEdge(256, Norm2d)

        # fusion different edge part
        self.edge_fusion = nn.Conv2d(256 + 48, 256, 1, bias=False)
        self.sigmoid_edge = nn.Sigmoid()
        self.edge_out = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=3, padding=1, bias=False),
            Norm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 1, kernel_size=1, bias=False)
        )

        # DSN for seg body part
        self.dsn_seg_body = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False)
        )

        # Final segmentation part
        self.final_seg = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False))

        self.channel_attention_256 = ChannelAttention(256)
        self.channel_attention_512 = ChannelAttention(512)
        self.spatial_attention = SpatialAttention()
        self.relu = nn.ReLU()

        # self.head = DranHead(512, num_classes, nn.BatchNorm2d)

    def forward(self, x, gts=None):

        x_size = x.size()  # 800
        x0 = self.layer0(x)  # 400
        x1 = self.layer1(x0)  # 400
        fine_size = x1.size()
        x2 = self.layer2(x1)  # 100
        x3 = self.layer3(x2)  # 100
        x4 = self.layer4(x3)  # 100
        xp = self.aspp(x4)

        aspp = self.bot_aspp(xp)

        name = str(int(round(time.time() * 1000)))

        #featuremap_visual(aspp, out_dir="./tmp/aspp/", feature_title=str(name)+"_aspp_before_attention")

        # aspp与经过注意力模块的aspp以残差形式相加
        # aspp 注意力模块并联
        """
        aspp_channel_attention = aspp * self.channel_attention_256(aspp)
        aspp_spatial_attention = aspp * self.spatial_attention(aspp)
        aspp_channel_attention = aspp_channel_attention + aspp
        aspp_spatial_attention = aspp_spatial_attention + aspp
        aspp = self.relu(aspp_channel_attention + aspp_spatial_attention + aspp)
        """
        #featuremap_visual(aspp, out_dir="./tmp/aspp/", feature_title=str(name)+"_aspp_after_attention")

        seg_body, seg_edge = self.squeeze_body_edge(aspp)

        if self.skip == 'm1':
            # use default low-level feature
            dec0_fine = self.bot_fine(x1)
        else:
            dec0_fine = self.bot_fine(x2)

        seg_edge = self.edge_fusion(torch.cat([Upsample(seg_edge, fine_size[2:]), dec0_fine], dim=1))

        # featuremap_visual(seg_edge, out_dir="./tmp/seg_edge/", feature_title=str(name)+"_seg_edge_before_attention")
        """
        seg_edge_channel_attention = seg_edge * self.channel_attention_256(seg_edge)
        seg_edge_spatial_attention = seg_edge * self.spatial_attention(seg_edge)
        seg_edge_channel_attention = seg_edge_channel_attention + seg_edge
        seg_edge_spatial_attention = seg_edge_spatial_attention + seg_edge
        seg_edge = self.relu(seg_edge_channel_attention + seg_edge_spatial_attention + seg_edge)        
        """
        # seg_edge与经过注意力模块的seg_edge以残差形式相加
        # seg_edge 注意力模块并联

        # featuremap_visual(seg_edge, out_dir="./tmp/seg_edge/", feature_title=str(name)+"_seg_edge_after_attention")

        seg_edge_out = self.edge_out(seg_edge)

        seg_body = Upsample(seg_body, fine_size[2:])

        seg_out = seg_edge + seg_body
        aspp = Upsample(aspp, fine_size[2:])

        seg_out = torch.cat([aspp, seg_out], dim=1)


        #featuremap_visual(seg_out, out_dir="./tmp/seg_out/", feature_title=str(name) + "_seg_out_before_attention")
        # ablation study
        """
        # seg_out与经过注意力模块的seg_out以残差形式相加
        # seg_out_final 注意力模块并联
        seg_out_channel_attention = seg_out * self.channel_attention_512(seg_out)
        seg_out_spatial_attention = seg_out * self.spatial_attention(seg_out)
        seg_out_channel_attention = seg_out_channel_attention + seg_out
        seg_out_spatial_attention = seg_out_spatial_attention + seg_out
        seg_out = self.relu(seg_out_channel_attention + seg_out_spatial_attention + seg_out)         
        """

        #featuremap_visual(seg_out, out_dir="./tmp/seg_out/", feature_title=str(name)+"_seg_out_after_attention")

        """
        # 注意力模块串联
        seg_out_attention = seg_out * self.channel_attention_512(seg_out)
        seg_out_attention = seg_out_attention * self.spatial_attention(seg_out_attention)
        seg_out = seg_out_attention + seg_out
        seg_out = self.relu(seg_out)
        """

        seg_final = self.final_seg(seg_out)

        seg_edge_out = Upsample(seg_edge_out, x_size[2:])
        seg_edge_out = self.sigmoid_edge(seg_edge_out)

        seg_final_out = Upsample(seg_final, x_size[2:])

        # featuremap_visual(seg_final_out, out_dir="./tmp/aspp/", feature_title=str(name)+"_seg_final_out")

        seg_body = self.dsn_seg_body(seg_body)
        seg_body_out = Upsample(seg_body, x_size[2:])

        if self.training:
            return self.criterion((seg_final_out, seg_body_out, seg_edge_out), gts)

        return seg_final_out


class DeepFCN(nn.Module):
    """
    Implement DeepFCN model
    Note that our FCN model is a strong baseline that
    A: stride8
    B: stride16
    with skip connections
    """

    def __init__(self, num_classes, trunk='seresnext-50', criterion=None, variant='D',
                 skip='m1', skip_num=48):
        super(DeepFCN, self).__init__()
        self.criterion = criterion
        self.variant = variant
        self.skip = skip
        self.skip_num = skip_num

        if trunk == 'resnet-50-deep':
            resnet = Resnet_Deep.resnet50()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnet-101-deep':
            resnet = Resnet_Deep.resnet101()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        else:
            raise ValueError("Not a valid network arch")

        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        if self.variant == 'D':
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        elif self.variant == 'D16':
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        else:
            print("Not using Dilation ")

        self.fcn_head = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
        )

        if self.skip == 'm1':
            self.bot_fine = nn.Conv2d(256, self.skip_num, kernel_size=1, bias=False)
        elif self.skip == 'm2':
            self.bot_fine = nn.Conv2d(512, self.skip_num, kernel_size=1, bias=False)
        else:
            raise Exception('Not a valid skip')

        # body edge generation module
        self.squeeze_body_edge = SqueezeBodyEdge(256, Norm2d)

        # fusion different edge part
        self.edge_fusion = nn.Conv2d(256 + 48, 256, 1, bias=False)
        self.sigmoid_edge = nn.Sigmoid()
        self.edge_out = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=3, padding=1, bias=False),
            Norm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 1, kernel_size=1, bias=False)
        )

        # DSN for seg body part
        self.dsn_seg_body = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False)
        )

        # Final segmentation part
        self.final_seg = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False))

    def forward(self, x, gts=None):
        x_size = x.size()
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        fine_size = x1.size()
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        aspp = self.fcn_head(x4)

        seg_body, seg_edge = self.squeeze_body_edge(aspp)

        if self.skip == 'm1':
            # use default low-level feature
            dec0_fine = self.bot_fine(x1)
        else:
            dec0_fine = self.bot_fine(x2)

        seg_edge = self.edge_fusion(torch.cat([Upsample(seg_edge, fine_size[2:]), dec0_fine], dim=1))
        seg_edge_out = self.edge_out(seg_edge)

        seg_out = seg_edge + Upsample(seg_body, fine_size[2:])
        aspp = Upsample(aspp, fine_size[2:])

        seg_out = torch.cat([aspp, seg_out], dim=1)
        seg_final = self.final_seg(seg_out)

        seg_edge_out = Upsample(seg_edge_out, x_size[2:])
        seg_edge_out = self.sigmoid_edge(seg_edge_out)

        seg_final_out = Upsample(seg_final, x_size[2:])

        seg_body_out = Upsample(self.dsn_seg_body(seg_body), x_size[2:])

        if self.training:
            return self.criterion((seg_final_out, seg_body_out, seg_edge_out), gts)

        return seg_final_out


class PSPNet(nn.Module):
    def __init__(self, num_classes, trunk='seresnext-50', criterion=None, variant='D',
                 skip='m1', skip_num=48):
        super(PSPNet, self).__init__()
        self.criterion = criterion
        self.variant = variant
        self.skip = skip
        self.skip_num = skip_num

        if trunk == 'resnet-50-deep':
            resnet = Resnet_Deep.resnet50()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)

        elif trunk == 'resnet-101-deep':
            resnet = Resnet_Deep.resnet101()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        else:
            raise ValueError("Not a valid network arch")

        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        if self.variant == 'D':
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        elif self.variant == 'D16':
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        else:
            print("Not using Dilation ")

        self.ppm = PSPModule(2048, 256, norm_layer=Norm2d)

        if self.skip == 'm1':
            self.bot_fine = nn.Conv2d(256, self.skip_num, kernel_size=1, bias=False)
        elif self.skip == 'm2':
            self.bot_fine = nn.Conv2d(512, self.skip_num, kernel_size=1, bias=False)
        else:
            raise Exception('Not a valid skip')

        # body_edge module
        self.squeeze_body_edge = SqueezeBodyEdge(256, Norm2d)

        # fusion different edge part
        self.edge_fusion = nn.Conv2d(256 + 48, 256, 1, bias=False)
        self.sigmoid_edge = nn.Sigmoid()
        self.edge_out = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=3, padding=1, bias=False),
            Norm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 1, kernel_size=1, bias=False)
        )

        # DSN for seg body part
        self.dsn_seg_body = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False)
        )

        # Final segmentation part
        self.final_seg = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False))

    def forward(self, x, gts=None):
        x_size = x.size()  # 800
        x0 = self.layer0(x)  # 400
        x1 = self.layer1(x0)  # 400
        fine_size = x1.size()
        x2 = self.layer2(x1)  # 100
        x3 = self.layer3(x2)  # 100
        x4 = self.layer4(x3)  # 100
        aspp = self.ppm(x4)

        seg_body, seg_edge = self.squeeze_body_edge(aspp)

        if self.skip == 'm1':
            # use default low-level feature
            dec0_fine = self.bot_fine(x1)
        else:
            dec0_fine = self.bot_fine(x2)

        seg_edge = self.edge_fusion(torch.cat([Upsample(seg_edge, fine_size[2:]), dec0_fine], dim=1))
        seg_edge_out = self.edge_out(seg_edge)

        seg_out = seg_edge + Upsample(seg_body, fine_size[2:])
        aspp = Upsample(aspp, fine_size[2:])

        seg_out = torch.cat([aspp, seg_out], dim=1)
        seg_final = self.final_seg(seg_out)

        seg_edge_out = Upsample(seg_edge_out, x_size[2:])
        seg_edge_out = self.sigmoid_edge(seg_edge_out)

        seg_final_out = Upsample(seg_final, x_size[2:])

        seg_body_out = Upsample(self.dsn_seg_body(seg_body), x_size[2:])

        if self.training:
            return self.criterion((seg_final_out, seg_body_out, seg_edge_out), gts)

        return seg_final_out


# add other models settings
# deeplab v3+, pspnet, FCN


def DeepR50V3PlusD_m1_deeply(num_classes, criterion):
    """
    ResNet-50 Based Network
    """
    return DeepV3Plus(num_classes, trunk='resnet-50-deep', criterion=criterion, variant='D', skip='m1')


def DeepR50V3PlusD_m1_deeply_4channel(num_classes, criterion):
    """
    ResNet-50 Based 4 Channel Network
    """
    return DeepV3Plus(num_classes, trunk='resnet-50-deep-4channel', criterion=criterion, variant='D', skip='m1')


def DeepR101V3PlusD_m1_deeply(num_classes, criterion):
    """
    ResNet-50 Based Network
    """
    return DeepV3Plus(num_classes, trunk='resnet-101-deep', criterion=criterion, variant='D', skip='m1')


def DeepR50FCN_m1_deeply(num_classes, criterion):
    """
    ResNet-50 Based Network
    """
    return DeepFCN(num_classes, trunk='resnet-50-deep', criterion=criterion, variant='D', skip='m1')


def DeepR101FCN_m1_deeply(num_classes, criterion):
    """
    ResNet-101 Based Network
    """
    return DeepFCN(num_classes, trunk='resnet-50-deep', criterion=criterion, variant='D', skip='m1')


def DeepR50PSP_m1_deeply(num_classes, criterion):
    """
    ResNet-50 Based Network
    """
    return PSPNet(num_classes, trunk='resnet-50-deep', criterion=criterion, variant='D', skip='m1')
