from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
import daa as DAA
import raa as raa
import MPF as F1
import do_conv_pytorch as doconv
def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(doconv.DOConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
                         nn.BatchNorm3d(out_planes))

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out

class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = torch.Tensor(np.reshape(np.array(range(maxdisp)),[1, maxdisp,1,1])).cuda()

    def forward(self, x):
        out = torch.sum(x*self.disp.data,1, keepdim=True)
        return out

class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        inplanes1=64
        k_size=3
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1,1,1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2,1,1) 
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1,1,2)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1,1,4)
#############DAA
        self.conv5a = nn.Sequential(nn.Conv2d(inplanes1*2, inplanes1, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inplanes1),
                                   nn.ReLU())
        
        self.conv5c = nn.Sequential(nn.Conv2d(inplanes1*2, inplanes1, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inplanes1),
                                   nn.ReLU())
        self.sa = DAA.PAM_Module(inplanes1)
        #self.sc = DA.CAM_Module(inplanes//2)
		
		##############RAA
        self.raa = raa.eca_layer(inplanes1, k_size)
		##
		
        self.conv51 = nn.Sequential(nn.Conv2d(inplanes1, inplanes1, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inplanes1),
                                   nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inplanes1, inplanes1, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inplanes1),
                                   nn.ReLU())

        #self.conv53 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inplanes//2, inplanes*2, 1))
        self.conv7 = nn.Conv2d(16, 64, kernel_size=1, padding=0, stride = 1, bias=False)
#####################
        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64,64)),
                                     convbn(64, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32,32)),
                                     convbn(64, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16,16)),
                                     convbn(64, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8,8)),
                                     convbn(64, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))
        self.fpt=F1.FPT(32)
        self.lastconv = nn.Sequential(convbn(384, 320, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True))
        self.lastconv1 = nn.Sequential(convbn(384, 128, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 12, kernel_size=1, padding=0, stride = 1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output      = self.firstconv(x)
        output      = self.layer1(output)
        output_raw  = self.layer2(output)
        output      = self.layer3(output_raw)
        output_skip = self.layer4(output)
################DANet
            #PAM
        feat1 = self.conv5a(output_skip)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        #先降维 CAM
        feat2 = self.conv5c(output_skip)
        sc_feat = self.raa(feat2)
        sc_conv = self.conv52(sc_feat)
        #sc_output = self.conv7(sc_conv)
        feat_sum = sa_conv+sc_conv
        #print("DA",feat_sum.size())
####################

        output_branch1 = self.branch1(feat_sum)
        #output_branch1 = F.upsample(output_branch1, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch2 = self.branch2(feat_sum)
        #output_branch2 = F.upsample(output_branch2, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch3 = self.branch3(feat_sum)
        #output_branch3 = F.upsample(output_branch3, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch4 = self.branch4(feat_sum)
        #output_branch4 = F.upsample(output_branch4, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')
        output_branch1,output_branch2,output_branch3,output_branch4=self.fpt(output_branch1,output_branch2,output_branch3,output_branch4)
        #金字塔上采样
        output_branch1 = F.upsample(output_branch1, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')
        output_branch2 = F.upsample(output_branch2, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')
        output_branch3 = F.upsample(output_branch3, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')
        output_branch4 = F.upsample(output_branch4, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')
        
        gwc_feature = torch.cat((feat_sum,output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        #print("out",gwc_feature.size())
        gwc_feature1 = self.lastconv(gwc_feature)
        concat_feature = self.lastconv1(gwc_feature)

        return {"gwc_feature": gwc_feature1, "concat_feature": concat_feature}



