from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from .submodule import *
from .Gwc_submodule import *
#Densenet
class BoundaryRefinement(nn.Module):
    def __init__(self,nChannels, growthRate):
        super(BoundaryRefinement, self).__init__()
        interChannels = 4*growthRate
        #self.conv1 = nn.Conv3d(input_channels, out_channels, 1, 1, 0)
        self.bn1 = nn.BatchNorm3d(nChannels)
        self.conv1 = nn.Conv3d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm3d(interChannels)
        self.conv2 = nn.Conv3d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):

        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)#可以增加一层3D卷积，将通道将为1/2
        out = nn.ReLU(inplace=True)(out)
        return out
        
class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        
        self.conv0 = nn.Sequential(convbn_3d(inplanes, inplanes, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.BR = BoundaryRefinement(inplanes, inplanes)
        
        self.conv1 = nn.Sequential(convbn_3d(inplanes*2, inplanes, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.BR1 = BoundaryRefinement(inplanes , inplanes )
        self.deconv = nn.Sequential(nn.ConvTranspose3d(inplanes * 2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                                    nn.BatchNorm3d(inplanes))
        self.redir1 = convbn_3d(inplanes, inplanes, kernel_size=1, stride=1, pad=0)
        
    def forward(self, x ):
        
        out0 = self.conv0(x)
        out = self.BR(out0)  # in:1/4 out:1/8
        out1 = self.conv1(out)
        out2 = self.BR1(out1)  # in:1/8 out:1/8

        #out7 = self.CA0(out0, out1) + self.redir1(x)
        out7 = self.deconv(out2) + self.redir1(x)

        return out7

class PSMNet(nn.Module):
    def __init__(self, maxdisp):
        super(PSMNet, self).__init__()
        self.maxdisp = maxdisp
        self.num_groups = 40
        self.feature_extraction = feature_extraction()

        self.dres0 = nn.Sequential(convbn_3d(self.num_groups + 12 * 2, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def forward(self, left, right):

        features_left     = self.feature_extraction(left)
        features_right  = self.feature_extraction(right)


        #matching
        gwc_volume = build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"], self.maxdisp // 4,
                                      self.num_groups)
        concat_volume = build_concat_volume(features_left["concat_feature"], features_right["concat_feature"],
                                                self.maxdisp // 4)
        volume = torch.cat((gwc_volume, concat_volume), 1)

        cost0 = self.dres0(volume)
        cost0 = self.dres1(cost0) + cost0

        out1 = self.dres2(cost0)
        #out1 = out1+cost0

        out2 = self.dres3(out1)
        #out2 = out2+cost0

        out3 = self.dres4(out2)
        #out3 = out3+cost0
        cost0 = self.classif0(cost0)
        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) 
        cost3 = self.classif3(out3) 


        cost0 = F.upsample(cost0, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
        cost0 = torch.squeeze(cost0, 1)
        pred0 = F.softmax(cost0, dim=1)
        pred0 = disparityregression(self.maxdisp)(pred0)

        cost1 = F.upsample(cost1, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
        cost1 = torch.squeeze(cost1, 1)
        pred1 = F.softmax(cost1, dim=1)
        pred1 = disparityregression(self.maxdisp)(pred1)

        cost2 = F.upsample(cost2, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
        cost2 = torch.squeeze(cost2,1)
        pred2 = F.softmax(cost2,dim=1)
        pred2 = disparityregression(self.maxdisp)(pred2)

        cost3 = F.upsample(cost3, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
        cost3 = torch.squeeze(cost3,1)
        pred3 = F.softmax(cost3,dim=1)
        #For your information: This formulation 'softmax(c)' learned "similarity" 
        #while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
        #However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
        pred3 = disparityregression(self.maxdisp)(pred3)

        if self.training:
            return [pred0, pred1, pred2, pred3]
        else:
            return [pred3]
