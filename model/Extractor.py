import torchvision
import torch.nn as nn
from collections import OrderedDict


class Resnet18FPN(nn.Module):
    def __init__(self):
        super(Resnet18FPN, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        children = list(self.resnet.children())
        self.conv1 = nn.Sequential(*children[:4])  # conv2d, bn, relu, maxpool
        self.conv2 = children[4]
        self.conv3 = children[5]
        self.conv4 = children[6]

    def forward(self, im_data):
        feat = OrderedDict()
        feat_map = self.conv1(im_data)
        feat_map = self.conv2(feat_map)
        feat_map3 = self.conv3(feat_map)
        feat_map4 = self.conv4(feat_map3)
        feat['map3'] = feat_map3
        feat['map4'] = feat_map4

        return feat


class Resnet50FPN(nn.Module):
    def __init__(self):
        super(Resnet50FPN, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        children = list(self.resnet.children())
        self.conv1 = nn.Sequential(*children[:4])  # conv2d, bn, relu, maxpool
        self.conv2 = children[4]
        self.conv3 = children[5]
        self.conv4 = children[6]

    def forward(self, im_data):
        feat = OrderedDict()
        feat_map = self.conv1(im_data)
        feat_map = self.conv2(feat_map)
        feat_map3 = self.conv3(feat_map)
        feat_map4 = self.conv4(feat_map3)
        feat['map3'] = feat_map3
        feat['map4'] = feat_map4

        return feat
