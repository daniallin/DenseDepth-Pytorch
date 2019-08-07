import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resdeep.resnet import resnet101
from models.resdeep.aspp import ASPP
from models.resdeep.decoder import Decoder


class ResDeep(nn.Module):
    def __init__(self, output_scale, sync_bn=False, pretrained=False):
        super(ResDeep, self).__init__()

        self.encoder = resnet101(pretrained, sync_bn=sync_bn, output_scale=output_scale)
        self.aspp = ASPP(output_scale, sync_bn)
        self.decoder = Decoder(sync_bn)

    def forward(self, input):
        x, low_level_feature = self.encoder(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feature)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x


if __name__ == '__main__':
    model = ResDeep(16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())

