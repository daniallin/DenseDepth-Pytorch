import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resdeep.general import BatchNorm, initial_weight


class Decoder(nn.Module):
    def __init__(self, sync_bn=False):
        super(Decoder, self).__init__()
        low_feature_size = 256

        self.conv1 = nn.Conv2d(low_feature_size, 48, 1, bias=False)
        self.bn1 = BatchNorm(48, sync_bn)
        self.relu = nn.ReLU()
        # here 304 = 256 + 48, is the sum size of low level feature and output feature
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256, sync_bn),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256, sync_bn),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, 1, kernel_size=1, stride=1))

        initial_weight(self.modules())

    def forward(self, x, low_level_feature):
        low_level_feature = self.conv1(low_level_feature)
        low_level_feature = self.bn1(low_level_feature)
        low_level_feature = self.relu(low_level_feature)

        x = F.interpolate(x, size=low_level_feature.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feature), dim=1)

        return self.last_conv(x)


if __name__ == '__main__':
    model = Decoder()
    model.eval()
    x = torch.randn(1, 256, 20, 20)
    y = model(x, x)
    print(y.size())

