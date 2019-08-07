import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class UpSample(nn.Sequential):
    def __init__(self, input_channel, output_channel):
        super(UpSample, self).__init__()        
        self.conv1 = nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.relu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.relu2 = nn.LeakyReLU(0.2)

    def forward(self, input):
        up_x = F.interpolate(input[0], size=input[1].size()[2:], mode='bilinear', align_corners=True)
        x = self.relu1(self.conv1(torch.cat((up_x, input[1]), dim=1)))
        return self.relu2(self.conv2(x))


class Decoder(nn.Module):
    def __init__(self, num_features=2208, decoder_width=0.5):
        super(Decoder, self).__init__()
        features = int(num_features * decoder_width)

        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=1)

        self.up1 = UpSample(features//1 + 384, features//2)
        self.up2 = UpSample(features//2 + 192, features//4)
        self.up3 = UpSample(features//4 + 96, features//8)
        self.up4 = UpSample(features//8 + 96, features//16)
        self.relu3 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(features//16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[3], features[4], features[6], features[8], features[11]
        x_d0 = self.conv2(x_block4)
        x_d1 = self.up1([x_d0, x_block3])
        x_d2 = self.up2([x_d1, x_block2])
        x_d3 = self.up3([x_d2, x_block1])
        x_d4 = self.up4([x_d3, x_block0])
        x_d5 = F.interpolate(x_d4, size=(x_d4.size()[2]*2, x_d4.size()[3]*2), mode='bilinear', align_corners=True)
        return self.conv3(self.relu3(x_d5))


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.original_model = models.densenet161(pretrained=True)

    def forward(self, x):
        features = [x]
        for k, v in self.original_model.features._modules.items():
            features.append(v(features[-1]))
        return features


class DenseDepthModel(nn.Module):
    def __init__(self):
        super(DenseDepthModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

