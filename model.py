import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.layers import ResBlock, AttenLayer, fake_atten_layer
class channel_AttenLayer(nn.Module):
    def __init__(self, in_channel, reduction):
        super(channel_AttenLayer, self).__init__()
        self.in_channel = in_channel
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // reduction, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // reduction, in_channel, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        
    def forward(self, input):
        atten_weight = self.pool(input)
        atten_weight = self.conv(atten_weight)

        return atten_weight

class spatial_AttenLayer(nn.Module):
    def __init__(self):
        super(spatial_AttenLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        atten_weight = torch.mean(input, dim=1, keepdim=True)
        atten_weight = self.conv(atten_weight)

        return atten_weight * input

class basic_resblock(nn.Module):
    def __init__(self, in_channel):
        super(basic_resblock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(in_channel),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(in_channel)
                                   )
    def forward(self, input):
        output1 = self.conv1(input)
        output = F.relu(input + output1)

        return output

class basic_block_down(nn.Module):
    def __init__(self, in_channel):
        super(basic_block_down, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.block = nn.Sequential(nn.Conv2d(in_channel, in_channel*2, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(in_channel*2),
                                   nn.ReLU(inplace=True),
                                   basic_resblock(in_channel*2))

    def forward(self, input):
        output = self.pool(input)
        output = self.block(output)

        return output

class basic_block_up(nn.Module):
    def __init__(self, in_channel):
        super(basic_block_up, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channel, in_channel//2, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(in_channel//2),
                                   nn.ReLU(inplace=True),
                                   basic_resblock(in_channel//2))
        self.upsampling = nn.Sequential(nn.ConvTranspose2d(in_channel//2, in_channel//2, kernel_size=2, stride=2, bias=False),
                                        # nn.BatchNorm2d(in_channel//2),
                                        nn.LeakyReLU(inplace=True))
    def forward(self, input):
        output = self.block(input)
        output = self.upsampling(output)

        return output

class Cascade_AttenUnet(nn.Module):
    def __init__(self, init_depth, gamma):
        ### init_depth=16
        super(Cascade_AttenUnet, self).__init__()
        self.gamma = gamma
        self.down_cascade1 = nn.Sequential(nn.Conv2d(1, init_depth, kernel_size=3, stride=1, padding=1, bias=False),
                                           nn.BatchNorm2d(init_depth),
                                           nn.ReLU(inplace=True))
        self.down_cascade2 = nn.Sequential(nn.Conv2d(init_depth, init_depth*2, kernel_size=3, stride=1, padding=1, bias=False),
                                           nn.BatchNorm2d(init_depth*2),
                                           nn.ReLU(inplace=True))

        self.down_block_1 = basic_block_down(init_depth*2)
        self.down_block_2 = basic_block_down(init_depth*4)
        self.down_block_3 = basic_block_down(init_depth*8)
        self.down_block_4 = basic_block_down(init_depth*16)

        self.bridge = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                    nn.Conv2d(init_depth*32, init_depth*32, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.ReLU(inplace=True),
                                    nn.ConvTranspose2d(init_depth*32, init_depth*32, kernel_size=2, stride=2, bias=False),
                                    # nn.BatchNorm2d(init_depth*32),
                                    nn.LeakyReLU(inplace=True)
                                    )

        self.up_block_4 = basic_block_up(init_depth*32)
        self.up_block_3 = basic_block_up(init_depth*16)
        self.up_block_2 = basic_block_up(init_depth*8)
        # self.up_block_1 = basic_block_up(init_depth*4)
        self.up_block_1_1 = nn.Sequential(nn.Conv2d(init_depth*4, init_depth*2, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(init_depth*2),
                                   nn.ReLU(inplace=True),
                                   basic_resblock(init_depth*2))
        self.up_block_1_2 = nn.Sequential(nn.ConvTranspose2d(init_depth*2, init_depth*2, kernel_size=2, stride=2, bias=False),
                                        # nn.BatchNorm2d(in_channel//2),
                                        nn.LeakyReLU(inplace=True))

        self.up_cascade2 = nn.Sequential(nn.Conv2d(init_depth*2, init_depth, kernel_size=3, stride=1, padding=1, bias=False),
                                           nn.BatchNorm2d(init_depth),
                                           nn.ReLU(inplace=True))
        self.up_cascade1 = nn.Sequential(nn.Conv2d(init_depth, 1, kernel_size=3, stride=1, padding=1, bias=False),
                                         nn.Tanh())

        self.ChannelAttenLayers = nn.ModuleList([channel_AttenLayer(init_depth*i, 8) for i in [4, 8, 16, 32]])
        self.SpatialAttenLayers = nn.ModuleList([spatial_AttenLayer() for i in [4, 8, 16, 32]])

        self.cascade1 = nn.Sequential(
            nn.Conv2d(in_channels=init_depth*2, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

        self.cascade2 = nn.Sequential(nn.ConvTranspose2d(init_depth*2, init_depth*2, kernel_size=2, stride=2, bias=False),
                                        # nn.BatchNorm2d(in_channel//2),
                                      nn.LeakyReLU(inplace=True),
                                      nn.Conv2d(init_depth*2, 1, kernel_size=3, stride=1, padding=1, bias=False),
                                      nn.Tanh())


    def forward(self, input):

        down_cascade_output1 = self.down_cascade1(input)
        down_cascade_output2 = self.down_cascade2(down_cascade_output1)

        down_block_output1 = self.down_block_1(down_cascade_output2)
        down_block_output2 = self.down_block_2(down_block_output1)
        down_block_output3 = self.down_block_3(down_block_output2)
        down_block_output4 = self.down_block_4(down_block_output3)

        up_block_output4 = self.bridge(down_block_output4)

        # up_block_output3 = self.up_block_4(up_block_output4 + self.SpatialAttenLayers[3](down_block_output4 * self.ChannelAttenLayers[3](down_block_output4)))
        up_block_output3 = self.up_block_4(up_block_output4)
        up_block_output2 = self.up_block_3(up_block_output3 + self.SpatialAttenLayers[2](down_block_output3 * self.ChannelAttenLayers[2](down_block_output3)))
        up_block_output1 = self.up_block_2(up_block_output2 + self.SpatialAttenLayers[1](down_block_output2 * self.ChannelAttenLayers[1](down_block_output2)))

        up_cascade_output2_1 = self.up_block_1_1(up_block_output1 + self.SpatialAttenLayers[0](down_block_output1 * self.ChannelAttenLayers[0](down_block_output1)))
        cascade_2 = self.cascade2(up_cascade_output2_1)
        cascade_2_mask = torch.pow(1 - cascade_2 * cascade_2, self.gamma)

        up_cascade_output2 = self.up_block_1_2(up_cascade_output2_1)
        cascade_1 = self.cascade1(up_cascade_output2 + cascade_2_mask * down_cascade_output2)
        cascade_1_mask = torch.pow(1 - cascade_1 * cascade_1, self.gamma)
        up_cascade_output1 = self.up_cascade2(up_cascade_output2 + cascade_2_mask * down_cascade_output2)


        final_output = self.up_cascade1(up_cascade_output1 + cascade_1_mask * down_cascade_output1)

        return final_output, cascade_1, cascade_2


if __name__ == '__main__':
    net = Cascade_AttenUnet(12, 3)
    print("parameters:", sum(param.numel() for param in net.parameters()))
    # input = torch.randn(3, 1, 512, 512)
    # output, cascade1, cascade2 = net(input)
    # print(output.size())








