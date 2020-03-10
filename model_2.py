import torch.nn as nn
import torch

class dual_conv_2d(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(DoubleConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False)
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False)
            nn.ReLU(inplace=True)
        )
    def forward(self, input):
        return self.op(input)
class unet_2d_bri_3d(nn.Module):
	def __init__(self, in_channels,scale):
		super(unet_2d_bri_3d, self).__init__()
		self.scale=scale
		self.pool_2d=nn.MaxPool3d((1,2,2),(1,2,2))


		self.depth_1_c1=nn.Conv3d(1, 128, kernel_size=(5,3,3), stride=1, padding=(0,1,1), bias=False)
		self.depth_1_c2=nn.Conv3d(128, 128, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False)
		self.to_1=self.conv3d_c1_1 = nn.Conv3d(128, 1, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False)

	def forward(self, x):
		out=self.depth_1_c1(x)
		out=self.depth_1_c2(out)



		stack_png
		stack_png=self.to_1(out)
		for i in range(1,self.scale):
			stack_png=torch.cat([stack_png,self.to_1(out)],dim=-3)
		out=nn.Sigmoid()(stack_png)
		return [out]