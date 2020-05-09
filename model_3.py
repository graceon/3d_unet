import torch
from torch import cat
from torch import nn

class dual_conv_2d(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(dual_conv_2d, self).__init__()
        self.op = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
            nn.InstanceNorm3d(out_channels,affine=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
            nn.InstanceNorm3d(out_channels,affine=False),
            nn.ReLU(inplace=True)
        )
    def forward(self, input):
        return self.op(input)

class bri_conv3d(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(bri_conv3d, self).__init__()
        self.op = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(5,3,3), stride=1, padding=(0,1,1), bias=False),
            nn.InstanceNorm3d(out_channels,affine=False),
            nn.ReLU(inplace=True)
        )
    def forward(self, input):
        return self.op(input)

class unet_2d_bri_3d(nn.Module):
	def __init__(self, in_channels,scale):
		super(unet_2d_bri_3d, self).__init__()
		
		base=32

		bri_channels=40
		
		self.scale=scale
		
		self.upsample_2d=nn.Upsample(scale_factor=(1,2,2), mode='nearest')

		self.pool_2d=nn.MaxPool3d((1,2,2),(1,2,2))
		


		self.x_0_0_conv2d=dual_conv_2d(1,base)
		self.x_0_bri=bri_conv3d(base,bri_channels)

		self.x_1_0_conv2d=dual_conv_2d(base*1,base*2)
		self.x_1_bri=bri_conv3d(base*2,bri_channels*2)

		self.x_2_0_conv2d=dual_conv_2d(base*2,base*4)
		self.x_2_bri=bri_conv3d(base*4,bri_channels*4)

		self.x_3_0_conv2d=dual_conv_2d(base*4,base*8)
		self.x_3_bri=bri_conv3d(base*8,bri_channels*8)

		#self.x_4_0_conv2d=dual_conv_2d(base*8,base*16)
		#self.x_4_bri=bri_conv3d(base*16,bri_channels*16)

		#self.x_4_l_conv2d=dual_conv_2d(bri_channels*16,bri_channels*16)

		self.x_3_l_conv2d=dual_conv_2d(bri_channels*(8),bri_channels*8)

		self.x_2_l_conv2d=dual_conv_2d(bri_channels*(8+4),bri_channels*4)

		self.x_1_l_conv2d=dual_conv_2d(bri_channels*(4+2),bri_channels*2)

		self.x_0_l_conv2d=dual_conv_2d(bri_channels*(2+1),bri_channels*1)

		self.output=nn.Sequential(
            nn.Conv3d(bri_channels*1, 1, kernel_size=(1,1,1), stride=1, padding=(0,0,0), bias=False),
            nn.Sigmoid()
        )



	def forward(self, x):
		x_0_0=self.x_0_0_conv2d(x)
		x_0_bri=self.x_0_bri(x_0_0)

		x_1_0=self.x_1_0_conv2d(self.pool_2d(x_0_0))
		x_1_bri=self.x_1_bri(x_1_0)

		x_2_0=self.x_2_0_conv2d(self.pool_2d(x_1_0))
		x_2_bri=self.x_2_bri(x_2_0)

		x_3_0=self.x_3_0_conv2d(self.pool_2d(x_2_0))
		x_3_bri=self.x_3_bri(x_3_0)

		#x_4_0=self.x_4_0_conv2d(self.pool_2d(x_3_0))
		#x_4_bri=self.x_4_bri(x_4_0)

		#x_4_l=self.x_4_l_conv2d(x_4_bri)

		#x_3_l=self.x_3_l_conv2d(cat([x_3_bri,self.upsample_2d(x_4_l)],dim=1))

		x_3_l=self.x_3_l_conv2d(x_3_bri)

		x_2_l=self.x_2_l_conv2d(cat([x_2_bri,self.upsample_2d(x_3_l)],dim=1))

		x_1_l=self.x_1_l_conv2d(cat([x_1_bri,self.upsample_2d(x_2_l)],dim=1))

		x_0_l=self.x_0_l_conv2d(cat([x_0_bri,self.upsample_2d(x_1_l)],dim=1))



		y=self.output(x_0_l)
		return [y]