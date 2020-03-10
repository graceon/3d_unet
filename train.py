import dataset
from model_2 import unet_2d_bri_3d


import torch
from torch import autograd, optim
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


if __name__ == '__main__':

	batch_size=1
	D_scale=5

	model = unet_2d_bri_3d(1,scale=D_scale).to(device)
	criterion = torch.nn.BCELoss()
	optimizer = optim.Adam(model.parameters())


	trainset=dataset.liver_3d(set_dir='./output/train/',scale=D_scale)
	trainset=DataLoader(trainset,batch_size=batch_size)

	
	for x,target in trainset:
		x=x.to(device)
		target[-1]=target[-1].to(device)
		y=model(x)
		print(x.shape)
		print(y[-1].shape)
		print(target[-1].shape)
		loss =0
		loss += criterion(y[-1], target[-1])
		break

