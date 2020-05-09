import dataset
from model_use import unet_2d_bri_3d


import torch
from torch import autograd, optim
from torch.utils.data import DataLoader

import numpy as np
import unet
import os

import cv2
def val_init(D_scale):
	model = unet.NestedUNet(1,1).to(device)

	print("load:./saved_model/%03d.pt"%(dataset.dir_max_index('./saved_model/')))
	cp=torch.load("./saved_model/%03d.pt"%(dataset.dir_max_index('./saved_model/')))
	#p=torch.load("./saved_model/%03d.pt"%1)
	model.load_state_dict(cp['model_state_dict'])
	return cp['epoch'],model
def val(batch_size,D_scale):

	criterion = torch.nn.BCELoss()

	valset=dataset.liver_3d(set_dir='./output/val/',scale=D_scale)
	valset=DataLoader(valset,batch_size=batch_size)


	epoch_trained,model=val_init(D_scale)
	print(epoch_trained)

	i=0
	for x,target in valset:

		y=0
		loss=0
		torch.cuda.empty_cache()

		x=x[:][:][0]
		x=x.to(device)

		target[-1]=target[-1].to(device)
		y=model(x)


		predict=y[-1]
		# loss = criterion(predict,target[-1])
		# print(loss)

		predict=(predict>0.5).float()*255

		
		#print(y[-1].shape)
		#print(y[-1][0][0][0][1][1])
		img_gen=np.array(predict.view(512,512).cpu())

		print(img_gen.shape)
		img_path='./saved_predict/'+str(i).zfill(3)+'_mask_predict.png'
		print(img_path)
		cv2.imwrite(img_path,img_gen) 
		i+=1



if __name__ == '__main__':
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(device)
	with torch.no_grad():
		val(
			batch_size=1,
			D_scale=1
		)
