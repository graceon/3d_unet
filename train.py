import dataset
from model_use import unet_2d_bri_3d


import torch
import metric
import unet
from torch import autograd, optim
from torch.utils.data import DataLoader


from tqdm import tqdm

from matplotlib import pyplot

import os
def save_checkpoint(epoch,model,optimizer):
	cp = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
	torch.save(cp, "./saved_model/%03d.pt"%epoch)
def save_metric(epoch,model,optimizer):
	cp = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
	torch.save(cp, "./saved_model/%03d.pt"%epoch)
def train_init(D_scale):
	#model = unet_2d_bri_3d(1,scale=D_scale).to(device)
	model = unet.NestedUNet(1,1).to(device)
	#optimizer = optim.SGD(model.parameters(),lr=0.)
	optimizer = optim.Adam(model.parameters())
	if len(os.listdir("./saved_model"))<1:
		return 0,model,optimizer
	else:
		print("./saved_model/%03d.pt"%dataset.dir_max_index('./saved_model/'))
		cp=torch.load("./saved_model/%03d.pt"%dataset.dir_max_index('./saved_model/'))
		model.load_state_dict(cp['model_state_dict'])
		optimizer.load_state_dict(cp['optimizer_state_dict'])
		return cp['epoch'],model,optimizer
def train(batch_size,D_scale):
	iou=[]

	criterion = torch.nn.BCELoss()
	

	trainset=dataset.liver_3d(set_dir='./output/train/',scale=D_scale,shuffle=False)
	trainset=DataLoader(trainset,batch_size=batch_size)


	epoch_end=40

	epoch_now,model,optimizer=train_init(D_scale)
	epoch_now+=1
	#print(epoch_trained)
	last_mean_loss=0
	while epoch_end >= epoch_now:
		total_iou=0
		total_loss=0
		count_item=0
		pbar = tqdm(range(len(trainset)))

		threshold_last=last_mean_loss*0.9
		for x,target in trainset:




			count_item+=1
			y=0
			loss=0
			optimizer.zero_grad()
			torch.cuda.empty_cache()

			x=x[:][:][0]
			x=x.to(device)


			target[-1]=target[-1].to(device)


			y=model(x)


			# print(x.shape)
			# print(y[-1].shape)

			#target_1=torch.unsqueeze(target[-1][0][D_scale-(D_scale//2)],0)
			#print(target_1.shape)


			predict=y[-1]
			mask=target[-1]
			weakness=metric.weakness(0.7,0.3)
			predict_weakness=weakness(predict,mask)



			loss = criterion(predict_weakness,mask)
			iou = metric.iou(predict,mask,0.5)


			
			total_loss+=loss.item()
			total_iou+=iou.item()
			#print(loss.item())
			mean_loss=total_loss/count_item
			mean_iou=total_iou/count_item
			if True:
			#if  epoch_trained <=2 or loss.item()>threshold_last:
				loss.backward()
				optimizer.step()


			#pbar.set_description("loss:%0.5f"%loss)
			pbar.set_description("[%03d]"%epoch_now +"mean_loss:%0.5f"% mean_loss+",mean_iou:%0.5f"% mean_iou)
			pbar.set_postfix(thre='%0.5f'%threshold_last,loss="%0.5f"%loss.item(),iou="%0.5f"%iou)
			pbar.update(batch_size)
		last_mean_loss=mean_loss
		
		pbar.close()

		if epoch_now==1 or (epoch_now)%10==0 or os.path.exists('./savenow'):
			save_checkpoint(epoch_now,model,optimizer)
		epoch_now+=1




if __name__ == '__main__':
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(device)
	train(
		batch_size=1,
		D_scale=1
	)
