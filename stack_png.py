import torch
from PIL import Image
#from matplotlib import pyplot
from torchvision.transforms import transforms
dataset_train_group_end_collection=[161,370,399]
dataset_train_path='/fdisk/liver/train/'

dataset_val_group_end_collection=[19]
dataset_val_path='/fdisk/liver/val/'

output_train='./output/train/'
output_val='./output/val/'


ToTensor=transforms.ToTensor()
def stack_png(path,start,end):
	x=list()
	y=list()
	for i in range(start,end+1):
		x.append(ToTensor(Image.open(path+('%03d'%i)+'.png'))[0]*255)
		y.append(ToTensor(Image.open(path+('%03d'%i)+'_mask.png'))[0]*255)
	x=torch.stack(x,dim=0)
	x=torch.tensor(x,dtype=torch.uint8)
	y=torch.stack(y,dim=0)
	y=torch.tensor(y,dtype=torch.uint8)
	return x,y
if __name__ =="__main__":


	last_end=0
	group_index=0
	for i in dataset_train_group_end_collection:
		x,y=stack_png(dataset_train_path,last_end,i)
		torch.save(x,output_train+str(group_index)+'.pt')
		torch.save(y,output_train+str(group_index)+'_mask.pt')
		print(x.shape,y.shape)
		last_end=i+1
		group_index+=1
	last_end=0
	group_index=0
	for i in dataset_val_group_end_collection:
		x,y=stack_png(dataset_val_path,last_end,i)
		torch.save(x,output_val+str(group_index)+'.pt')
		torch.save(y,output_val+str(group_index)+'_mask.pt')
		print(x.shape,y.shape)
		last_end=i+1
		group_index+=1