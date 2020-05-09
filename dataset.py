import torch
import os
import re
import random


from torch.utils import data
from torch import unsqueeze

import matplotlib.pyplot as plt

def dir_max_index(set_dir):
	x=os.listdir(set_dir)[-1]

	#print(os.listdir(set_dir))
	res=0
	while re.match('\d',x[0]):
		res=res*10+int(x[0])
		x=x[1:]
	return res


global first
first=0
def uint8_transfrom(x,y,scale):
	# global first
	# if first==0:
	# 	plt.imshow((y.float().cpu()/255)[scale//2])

	# 	plt.show()
	# 	first=1
	return unsqueeze((x.float()/127-1),0),[unsqueeze(unsqueeze((y.float()/255)[scale//2],0),0)]

class liver_3d(data.Dataset):
    def __init__(self,set_dir,scale,stride=1,shuffle=False):
        self.set_dir = set_dir
        self.scale=scale
        self.stride=stride

        self.max=dir_max_index(set_dir)

        self.group=[]
        self.group_D=[]
        

        self.group_mask=[]
        

        self.group_mask_bold=self.group_mask



        for i in range(self.max+1):
            self.group.append(torch.load(self.set_dir+'%d.pt'%i).cuda())

            self.group_D.append(self.group[-1].shape[0])

            self.group_mask.append(torch.load(self.set_dir+'%d_mask.pt'%i).cuda())
            #self.group_mask_bold.append()


        #prepare for __getitem__()
        self.getitem_index_max_map=[]
        for D in self.group_D:
            self.getitem_index_max_map.append((D-self.scale)//(self.stride)+1)
        for i in range(1,len(self.getitem_index_max_map)):
            self.getitem_index_max_map[i]+=self.getitem_index_max_map[i-1]
        self.len=self.getitem_index_max_map[-1]


        self.index_shuffle=[i for i in range(self.len)]
        if shuffle:
        	random.shuffle(self.index_shuffle)


    def __getitem__(self, index):
        if index>=self.len:
            raise StopIteration

        index=self.index_shuffle[index]
        last_max=0
        for i in self.getitem_index_max_map:
            if i>index:
                #print(index)
                num=self.getitem_index_max_map.index(i)
                start=self.stride*(index-last_max)
                end=start+self.scale
                return uint8_transfrom(self.group[num][start:end],self.group_mask[num][start:end],self.scale)
            last_max=i
    def __len__(self):
   		return self.len