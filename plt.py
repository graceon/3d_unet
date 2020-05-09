import torch
import numpy as np
from matplotlib import pyplot

from PIL import Image
plt =pyplot
# pyplot.imshow()
# pyplot.show()

# [162, 209, 29]

# [158, 205, 25]

# [158, 363, 388]

# x=torch.load('/fdisk/3d_unet/output/train/2.pt')
# max=x.shape[0]-1
# print(x.shape)



# fig=pyplot.figure()
# fig.add_subplot(1, 2, 1)
# pyplot.imshow(x[0],cmap='Greys_r')

# fig.add_subplot(1, 2, 2)
# pyplot.imshow(x[-1],cmap='Greys_r')
# pyplot.show()


# pyplot


('''
N = 1000
x=[]
y=[]
for i in range(10):
	x.append(i)
	y.append(1)
	x.append(i)
	y.append(2)


pyplot.scatter(x, y,s=0.1)#edgecolors='none')
pyplot.show()
''')

('''
a=torch.zeros(2,1,4,4,4)
for i in range(64):
	if(i<32):
		sign=-1
	else :sign=1
	a[0][0][i//16%4][i//4%4][i%4]=2+(2**0.5)*3*sign
	a[1][0][i//16%4][i//4%4][i%4]=2+(2**0.5)*4*sign
print(a)
m = torch.nn.InstanceNorm3d(1, affine=False)
#m = torch.nn.BatchNorm3d(1, affine=False)
a=m(a)

x=[]
y=[]
for i in range(64):
	x.append(i)
	y.append(a[0][0][i//16%4][i//4%4][i%4].numpy())
	x.append(i)
	y.append(a[1][0][i//16%4][i//4%4][i%4].numpy())
pyplot.scatter(x, y,s=1)#edgecolors='none')

pyplot.show()
print(a)

''')



import random
import matplotlib  
import matplotlib.pyplot as plt  
 
 
def list2mat(data_list,w):
    '''
    切片、转置
    '''
    mat=[]
    res=[]
    for i in range(0,len(data_list)-w+1,w):
        mat.append(data_list[i:i+w])
    for i in range(len(mat[0])):
        one_list=[]
        for j in range(len(mat)):
            one_list.append(mat[j][i])
        res.append(one_list)
    return res
 
 
 
def draw_pic_test():
    '''
    作图
    '''
    data_list=[]
    for i in range(100):
        data_list.append(random.randint(80,150))


    month_list=range(1,11,1)
    mat=list2mat(data_list,w=10)

    print(torch.Tensor(data_list).shape)
    print(torch.Tensor(month_list).shape)
    print(torch.Tensor(mat).shape)

    cnt=0
    for one_list in mat:
        one_list=[int(one) for one in one_list]
        print(torch.Tensor(month_list[0:5]).shape)
        print(torch.Tensor(one_list).shape)
        plt.scatter(month_list,one_list,marker='.',label='test_sandian',s=30) 
        if cnt < 1 :
            cnt+=1
        else :
            break
    plt.show()
    plt.close()
 
 
 
if __name__ == '__main__':
    draw_pic_test()
