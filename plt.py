import torch
from matplotlib import pyplot

from PIL import Image
# pyplot.imshow()
# pyplot.show()

[162, 209, 29]

[158, 205, 25]

[158, 363, 388]

x=torch.load('/fdisk/3d_unet/output/train/2.pt')
max=x.shape[0]-1
print(x.shape)



fig=pyplot.figure()
fig.add_subplot(1, 2, 1)
pyplot.imshow(x[0],cmap='Greys_r')

fig.add_subplot(1, 2, 2)
pyplot.imshow(x[-1],cmap='Greys_r')
pyplot.show()
