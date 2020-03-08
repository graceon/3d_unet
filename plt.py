from matplotlib import pyplot
from PIL import Image
pyplot.imshow(Image.open('/fdisk/liver/train/000.png'),cmap='Greys_r')
pyplot.show()