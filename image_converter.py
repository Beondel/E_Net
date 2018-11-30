import numpy as np
import e_kernel
import pandas as pd
import math
import scipy.misc as sc
from PIL import Image
import torch

np.set_printoptions(threshold=np.nan)
imagepath = "./images/street1880.jpg"

kernel = e_kernel.Kernel(2)
scalar = 8
#image = np.array(pd.read_csv("./watch.txt", delimiter="\t"))
#orig = Image.open("peptide5.jpg")
# orig.show()
image = np.asarray(Image.open(imagepath).convert('L'))
newRange = 1
oldRange = np.amax(image) - np.amin(image)
image = scalar * (newRange / oldRange)*(image - np.amin(image))

entropyArray = np.zeros((len(image) - 1, len(image[0]) - 1))
img_len = len(image) - 1
for i in range(img_len):
    if i % 100 == 0:
        print("computing ", i, " of", img_len)
    for j in range(len(image[0]) - 1):
        subArray = np.array([[image[i][j], image[i][j + 1]],
                             [image[i + 1][j], image[i + 1][j + 1]]])
        ent = kernel.entropic_val(subArray)
        entropyArray[i][j] = ent

newRange = 255
oldRange = np.amax(entropyArray) - np.amin(entropyArray)
enImage = (newRange / oldRange)*(entropyArray - np.amin(entropyArray))


for i in range(len(enImage)):
    for j in range(len(enImage[0])):
        if enImage[i][j] < 120:
            enImage[i][j] = 0


im2 = Image.fromarray(enImage)
im2.show()


image2 = np.asarray(Image.open(imagepath))


finalImage = image2
finalImage.flags.writeable = True
img_len_2 = len(enImage) - 1
for i in range(img_len_2):
    if i % 100 == 0:
        print("computing ", i, " of", img_len_2)
    for j in range(len(enImage[0])-1):
        if enImage[i][j] == 0:
            finalImage[i][j] = [0, 0, 0]


finalImage = Image.fromarray(finalImage, mode='RGB')
finalImage.show()

sc.imsave('out.jpg', finalImage)
