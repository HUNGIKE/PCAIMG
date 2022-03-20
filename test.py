import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import time
import os
from sklearn.decomposition import PCA

inputDir = "img"
output_dir = "output"
epochNum = "testDir" #int(time.time())
output_proj_dir = output_dir+"/"+epochNum

os.path.exists(output_proj_dir) or os.makedirs(output_proj_dir)


def getImgArry(inputDir):
    arry = np.empty((0,))
    for f in os.listdir(inputDir):
        im = Image.open(inputDir+"/"+f).resize( (28,28) )
        # print(f)
        # print(np.shape(np.asarray(im)))

        arry = np.append(arry,np.reshape(np.asarray(im),(28*28*3)))
        im.save(output_proj_dir+"/"+f);
        im.close()
    return {"arry":arry,"imgNum":len(os.listdir(inputDir))}

arry1 = getImgArry(inputDir+"/"+"img1")
arry2 = getImgArry(inputDir+"/"+"img2")
arry3 = getImgArry(inputDir+"/"+"img3")

numInputImg = len(os.listdir(inputDir))
pca = PCA(2)


projected = pca.fit_transform(np.reshape(arry1["arry"],(arry1["imgNum"],28*28*3)))
plt.scatter(projected[:,0],projected[:,1])

projected = pca.fit_transform(np.reshape(arry2["arry"],(arry2["imgNum"],28*28*3)))
plt.scatter(projected[:,0],projected[:,1])

projected = pca.fit_transform(np.reshape(arry3["arry"],(arry3["imgNum"],28*28*3)))
plt.scatter(projected[:,0],projected[:,1])

plt.show()


# plt.imshow(im);
# plt.show()
