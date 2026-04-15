# Ch9 Machine Learning
# Supervised Learning vs. Unsupervised Learning
# Unsupervised Learning : Clusering, PCA, Eigenface
# K-means clustering : Color quantization 
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.io import imread

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils   import shuffle
from skimage         import img_as_float #0~1

########################################################
## Example 1. K-means : Quantization
########################################################
fish = img_as_float(imread("fish.jpg"))
h,w,d = original_shape = fish.shape #tuple(fish.shape)
assert d == 3  #if error, AssertionError
image_array = np.reshape(fish, (w*h,d))

# codebook과 label로 부터 영상 재생성 
def recreate_image(codebook, labels, w, h):
    d = codebook.shape[1]
    labels = labels.reshape(h,w)
    image = [[codebook[labels[i][j]] for j in range(w)] for i in range(h)]
    return image

# 원본 영상
plt.figure(1)#, plt.clf(), ax = plt.axes([0,0,1,1]) 
plt.axis('off')
plt.title('Original image (%d colors)' %(len(np.unique(fish))),size=20)
plt.imshow(fish)

# Quantization    
plt.figure(2, figsize=(10,7))
i = 1
for k in [64,32,16,8]:
    # 영상에서 랜덤하게 1000화소를 가져와 k-means 실행 : 양자화
    image_array_sample = shuffle(image_array, random_state=0)[:1000]
    kmeans = KMeans(n_clusters=k, random_state=0).fit(image_array_sample)
    # 전체 영상 색상 예측
    labels = kmeans.predict(image_array) 
    plt.subplot(2,2,i), plt.axis('off')
    plt.title("Quantized image ("+str(k)+' colors, K-means)')
    tmp = recreate_image(kmeans.cluster_centers_,labels, w, h)
    plt.imshow(img_as_float(tmp))
    i += 1
plt.suptitle('K-means')    
plt.show()  


plt.figure(3, figsize=(10,7))
i = 1
for k in [64,32,16,8]:
    # 영상에서 랜덤하게 k화소를 가져옴
    codebook_random = shuffle(image_array, random_state=0)[:k+1]
    # 전체 영상 색상 예측
    labels = pairwise_distances_argmin(codebook_random,image_array,axis=0)
    plt.subplot(2,2,i), plt.axis('off')
    plt.title("Quantized image ("+str(k)+' colors, Random)')
    tmp = recreate_image(codebook_random,labels, w, h)
    plt.imshow(img_as_float(tmp))
    i += 1
plt.suptitle('Random')      
plt.show()  

#################################
# Example 2. Spectrum clustering
#################################
from sklearn import cluster
#from scipy.misc import imresize : version 1.2.0 이후 지원하지 않음
from skimage.color import rgb2gray

fish = img_as_float(imread("fish.jpg"))
im0 = cv2.resize(src=fish[:,:,0], dsize=(100,100))
im1 = cv2.resize(src=fish[:,:,1], dsize=(100,100))
im2 = cv2.resize(src=fish[:,:,2], dsize=(100,100))
im = np.zeros([100,100,3]) 
im[:,:,0] = im0; im[:,:,1] = im1; im[:,:,2] = im2

k=2 # 이진 분할
X = np.reshape(im, (-1,im.shape[-1]))
two_means = cluster.MiniBatchKMeans(n_clusters=k, random_state=10)
two_means.fit(X)
y_pred1 = two_means.predict(X)
spectral = cluster.SpectralClustering(n_clusters=k, eigen_solver='arpack',\
                affinity='nearest_neighbors',n_neighbors=100,random_state=10)
spectral.fit(X)
y_pred2 = spectral.labels_.astype(int)

labels = [np.reshape(y_pred1, im.shape[:2]), np.reshape(y_pred2, im.shape[:2])]
# k-means clustering vs. spectral clustering
titles = ["k-means","spectral"]

plt.figure(figsize=(20,20))
for i,label in enumerate(labels):
    plt.subplot(2,2,i*2+1), plt.imshow(label, cmap='gray')
    plt.title(titles[i]+' segmentation (k=2)',size=30)
    plt.subplot(2,2,i*2+2), plt.imshow(im)
    plt.contour(label == 0, contours=10, colors='blue')
    plt.title(titles[i]+' contour (k=2)', size=30), plt.axis('off')
plt.tight_layout()
plt.show()    

##########################################3
# Example 3. PCA
##########################################
# Mini MNIST
from sklearn.datasets import load_digits #Mini MNIST
digits = load_digits()
print(digits.data.shape)

j=1
np.random.seed(1)
fig = plt.figure(figsize=(3,3))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in np.random.choice(digits.data.shape[0],25):
    plt.subplot(5,5,j), plt.axis('off')
    plt.imshow(np.reshape(digits.data[i,:], (8,8)), cmap='binary')
    j += 1
plt.show()  

# PCA
from sklearn.decomposition import PCA

pca_digits = PCA(2)
digits.data_proj = pca_digits.fit_transform(digits.data)
plt.figure(num='',figsize=(15,10))
plt.scatter(digits.data_proj[:,0], digits.data_proj[:,1], lw=0.25, \
            c=digits.target, edgecolor='k', s=100, cmap=plt.cm.get_cmap('cubehelix',10) )
plt.xlabel('PC1', size=20), plt.ylabel("PC2",size=20)
plt.title('2DProjection of handwritten digits with PCA', size=25)
plt.colorbar(ticks=range(10), label='digit value')
plt.clim(-0.5,9.5)
plt.show()

##########################################3
# Example 4. Mean face and SD face
########################################## 
from sklearn.datasets import fetch_olivetti_faces

faces = fetch_olivetti_faces().data
print(faces.shape) #(400, 64*64)

fig = plt.figure(figsize=(5,5))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.1, wspace=0.1)
# 25 random faces
j=1
np.random.seed(0)
for i in np.random.choice(range(faces.shape[0]),25):
    ax = fig.add_subplot(5,5,j,xticks=[],yticks=[])
    face = np.reshape(faces[i,:],(64,64))
    ax.imshow(face, cmap='bone', interpolation='nearest')
    j += 1
plt.show()    

#########################################################
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

n_comp = 64
pipeline = Pipeline([('scaling',StandardScaler()),('pca',PCA(n_comp))])
faces_proj = pipeline.fit_transform(faces)
print(faces_proj.shape) #(400,64)

#######################################################
mean_face = np.reshape(pipeline.named_steps['scaling'].mean_, (64,64))
sd_face   = np.reshape(np.sqrt(pipeline.named_steps['scaling'].var_),(64,64))

plt.figure(figsize=(8,6))
variance_ratio = pipeline.named_steps['pca'].explained_variance_ratio_
plt.plot(np.cumsum(variance_ratio), linewidth=2)
plt.grid(), plt.axis('tight'), plt.xlabel('n_components')
plt.show()

#######################################################
plt.figure(figsize=(10,5)) 
plt.subplot(121), plt.imshow(mean_face, cmap='bone')
plt.axis('off'), plt.title('Mean face')
plt.subplot(122), plt.imshow(sd_face, cmap='bone' )
plt.axis('off'), plt.title('SD face')
plt.show()

##########################################3
# Example 5. Eigenface
##########################################
fig = plt.figure(figsize=(5,2))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)

#First 10 eigenfaces
for i in range(10):
    face = np.reshape(pipeline.named_steps['pca'].components_[i,:],(64,64))
    ax = fig.add_subplot(2,5,i+1,xticks=[],yticks=[])
    ax.imshow(face, cmap='bone', interpolation='nearest')
plt.show()    

##########################################3
# Example 6. Reconstrunction
##########################################
faces_inv_proj = pipeline.named_steps['pca'].inverse_transform(faces_proj)
faces_inv_proj = np.reshape(faces_inv_proj, (400,64,64))
# 64*64 images 400 samples transforms
fig = plt.figure(figsize=(5,5))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.1, wspace=0.1)
# 64*64 dimensional face plotting 
j=1
np.random.seed(0)
for i in np.random.choice(range(faces.shape[0]),25):
    reconst_face = mean_face + sd_face*faces_inv_proj[i,:]
    ax = fig.add_subplot(5,5,j,xticks=[],yticks=[])
    ax.imshow(reconst_face, cmap='bone',interpolation='nearest') #, ax.axis('off')
    j += 1
plt.show()    

##########################################3
# Example 7. Reconstruction : Glasses
##########################################
orig_face = np.reshape(faces[0,:], (64,64))
reconst_face = faces_proj[0]@pipeline.named_steps['pca'].components_
reconst_face = mean_face + sd_face * np.reshape(reconst_face, (64,64))

plt.figure(figsize=(10,5))
plt.subplot(121), plt.axis('off'), plt.title('original',size=20)
plt.imshow(orig_face, cmap='bone', interpolation='nearest')
plt.subplot(122), plt.axis('off'), plt.title('Reconstructed',size=20)
plt.imshow(reconst_face, cmap='bone',interpolation='nearest')
plt.show()

