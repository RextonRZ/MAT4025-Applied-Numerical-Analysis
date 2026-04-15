######  Ch6. 형태학적 영상처리
# Scikit-image morphology 모듈
# 이진연산
# 함수를 호출하기 전에 이진 입력 영상을(고정된 단순 임계치) 생성해야 한다.
# 침식 Erosion
# 침식은 전경 객체의 크기를 줄이고 객체 경계를 부드럽게 하며, 반도 peninsulas, 
# 가락 및 작은 객체를 제거하는 기본 형태학적 연산
# binary_erosion()
########################################################
## Example 1. 침식
########################################################
from skimage import color
from skimage.io import imread
from skimage.morphology import binary_erosion, rectangle

im = color.rgb2gray(imread('clock.jpg'))
im[im<=0.5]=0
im[im>0.5]=1
im1 = binary_erosion(im, rectangle(1,5))
im2 = binary_erosion(im, rectangle(1,15))

import matplotlib.pyplot as plb
plb.figure(figsize=(20,10)), plb.gray()
plb.subplot(131), plb.imshow(im), plb.axis('off'), plb.title('Ori',size=20)
plb.subplot(132), plb.imshow(im1), plb.axis('off'), plb.title('Erosion with size (1,5)',size=20)
plb.subplot(133), plb.imshow(im2), plb.axis('off'), plb.title('Erosion with size (1,15)',size=20)

#######################################################
# Example 2. 팽창 Dilation
#########################################################
# 전경 객체의 크기를 확장하고 객체 경계를 부드럽게 하며
# 이진 영상의 구멍과 틈을 닫는 또 다른 기본 형태학적 연산
im = color.rgb2gray(imread('Tagore.jpg'))
im[im<=0.5] = 0
im[im>0.5] = 1

from skimage.morphology import binary_dilation, disk
plb.figure(figsize=(18,9)), plb.gray()
plb.subplot(131), plb.imshow(im)
plb.axis('off'), plb.title('Tagore',size=20)
for d in range(1,3):
    im1 = binary_dilation(im,disk(d))
    plb.subplot(1,3,d+1), plb.imshow(im1) 
    plb.axis('off')
    plb.title('Dilation with disk size '+str(2*d),size=20)
plb.show()

##################################################
# Example 3. 열림과 닫힘 Opening and Closing
##################################################
# 이중연산
# 열림 : 침식과 팽창 연산의 조합, 이진영상에서 작은 객체 제거
# 닫힘 : 팽창과 침식 연산의 조합, 이진영상에서 작은 구멍 제거
from skimage.morphology import binary_opening, binary_closing, disk

im = color.rgb2gray(imread('circles.jpg'))
im[im<=0.5]=0
im[im>0.5]=1
im1 = binary_erosion(im,disk(3))
im2 = binary_dilation(im,disk(3))
im3 = binary_opening(im,disk(3))
im4 = binary_closing(im,disk(3))

plb.figure(figsize=(20,10)), plb.gray()
plb.subplot(231), plb.imshow(im), plb.axis('off'), plb.title('Circles',size=20)
plb.subplot(232), plb.imshow(im1), plb.axis('off'), plb.title('Erosion with disk'+str(3),size=20)
plb.subplot(233), plb.imshow(im2), plb.axis('off'), plb.title('Dilation with disk'+str(3),size=20)
plb.subplot(235), plb.imshow(im3), plb.axis('off'), plb.title('Opening with disk'+str(3),size=20)
plb.subplot(236), plb.imshow(im4), plb.axis('off'), plb.title('Closing with disk'+str(3),size=20)
plb.show()

#########################################################
# Example 4. 골격화 skeletonizing
########################################################
# 이진 영상의 연결된 각 구성요소는 형태학적 세선화 연산을 사용하여
# 단일화서 너비 골격으로 축소된다.
def plot_images_horizontally(original, filtered, filter_name, sz=(18,7)):
    plb.gray()
    plb.figure(figsize=sz)
    plb.subplot(121), plb.imshow(original), plb.axis('off'), plb.title('Original',size=20)
    plb.subplot(122), plb.imshow(filtered), plb.axis('off'), plb.title(filter_name,size=20)
    plb.show()

from skimage import img_as_float    
from skimage.morphology import skeletonize
im = img_as_float(color.rgb2gray(imread('Dinosaur.jpg')))    
threshold = 0.7
im[im<= threshold] = 0
im[im>threshold] = 1
skeleton = skeletonize(im)
plot_images_horizontally(im,skeleton,'skeleton',sz=(18,9))

#########################################################
# Example 5. 볼록 선체 convex hull 계산하기
########################################################
from skimage.morphology import convex_hull_image
im = color.rgb2gray(imread('Dinosaur.jpg'))
threshold = 0.5
im[im<threshold]=0
im[im>=threshold]=1
im = 1 - im
chull = convex_hull_image(im)
plot_images_horizontally(im,chull,'convex hull',sz=(18,9))

im = im.astype(np.bool)
chull_diff = img_as_float(chull.copy())
chull_diff[im] = 2

plb.figure(figsize=(20,10))
plb.imshow(chull_diff, cmap=plb.cm.gray, interpolation='nearest')
plb.title('Difference Image',size=20)
plb.show()

##########################################################
# Example 6. 작은 객체 제거
##########################################################
# remove_small_object(): 지정된 최소 임계치보다 작은 객체를 제거하는 방법
from skimage.morphology import remove_small_objects
im = color.rgb2gray(imread('circles.jpg'))
threshold = 0.7
im[im<=threshold],im[im>threshold] = 0,1
im = im.astype(np.bool)
plb.figure(figsize=(20,15))
plb.subplot(221), plb.imshow(im), plb.axis('off'), plb.title('Original',size=20)
for i,osz in enumerate([50,200,500]):
    im1 = remove_small_objects(im,osz,connectivity=1)
    plb.subplot(2,2,i+2)
    plb.imshow(im1)
    plb.axis('off')
    plb.title('Removing small objects below size '+str(osz),size=20)
plb.show() 

########################################
# Example 7. 흰색과 검은색 탑-햇 top-hats
###########################################
# 흰색 탑-햇: 구조요소보다 작은 밝은 점들을 계산, 원본과 형태학적 열림의 차
# 검은색 탑-햇: 구조요소보다 작은 어두운 점들을 계산, 원본과 닫힘의 차
#              검은색 탑-햇 연산 후에 원 영상의 어두운 점들이 밝은 점들이 된다.
from skimage.morphology import white_tophat, black_tophat, square
im = color.rgb2gray(imread('Tagore.jpg'))
threshold = 0.3
im[im<=threshold], im[im>threshold] = 0,1
im1 = white_tophat(im,square(5))
im2 = black_tophat(im,square(5))

plb.figure(figsize=(20,15)), plb.gray()
plb.subplot(131), plb.imshow(im), plb.axis('off'), plb.title('Tagore',size=20)
plb.subplot(132), plb.imshow(im1), plb.axis('off'), plb.title('white tophat',size=20)
plb.subplot(133), plb.imshow(1-im2), plb.axis('off'), plb.title('black tophat',size=20)
plb.show()

########################################
# Example 8. 경계추출
###########################################
# 침식연산은 이진연산의 경계를 추출하는데 사용할 수 있다. 
# 침식된 영상을 이진 영상에서 빼서 경계추출   
from skimage.morphology import binary_erosion
im = color.rgb2gray(imread('Tagore.jpg'))
threshold = 0.5
im[im<threshold], im[im>= threshold] = 0,1
boundary = im - binary_erosion(im)
plot_images_horizontally(im,boundary,'boundary',sz=(18,9))

########################################
# Example 9. 열림과 닫힘을 이용한 지문개선
###########################################
# 열림과 닫힘을 순차적으로 사용하여 이진 영상에서 잡음, 작은 전경 객체를 제거할 수 있다.
# 전처리 단계로서 지문 영상을 개선하는데 사용할 수 있다.
from skimage.morphology import binary_closing, binary_opening
im = color.rgb2gray(imread('fingerprint.jpg'))
im = im + 0.3*np.random.rand(im.shape[0],im.shape[1])
im[im<=0.5], im[im>0.5] = 0,1
#np.random.rand(im.shape)
imo = binary_opening(im,square(2))
imc = binary_closing(im,square(2))
imoc = binary_closing(imo,square(2))

plb.figure(figsize=(20,20))
plb.subplot(221), plb.imshow(im), plb.axis('off'), plb.title('Original',size=30)
plb.subplot(222), plb.imshow(imo), plb.axis('off'), plb.title('Opening',size=30)
plb.subplot(223), plb.imshow(imc), plb.axis('off'), plb.title('closing',size=30)
plb.subplot(224), plb.imshow(imoc), plb.axis('off'), plb.title('Opening+Closing',size=30)
plb.show()

##################################
# Example 10. 명암도 연산
########################################
from skimage.morphology import dilation, erosion, closing, opening, square
im = color.rgb2gray(imread('zebra.jpg'))
struct_elem = square(2)
eroded = erosion(im,struct_elem)
dilated = dilation(im,struct_elem)
opened = opening(im,struct_elem)
closed = closing(im,struct_elem)

plb.figure(figsize=(15,10))
plb.subplot(231), plb.imshow(im), plb.axis('off'), plb.title('Zebra',size=30)
plb.subplot(232), plb.imshow(eroded), plb.axis('off'), plb.title('Erosion',size=30)
plb.subplot(233), plb.imshow(dilated), plb.axis('off'), plb.title('Dilation',size=30)
plb.subplot(235), plb.imshow(opened), plb.axis('off'), plb.title('Opening',size=30)
plb.subplot(236), plb.imshow(closed), plb.axis('off'), plb.title('Closing',size=30)
plb.show()

##################################################
# Example 11. Scikit-image filter.rank 모듈
###################################################
# 형태학적 콘트라스트 향상
# 형태학적 콘트라스트 향상 필터는 구조요소에 의해 정의된 이웃의 화소만을 고려하여 각 화소에 작동
# 원 화소가 어떤 화소에 가장 가까이 있는가에 따라 중심 화소를 로컬 최소 또는 로컬 최대 화소로 바꾼다.
# 노출 모듈의 적응형 히스토그램 평활화를 사용하여 얻은 출력을 비교
def plot_gray_image(ax,image,title):
    ax.imshow(image,vmin=0,vmax=255,cmap='gray')
    ax.set_title(title), ax.axis('off')
    ax.set_adjustable('box') #'-forced')
    
from skimage.filters.rank import enhance_contrast
from skimage import exposure

image = color.rgb2gray(imread('squirrel.jpg'))    
sigma = 0.05
noise = sigma*np.random.standard_normal(image.shape)
noisy_image = (np.clip(image+noise,0,1)*255).astype(np.uint8)
enhanced_image = enhance_contrast(noisy_image,disk(5))
equalized_image = exposure.equalize_adapthist(noisy_image)
equalized_image = (equalized_image*255).astype(np.uint8)

fig, axes = plb.subplots(1,3,figsize=[18,7],sharex='row',sharey='row')
axes1, axes2, axes3 = axes.ravel()
plot_gray_image(axes1, noisy_image, 'Squirrel, 5% Noise')
plot_gray_image(axes2, enhanced_image, 'Local morphological contrast enhancement')
plot_gray_image(axes3,equalized_image, 'Adaptive Histogram equalization')

#####################################################
# Example 12. 메디안 필터를 사용한 잡음 제거
#####################################################
# scikit-images filters.rank 모듈의 형태학적 메디안 필터를 사용하는 방법
from skimage.filters.rank import median

a = imread('Lenna.png')[:,:,:3]
#plb.imshow(a)
noisy_image = color.rgb2gray(a)*255 #).astype(np.uint8)
#plb.imshow(noisy_image)
noise = np.random.random(noisy_image.shape)
noisy_image[noise>0.99] = 255
noisy_image[noise<0.01] = 0

fig, axes = plb.subplots(2,2,figsize=(10,10),sharex=True,sharey=True)
axes1, axes2, axes3, axes4 = axes.ravel()
plot_gray_image(axes1, noisy_image, 'Noisy Image, 1% salt, 1% pepper')
plot_gray_image(axes2, median(noisy_image/255,disk(1)), 'Median $r=1$')
plot_gray_image(axes3, median(noisy_image/255,disk(5)), 'Median $r=5$')
plot_gray_image(axes4, median(noisy_image/255,disk(20)), 'Median $r=20$')

####################################################
# Example 13.로컬 엔트로피 계산
####################################################
# 엔트로피는 영상의 불확실성 또는 임의성의 척도
#  H = -\sum_{i=0}^255 p_i log2 p_i
# p_i 는 명암도 i와 연관된 확률(영상의 정규화된 히스토그램으로부터 얻어 짐)
# 유사한 방식으로 로컬 엔트로피를 정의하여 영상 복잡성을 정의할 수 있다. 로컬 히스토그램 사용하여 계산
# skimage.rank.entropy() : 주어진 구조요소에서 영상의 로컬 엔트로피(로컬 명암도 분포를 인코딩하는 데 필요한 최소 비트 수)
from skimage.filters.rank import entropy
image = color.rgb2gray(imread('birds.jpg'))

fig, (axes1,axes2) = plb.subplots(1,2,figsize=(15,5))
fig.colorbar(axes1.imshow(image,cmap='gray'),ax=axes1)
axes1.axis('off'), axes1.set_title('Image',size=20)
#axes1.set_adjustable('box')
fig.colorbar(axes2.imshow(entropy(image,disk(5)),cmap='inferno'),ax=axes2)
axes2.axis('off'), axes2.set_title('Entropy',size=20)
#axes2.set_adjustable('box')
plb.show()

#########################################
## Example 14. SciPy ndimage.morphology 모듈
#################################################
# 이진 객체에 구멍 채우기
from scipy.ndimage.morphology import binary_fill_holes

im = rgb2gray(imread('Hands002.png')[:,:,:3])
im = 1 - im
im[im <= 0.5] = 0 
im[im > 0.5] = 1

plb.figure(figsize=(20,12)), plb.gray()
plb.subplot(231), plb.imshow(im), plb.axis('off'), plb.title('Original',size=20)
for i,n in enumerate([2,3,4,5,6]):
    im1 = binary_fill_holes(im,structure=np.ones((n,n)))
    #title = 'binary_fill_holse with structure square side ' + str(n)
    title = 'Square side '+str(n) 
    plb.subplot(2,3,i+2), plb.imshow(im1)
    plb.axis('off'), plb.title(title,size=20)
plb.show()    
# 구조요소(사각형)가 클수록 채워지는 구멍의 수가 적다.

################################################
# Example 15. 열림과 닫힘을 사용한 (소금후추)잡음 제거
#####################################################
# 소금 후추 잡음 추가
from PIL import Image, ImageFont, ImageDraw
im = Image.open("Mandrill2.jpg") # RGB 저장
iml = im.copy()
n = int(im.width*im.height*0.1)
x = np.random.randint(0, im.width, n)
y = np.random.randint(0,im.height,n)
for (x,y) in zip(x,y):
    new_pix = (0,0,0) if np.random.rand() < 0.5 else (255,255,255)
    iml.putpixel((x,y),new_pix)
iml.show()
# iml.save('Mandrill2_01.jpg')
# 소금후추잡음제거
from scipy import ndimage
im = rgb2gray(imread('Mandrill2_01.jpg'))
imo = ndimage.grey_opening(im,size=(2,2))
imc = ndimage.grey_closing(im,size=(2,2))
imoc = ndimage.grey_closing(imo,size=(2,2))

plb.figure(figsize=(20,20)), plb.gray()
plb.subplot(221), plb.imshow(im), plb.axis('off'), plb.title('Mandrill,10% Noise',size=20)
plb.subplot(222), plb.imshow(imo), plb.axis('off'), plb.title('Opening',size=20)
plb.subplot(223), plb.imshow(imc), plb.axis('off'), plb.title('Closing',size=20)
plb.subplot(224), plb.imshow(imoc), plb.axis('off'), plb.title('Opening + Closing',size=20) 
plb.show()   

################################################
# Example 16. 형태학적 베커(Beucher)의 그레디언트 계산
######################################################
# 입력된 명암도 영상의 팽창버전과 침식버전의 차 영상
im = rgb2gray(imread('Einstein.jpg'))
imd = ndimage.grey_dilation(im,size=(3,3))
ime = ndimage.grey_erosion(im,size=(3,3))
imbg = imd - ime
img = ndimage.morphological_gradient(im,size=(3,3))

plb.figure(figsize=(20,18)), plb.gray()
plb.subplot(231), plb.imshow(im), plb.axis('off'), plb.title('Einstein',size=20)
plb.subplot(232), plb.imshow(imd), plb.axis('off'), plb.title('dilation',size=20)
plb.subplot(233), plb.imshow(ime), plb.axis('off'), plb.title('erosion',size=20)
plb.subplot(234), plb.imshow(imbg), plb.axis('off'), plb.title('Beucher gradient',size=20)
plb.subplot(235), plb.imshow(img), plb.axis('off'), plb.title('ndimage gradient',size=20)
plb.subplot(236), plb.imshow(imbg-img), plb.axis('off'), plb.title('bg - g',size=20) # 동일
plb.show()

###########################################
# Example 17.형태학적 라플라스 계산
############################################
# 다양한 크기의 구조요소를 가진 형태학적 그래디언트와 비교
# 그래디언트가 있는 더 작은 구조요소와 라플라스가 있는 더 큰 구조요소 : 추출된 에지측면에서 더 낫다.
im = rgb2gray(imread('Tagore.jpg'))
img3 = ndimage.morphological_gradient(im,size=(3,3))
img5 = ndimage.morphological_gradient(im,size=(5,5))
iml3 = ndimage.morphological_laplace(im,size=(3,3))
iml5 = ndimage.morphological_laplace(im,size=(5,5))

plb.figure(figsize=(10,10))
plb.subplot(231), plb.imshow(im), plb.axis('off'), plb.title('Tagore',size=20)
plb.subplot(232), plb.imshow(img3), plb.axis('off'), plb.title('gradient 3',size=20)
plb.subplot(233), plb.imshow(img5), plb.axis('off'), plb.title('gradient 5',size=20)
plb.subplot(235), plb.imshow(iml3), plb.axis('off'), plb.title('laplace 3',size=20)
plb.subplot(236), plb.imshow(iml5), plb.axis('off'), plb.title('laplace 5',size=20)
plb.show()





















































