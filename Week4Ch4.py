# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 19:47:33 2021

@author: USER
"""

#### Ch.4. 영상 향상 ==> 10주차, 2주차,4주차
## 선형 잡음 필터링 ==> 4주차
# 박스필터, 가우시안 필터

# PIL을 사용한 필터링
# PIl ImageFilter 모듈을 사용한 필터링
# ImageFilter.BLUR을 사용한 평활화
# 잡음이 많은 영상에서 잡음제거
# 맨드릴 Mandrill 개코원숭이 영상 : https://www.flickr.com/photos/uhuru1701/2249220078 : Madrill2.jpg
import numpy as np
def salt_pepper_noise(im,n):
    x,y = np.random.randint(0,im.width,n), np.random.randint(0,im.height,n)
    for (x,y) in zip(x,y):
        pix = ((0,0,0) if np.random.rand()<0.5 else (255,255,255))
        im.putpixel((x,y), pix)
i=1
import matplotlib.pylab as plb
from PIL import Image, ImageFilter
#from skimage.io import imread, imshow, show 
plb.figure(figsize=(15,20))
for prop_noise in np.linspace(0.05,0.3,3):
    im = Image.open('Mandrill2.jpg')
    n = int(im.width*im.height*prop_noise)
    salt_pepper_noise(im,n)
    plb.subplot(2,3,i), plb.imshow(im), plb.title('Noise '+str(prop_noise))
    plb.subplot(2,3,i+3), plb.imshow(im.filter(ImageFilter.BLUR)), plb.title('blurred for noise '+str(prop_noise))
    i += 1
plb.tight_layout(), plb.show()    

# 박스 블러 커널로 평균화하여 평활화
# PIL.ImageFilter.Kernel()함수 및 3*3, 5*5 크기의 박스 블러커널(평균필터)을 사용하여 잡음이 많은 영상 평활화
im = Image.open('Mandrill2.jpg')
n = int(im.width*im.height*0.1)
salt_pepper_noise(im,n)
plb.figure(figsize=(20,7))
plb.subplot(221),plb.imshow(im), plb.title('Original') 
plb.subplot(222),plb.imshow(im.filter(ImageFilter.BLUR)), plb.title('Just blur')
for n in [3,5]:
    box_blur_kernel = np.reshape(np.ones(n*n),(n,n))/(n*n)
    im1 = im.filter(ImageFilter.Kernel((n,n),box_blur_kernel.flatten()))
    plb.subplot(2,2,(3 if n==3 else 4))
    plb.imshow(im1), plb.title("Blurred with kernel size= "+str(n)+'x'+str(n))
plb.suptitle('PIL Mean Filter (Box Blur) with different Kernel size',size=20)
plb.show()

# 가우시안 블러 필터로 평활화, Box 블러, Just 블러, Median 추가
# 윈도우 내부 화소들의 가중 평균을 사용
im = Image.open('Mandrill2.jpg')
n = int(im.width*im.height*0.4)
#salt_pepper_noise(im,n) # 있을때와 없을때 비교
plb.figure(figsize=(15,10))
for radius in range(1,4):
    plb.subplot(3,3,radius+3)
    plb.imshow(im.filter(ImageFilter.GaussianBlur(radius)))
    plb.title('Gaussian, radius = '+str(radius))
    plb.subplot(3,3,radius)
    plb.imshow(im.filter(ImageFilter.BoxBlur(radius)))
    plb.title('Box, radius = '+str(radius))
plb.subplot(337), plb.imshow(im.filter(ImageFilter.MedianFilter()))
plb.title('Median, radius = 3')
plb.subplot(338), plb.imshow(im.filter(ImageFilter.BLUR)), plb.title('Just blur') # Gaussian radius 2와 비슷   
plb.suptitle('PIL Gaussian, Box, and Median',size=20)
plb.show()

# 순위값 필터
im = Image.open('Mandrill2.jpg')
n = int(im.width*im.height*0.4)
salt_pepper_noise(im,n) # 있을때와 없을때 비교
plb.figure(figsize=(15,10))
plb.subplot(221),plb.imshow(im),plb.title('Original')
plb.subplot(222),plb.imshow(im.filter(ImageFilter.MedianFilter())),plb.title('Median, radius = 3')
plb.subplot(223),plb.imshow(im.filter(ImageFilter.MaxFilter())),plb.title('Max, radius = 3')
plb.subplot(224),plb.imshow(im.filter(ImageFilter.MinFilter())),plb.title('Min, radius = 3') 
plb.suptitle('순위 필터',size=20)
plb.show()

# SciPy ndimage 를 사용한 박스 커널과 가우시안 커널의 평활화 비교
from scipy import ndimage
im = Image.open('Mandrill2.jpg')
n = int(im.width*im.height*0.1)
salt_pepper_noise(im,n) # 있을때와 없을때 비교
k,s=7,2   # 커널크기 7*7, 표준편차 2
im_box = ndimage.uniform_filter(im,size=(k,k,1))
t = (((k-1)/2)-0.5)/s  # 커널 크기와 표준편차로 truncate 파라미터 계산
im_gaussian = ndimage.gaussian_filter(im,sigma=(s,s,0),truncate=t)

fig = plb.figure(figsize=(15,10))
plb.subplot(131), plb.imshow(im), plb.title('Original')
plb.subplot(132), plb.imshow(im_box), plb.title('Box filter')
plb.subplot(133), plb.imshow(im_gaussian), plb.title('Gaussian filter')
plb.show()

## 비선형 잡음 평활화 ==> 4주차
# 필터링 연산은 조건에 따라 이웃화소값을 기반으로 하여 일반적으로 곱의 합(sum of product) 방식으로 
# 계수를 명시적으로 사용하지 않음
# 잡음 감소는 필터가 위치한 인접 영역의 중간 회색조 값을 계산하는 것이 기본 기능인 
# 비선형 필터를 사용하여 효과적으로 수행가능
# 메디안 필터 : impulse 잡음에 평균필터보다 효과적
# 스파이크와 같은 비가우시안 잡음의 억제 및 에지/텍스쳐 보존 특성
# 메디안(Median), 양방향(bidirectional), 비로컬(nonlocal means), 형태학적 필터(Morphological filter)

# PIL
# 메디안
# 각 화소를 이웃화소들의 메디안으로 바꿈
# 소금 후추 잡음 제거에 적합
# 통계적 특이치에 대한 복원력 있음
# 흐려짐이 적고 구현하기 쉽다.
import matplotlib.pylab as plb
from PIL import Image, ImageFilter
i=1
plb.figure(figsize=(30,45))
for prop_noise in np.linspace(0.05, 0.3, 3):
    im = Image.open('Mandrill2.jpg')
    n = int(im.width*im.height*prop_noise)
    salt_pepper_noise(im,n) 
    plb.subplot(6,4,i)
    title = str(int(100*prop_noise)) + '% added noise'
    plb.imshow(im), plb.title(title)
    i += 1
    for sz in [3,7,11]: #[3,5,7]:
        im1 = im.filter(ImageFilter.MedianFilter(size=sz))
        plb.subplot(6,4,i)
        plb.imshow(im1), plb.title("Median with size="+str(sz))
        i += 1
plb.show()

# 최대 및 최소 필터 사용
im = Image.open('Mandrill2.jpg')
n = int(im.width*im.height*0.1)
salt_pepper_noise(im,n) 
sz = 3
im1 = im.filter(ImageFilter.MaxFilter(size=sz))
im2 = im.filter(ImageFilter.MinFilter(size=sz))

plb.figure(figsize=(20,35))
plb.subplot(131), plb.imshow(im), plb.title('Original Image with 10% added noise')
plb.subplot(132), plb.imshow(im1), plb.title("Max filter with size= "+str(sz))
plb.subplot(133), plb.imshow(im2), plb.title("Min filter with size= "+str(sz))
plb.show()

# Scikit-image 를 사용한 평활화(잡음제거)
# restoration 모듈에 일련의 비선형 필터를 제공한다.
# 양방향 필터와 비로컬 필터
# 양방향 필터 사용
# 양방향 필터는 에지보존 평활화 필터
# 중심화소는 중심화소와 대략 비슷한밝기를 가진 화솟값 중 일부 화소 값의 가중 평균으로 설정
from skimage import color, img_as_float#, data
import numpy as np
import skimage
im = color.rgb2gray(img_as_float(imread('Mountain.png')))
sigma = 0.155
noisy = skimage.util.random_noise(im,var=sigma**2)
plb.imshow(noisy)
# sigma_color 과 sigma-spatial
plb.figure(figsize=(20,15))
i=1
for sigma_sp in [5,10,20]:
    for sigma_col in [0.1,0.25,5]:
        plb.subplot(3,3,i)
        plb.imshow(skimage.restoration.denoise_bilateral(noisy,\
                     sigma_color=sigma_col, \
                     sigma_spatial=sigma_sp,multichannel=False))
        plb.title(r'$\sigma_r=$'+str(sigma_col)+\
                     r',$\sigma_s=$'+str(sigma_sp),size=10)
        i += 1
plb.show()        
                     
# 비로컬 평균 사용
# 텍스쳐를 보존하는 비선형 잡음 알고리즘
# 임의의 주어진 화소에 대해 관심있는 화소와 유사한 로컬 이웃을 갖는 인근 화소들만의
# 가중 평균이 주어진 화소값을 설정하는 데 사용됨
# 다른 화소를 중심으로 하는 작은 패치는 관심화소를 중심으로 한 패치화 비교됨
# h 매개 변수는 패치 간의 거리의 함수로 패치 가중치의 감소를 제어함
# h가 크면 다른 패치 들 사이에서 더 부드럽게 할 수 있다.
from skimage import restoration
parrot = img_as_float(imread("parrot.png"))
sigma = 0.5
noisy = parrot + sigma*np.random.standard_normal(parrot.shape)
noisy = np.clip(noisy,0,1)

# 잡음 영상에서 잡음 표준 편차 추정
sigma_est = np.mean(skimage.restoration.estimate_sigma(noisy,multichannel=True))
sigma_est = sigma
print("Estimated noise standard deviation = {}".format(sigma_est))
# Estimated noise standard deviation = 0.1470
patch_kw = dict(patch_size=5, patch_distance=6,multichannel=True) 
#5*5 패치크기, 13*13 검색영역   
# slow algorithm
denoise = skimage.restoration.denoise_nl_means(noisy, h=1.15*sigma_est,\
    fast_mode=False,**patch_kw)
# fast algorithm     
denoise_fast = skimage.restoration.denoise_nl_means(noisy, h=1.15*sigma_est,\
    fast_mode=True,**patch_kw)
    
plb.figure(figsize=(15,12))
plb.subplot(221),plb.imshow(noisy),plb.title('Original with noise')
plb.subplot(222),plb.imshow(denoise), plb.title('non-local slow')
plb.subplot(223),plb.imshow(parrot), plb.title('Original')
plb.subplot(224),plb.imshow(denoise_fast),plb.title('non-local fast')
plb.show()   

# from skimage import measure
# print(skimage.measure.compare_psnr(parrot,noisy,data_range=1.))  

## Scipy ndimage 를 사용한 평활화
# 메디안 필터의 일반 버전인 perentil_filter() 함수 제공
from skimage.io import imread, imshow, show #, imsave
from skimage import color, viewer, img_as_float, data
from scipy import ndimage
lena = imread("lenna.png")
noise = np.random.random(lena.shape)  # 소금후추 잡음
lena[noise>0.9] = 255
lena[noise<0.1] = 0
imshow(lena)
fig = plb.figure(figsize=(20,15))
i=1
for p in range(25,100,25):
    for k in range(5,25,5):
        plb.subplot(3,4,i)
        filtered = ndimage.percentile_filter(lena,percentile=p,size=(k,k,1))
        plb.imshow(filtered)
        plb.title('p='+str(p)+'  k='+str(k))
        i += 1
plb.show()

### Ch5_Highpassfilter.py

