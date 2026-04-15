#### Ch.4. 영상 향상 
## 점 기반의 명암 변환-화소변환 
# 입력 영상의 각 화소 f(x,y)에 전달 함수 T를 적용하여 출력 영상에 해당 화소를 생성
# g(x,y)=T(f(x,y)), g=T(r), r: 입력영상 명암도, s: 출력 영상에서 동일한 화소의 변환된 명암도
# 메모리가 없는 연산, 위치 (x,y)의 출력 명암은 같은 지점의 입력 명암에만 의존
# 동일한 명암의 화소는 동일한 변환, 새로운 정보를 가져오지 않고 정보 손실을 가져옴
# 시각적인 외관을 개선하거나 특징을 쉽게 감지
# 영상처리 파이프라인의 사전 처리 단계
# 일반적으로 사용하는 밝기 변환
# 영상 네거티브, 색 공간 변환, 로그 변환, 파워-로우 변환, 콘트라스트 스트레칭, 스칼라 양자화, 임계화
import numpy as np
from skimage import img_as_float, exposure, img_as_ubyte
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.restoration import denoise_bilateral, denoise_nl_means, estimate_sigma
#from skimage.measure import compare_psnr
from skimage.util import random_noise
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage
import matplotlib.pyplot as plt

#################################
# 실습 1. 로그변환
#################################
# 영상의 특정 범위의 명암도를 압축하거나 늘일 필요가 있을 때 유용
# 푸리에 스펙트럼을 표시하기 위해 사용. 
# DC 성분 값이 다른 것보다 훨씬 크기 때문에 로그 변환없이 다른 주파수 성분은 거의 표시될 수 없다.
# s = T(r) = c.log(1+r)
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.util import img_as_ubyte

def plot_image(image,title=''):
    plt.title(title,size=20), plt.imshow(image), plt.axis('off')
def plot_hist(r,g,b,title=''):
    r,g,b = img_as_ubyte(r), img_as_ubyte(g), img_as_ubyte(b)
    r,g,b = np.array(r).ravel(), np.array(g).ravel(), np.array(b).ravel()
    plt.hist(r,bins=256,range=(0,256),color='r',alpha=0.5)
    plt.hist(g,bins=256,range=(0,256),color='g',alpha=0.5)
    plt.hist(b,bins=256,range=(0,256),color='b',alpha=0.5)
    plt.xlabel('pixel value',size=20)
    plt.ylabel('frequency',size=20), plt.title(title,size=20)
    plt.axis([0,255,0,10000])
    
im = Image.open("parrot.png")    
imr, img, imb, d = im.split()

plt.style.use('ggplot')
plt.figure(figsize=(15,5))
plt.subplot(121), plot_image(im,'original image')
plt.subplot(122), plot_hist(imr,img,imb,'histogram for RGB channels')
plt.show()

im = im.point(lambda i: 255/np.log(2)*np.log(1+i/255))
imr,img,imb,d = im.split()

plt.style.use('ggplot')
plt.figure(figsize=(15,5))
plt.subplot(121), plot_image(im,'image after log transform')
plt.subplot(122), plot_hist(imr,img,imb,'histogram of RGB channels log transform')
plt.show()

# 파워-로우 변환

## 히스토그램 처리- 히스토그램 평활화와 매칭 
# 영상의 화소값의 동적 범위를 변경하여 밝기 히스토그램이 원하는 모양을 갖도록 함
# 콘트라스 스트레칭 영상 향상은 선형 스케일링 기능만 적용
# 비선형  및 비 모노톤 전달함수를 사용하여 입력화소밝기를 출력 화소 밝기에 매핑
# scikit-image 라이브러리의 exposure 모듈 사용
# 히스토그램 평활화와 히스토그램 매칭 구현

# Scikit-image 를 이용한 콘트라스트 스트레칭과 히스토그램 평활화
# 히스토그램 평활화는 출력영상이 균일한 분포의 밝기를 가지며 영상의 콘트라스트를 향상
# 입력 영상의 화소 밝기 값을 재할당한느 단조로운 비선형 매핑
# s_k = T(r_k) = \sum_{j=0}^k P_r(r_j) = \sum_{j=0}^k n_j/N, 0\le r_k\le 1, k=0,...,255
# exposure 모듈의 equalize_hist() 함수를 사용하여 scikit-image 로 히스토그램 평활화를 수행하는 방법
# 전역 평활화 / 로컬 평활화(영상을 블록으로 나누고 각각에 히스토그램 평활화)
# import numpy as np
#from PIL import Image, ImageFont, ImageDraw
# from PIL.ImageChops import add, subtract, multiply, difference, screen
# import PIL.ImageChops as stat 
from skimage.io import imread, imshow, show #, imsave
#from skimage import color#, viewer, img_as_float, data
# from skimage.transform import SimilarityTransform, warp, swirl
# from skimage.util import random_noise
# import matplotlib.image as mpimg
#import matplotlib.pylab as plt
# from scipy import misc

from skimage import color
img = color.rgb2gray(imread("parrot.png")[:,:,:3])  
imshow(img)
from skimage import exposure
img_eq = exposure.equalize_hist(img)
imshow(img_eq)
img_adapeq = exposure.equalize_adapthist(img,clip_limit=0.03) 
imshow(img_adapeq)

###################################
## 실습 2. 평활화와 적응 평활화
###################################
# 세가지 영상 비교하기
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage import color, exposure

images =[img, img_eq, img_adapeq]
titles = ['original input', 'after hist eq',\
           'after adap hist eq']

plt.figure(figsize=(8,12))
plt.gray() 
for i in range(3):
    plt.subplot(1,3,i+1), plt.imshow(images[i]), plt.title(titles[i]),plt.axis('off')
plt.tight_layout()

plt.figure(figsize=(21,7))
for i in range(3):
    plt.subplot(1,3,i+1), plt.title(titles[i],size=15)
    plt.hist(images[i].ravel(), color='g')
plt.show()

############################################################
## 실습 3. 평활화, 콘트라스트 스트레칭, 적응 평활화
############################################################
# 히스토그램 평활화 이후 출력영상 히스토그램은 거의 균일해진다.
# 적응형 히스토그램 평활화는 전역 히스토그램 평활화보다 영상의 디테일을 더 명확하게 나타낸다.
# 다음 코드 블록은 두개의 서로 다른 히스토그램 처리 기술, 즉 콘트라스트 스트레칭과 
# 히스토그램 평활화를 사용하여 얻은 영상 향상을 scikit-image와 비교한다.
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, show #, imsave
from skimage import color, img_as_float
#import matplotlib.pylab as plb
from skimage import exposure
def plot_image_and_hist(image,axes,bins=256):
    image = img_as_float(image)
    axes_image, axes_hist = axes
    axes_cdf = axes_hist.twinx()
    axes_image.imshow(image,cmap=plt.cm.gray)
    axes_image.set_axis_off()
    
    axes_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    axes_hist.set_xlim(0,1)
    axes_hist.set_xlabel('Pixel intensity',size=15)
    axes_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    axes_hist.set_yticks([])
    
    image_cdf, bins = exposure.cumulative_distribution(image,bins)
    axes_cdf.plot(bins,image_cdf,'r')
    axes_cdf.set_yticks([])
    return axes_image, axes_hist, axes_cdf

#im = imread('LowContrastCastle.png')
#im_rescale = exposure.rescale_intensity(im, in_range=(20,80)) # 대조 스트레칭
im = imread('LowContastLenna.png')
im_rescale = exposure.rescale_intensity(im, in_range=(100,150)) # 대조 스트레칭
im_eq = exposure.equalize_hist(im)                           # 히스토그램 평활화  
im_adapteq = exposure.equalize_adapthist(im, clip_limit=0.03) # 적응적 평활화

import numpy as np
#matplotlib.rcParams(fontsize=8)
fig = plt.figure(figsize=(15,7))
axes = np.zeros((2,4), dtype=object)
for i in range(0,4):
    axes[0,i] = fig.add_subplot(2,4,i+1)
    axes[1,i] = fig.add_subplot(2,4,i+5)
    
axes_image, axes_hist, axes_cdf = plot_image_and_hist(im,axes[:,0])
axes_image.set_title('Low contrast image',size=10)
y_min, y_max = axes_hist.get_ylim()
axes_hist.set_ylabel('Number of pixels',size=10)
axes_hist.set_yticks(np.linspace(0,y_max,5))

axes_image, axes_hist, axes_cdf = plot_image_and_hist(im_rescale,axes[:,1])
axes_image.set_title('Contrast stretching',size=10)

axes_image, axes_hist, axes_cdf = plot_image_and_hist(im_eq,axes[:,2])
axes_image.set_title('Histogram equalization',size=10)

axes_image, axes_hist, axes_cdf = plot_image_and_hist(im_adapteq,axes[:,3])
axes_image.set_title('Adaptive equalization',size=10)

axes_cdf.set_ylabel("Fraction of totol intensity",size=10)
axes_cdf.set_yticks(np.linspace(0,1,5))
fig.tight_layout()
plt.show()

##################################
# 실습 4. 히스토그램 매칭
###################################
# 히스토그램이 다른 참조(탬플릿) 영상의 히스토그램의 영상과 일치하도록 영상이 변경되는 처리
# 1. 누적 히스토그램 계산 cdf
# (2,3. 주어진 화솟값, 즉 조정할 입력영상의 x_i 에 대해 )
# (   입력영상의 히스토그램을 템플릿 영상의 히스토그램과 일치시켜서 출력영상에서 해당 화솟값 x_j 를 찾는다.)
# 2. x_i 화솟값에 대한 누적 히스토그램 G(x_i)를 구한다. 
# 3. 참조영상의 누적 분포값, 즉 H(x_j)가 G(x_i)와 같도록 화솟값 x_j를 찾는다.
# 4. 입력 데이터 값 x_i 를 x_j로 대체한다.
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, color
from skimage.io import imread
def cdf(im):
    """ 2D ndarray 로서 im 영상의 cdf 계산 """
    c,b = exposure.cumulative_distribution(im)  #c: 픽셀이 있는 밝기값에 대한  cdf 값,b : 픽셀이 있는 밝기 값
    c = np.insert(c,0,[0]*b[0])     # [0]*3 =[0,0,0], c의 제일 앞 자리에 제일 작은 밝기값 b[0] 수 만큼 0을 넣어준다.[0]*b[0]
    c = np.append(c,[1]*(255-b[-1])) # 뒤에는 1을 255-b[-1] 만큼 넣어준다.
    return c

def hist_matching(c, c_t, im):
    '''
    c: cdf() 함수로 계산된 입력 영상의 cdf
    c_t: cdf() 함수로 계산된 템플릿 영상의 cdf
    im: 2D numpy ndarray로서 입력영상
    반환 값은 입력 영상에 대해 수정된 화솟값들
    '''
    # 템플릿 영상의 cdf h 값이 주어지면, 입력 영상의 cdf 에 상응하는
    # 가장 가까운 화소 매칭 검색 c_t = H(pixels)<=> pixels = H^-1(c_t)
    pixels = np.arange(256)
    new_pixels = np.interp(c,c_t,pixels)
    im = (np.reshape(new_pixels[im.ravel()], im.shape)).astype(np.uint8)
    return im        

from skimage.io import imread, imshow, show 
from skimage import color
im = (color.rgb2gray(imread('LowContrastCastle.png')[:,:,:3])*255).astype(np.uint8)
im_t = (color.rgb2gray(imread('Lenna.png')[:,:,:3])*255).astype(np.uint8)   
c, c_t = cdf(im), cdf(im_t)
    
im1 = hist_matching(c,c_t,im)
c1 = cdf(im1)

p = np.arange(256)
plt.figure(figsize=(20,12)), plt.gray()
plt.subplot(221), imshow(im), plt.axis('off'), plt.title('Input image')
plt.subplot(222), imshow(im_t), plt.axis('off'), plt.title('Template image')
plt.subplot(224), plt.plot(p,c,'r.-', label='input'),plt.plot(p,c_t,'b.-',label='template')
plt.subplot(223), plt.plot(p,c1,'g-.',label='Matching')
plt.show()

plt.imshow(im1, cmap='gray')   # show the transformed image
plt.axis('off')
plt.title('Matched image')
plt.show()
##################################
# 실습 5. RGB 영상의 히스토그램 매칭
##################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from skimage import color
from skimage.io import imread

im = (color.rgba2rgb(imread('LowContrastCastle.png'))*255).astype(np.uint8)
im_t = (color.rgba2rgb(imread('Lenna.png'))*255).astype(np.uint8) 
#imR = im[:,:,1]  
cR , cG, cB, c_tR, c_tG, c_tB = cdf(im[:,:,0]), cdf(im[:,:,1]), cdf(im[:,:,2]), cdf(im_t[:,:,0]), cdf(im_t[:,:,1]), cdf(im_t[:,:,2])
    
im1R, im1G, im1B = hist_matching(cR,c_tR,im[:,:,0]), hist_matching(cG,c_tG,im[:,:,1]), hist_matching(cB,c_tB,im[:,:,2])
c1R, c1G, c1B = cdf(im1R), cdf(im1G), cdf(im1B)
# a = im1R.shape; im1 = np.zeros(int(a[0]), int(a[1]), 3)
im1 = im
im1[:,:,0] = im1R;  im1[:,:,1]=im1G;  im1[:,:,2]=im1B #, imshow(im1)

#import matplotlib.pylab as plb
p = np.arange(256)
plt.figure(figsize=(20,10)) #, plb.gray()
plt.subplot(231), imshow(im), plt.axis('off'), plt.title('Input image',size=20)
plt.subplot(232), imshow(im_t), plt.axis('off'), plt.title('Template image',size=20)
plt.subplot(233), imshow(im1), plt.axis('off'), plt.title('Hist.Matching',size=20)
plt.subplot(234), plt.plot(p,cR,'r.-', label='input'),plt.plot(p,c_tR,'y.-',label='template'), plt.plot(p,c1R,'b-.',label='Matching')
plt.legend(), plt.title('CDF_Red',size=20)          
plt.subplot(235), plt.plot(p,cG,'r.-', label='input'),plt.plot(p,c_tG,'y.-',label='template'), plt.plot(p,c1G,'k-.',label='Matching')
plt.legend(), plt.title('CDF_Green',size=20)   
plt.subplot(236), plt.plot(p,cB,'r.-', label='input'),plt.plot(p,c_tB,'y.-',label='template'), plt.plot(p,c1B,'k-.',label='Matching')
plt.legend(), plt.title('CDF_Blue',size=20)   
plt.show()

#####################################
## 실습 6. 소금후추 잡음과 블러 필터링
####################################
## 선형 잡음 필터링 
# 박스필터, 가우시안 필터
# PIL을 사용한 필터링
# PIl ImageFilter 모듈을 사용한 필터링
# ImageFilter.BLUR을 사용한 평활화
# 잡음이 많은 영상에서 잡음제거
# 맨드릴 Mandrill 개코원숭이 영상 : https://www.flickr.com/photos/uhuru1701/2249220078 : Madrill2.jpg
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

def salt_pepper_noise(im,n):
    x,y = np.random.randint(0,im.width,n), np.random.randint(0,im.height,n)
    for (x,y) in zip(x,y):
        pix = ((0,0,0) if np.random.rand()<0.5 else (255,255,255))
        im.putpixel((x,y), pix)
i=1

plt.figure(figsize=(15,10))
for prop_noise in np.linspace(0.05,0.3,3):
    im = Image.open('Mandrill2.jpg')
    n = int(im.width*im.height*prop_noise)
    salt_pepper_noise(im,n)
    plt.subplot(2,3,i), plt.imshow(im), plt.title('Noise '+str(prop_noise),size=20), plt.axis('off')
    plt.subplot(2,3,i+3), plt.imshow(im.filter(ImageFilter.BLUR)), plt.axis('off')
    plt.title('blurred for noise '+str(prop_noise),size=20)
    i += 1
plt.tight_layout(), plt.show()    

###########################################
# 실습 7.박스 블러 커널로 평균화하여 평활화
############################################
# PIL.ImageFilter.Kernel()함수 및 3*3, 5*5 크기의 박스 블러커널(평균필터)을 사용하여 잡음이 많은 영상 평활화
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

im = Image.open('Mandrill2.jpg')
n = int(im.width*im.height*0.1)
salt_pepper_noise(im,n)
plt.figure(figsize=(12,10))
plt.subplot(221),plt.imshow(im), plt.axis('off'), plt.title('Original') 
plt.subplot(222),plt.imshow(im.filter(ImageFilter.BLUR)), plt.axis('off'), plt.title('Just blur')
for n in [3,5]:
    box_blur_kernel = np.reshape(np.ones(n*n),(n,n))/(n*n)
    im1 = im.filter(ImageFilter.Kernel((n,n),box_blur_kernel.flatten()))
    plt.subplot(2,2,(3 if n==3 else 4))
    plt.imshow(im1), plt.axis('off'), plt.title("Blurred with kernel size= "+str(n)+'x'+str(n))
plt.suptitle('PIL Mean Filter (Box Blur) with different Kernel size',size=20)
plt.show()

################################
# 실습 8. 가우시안 블러 필터로 평활화
####################################
# Box 블러, Just 블러, Median 추가
# 윈도우 내부 화소들의 가중 평균을 사용
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

im = Image.open('Mandrill2.jpg')
n = int(im.width*im.height*0.4)
#salt_pepper_noise(im,n) # 있을때와 없을때 비교
plt.figure(figsize=(12,10))
for radius in range(1,4):
    plt.subplot(3,3,radius+3), plt.imshow(im.filter(ImageFilter.GaussianBlur(radius)))
    plt.axis('off'), plt.title('Gaussian, radius = '+str(radius))
    plt.subplot(3,3,radius), plt.imshow(im.filter(ImageFilter.BoxBlur(radius)))
    plt.axis('off'), plt.title('Box, radius = '+str(radius))
plt.subplot(337), plt.imshow(im.filter(ImageFilter.MedianFilter()))
plt.axis('off'), plt.title('Median, radius = 3')
plt.subplot(338), plt.imshow(im.filter(ImageFilter.BLUR)), plt.title('Just blur') # Gaussian radius 2와 비슷   
plt.axis('off'), plt.suptitle('PIL Gaussian, Box, and Median',size=20)
plt.show()

##########################
# 실습 9. 순위값 필터
##########################

import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

im = Image.open('Mandrill2.jpg')
n = int(im.width*im.height*0.4)
salt_pepper_noise(im,n) # 있을때와 없을때 비교
plt.figure(figsize=(12,10))
plt.subplot(221),plt.imshow(im), plt.axis('off'), plt.title('Original')
plt.subplot(222),plt.imshow(im.filter(ImageFilter.MedianFilter())), plt.axis('off'), plt.title('Median, radius = 3')
plt.subplot(223),plt.imshow(im.filter(ImageFilter.MaxFilter())), plt.axis('off'), plt.title('Max, radius = 3')
plt.subplot(224),plt.imshow(im.filter(ImageFilter.MinFilter())), plt.axis('off'), plt.title('Min, radius = 3') 
plt.suptitle('Order Filter',size=20)
plt.show()

#####################################
## 실습 10. SciPy ndimage 를 사용한 박스 커널과 가우시안 커널의 평활화 비교
#################################################
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image
im = Image.open('Mandrill2.jpg')
n = int(im.width*im.height*0.1)
salt_pepper_noise(im,n) # 있을때와 없을때 비교
k,s=7,2   # 커널크기 7*7, 표준편차 2
im_box = ndimage.uniform_filter(im,size=(k,k,1))
t = (((k-1)/2)-0.5)/s  # 커널 크기와 표준편차로 truncate 파라미터 계산
im_gaussian = ndimage.gaussian_filter(im,sigma=(s,s,0),truncate=t)

fig = plt.figure(figsize=(15,10))
plt.subplot(131), plt.imshow(im), plt.axis('off'), plt.title('Original')
plt.subplot(132), plt.imshow(im_box), plt.axis('off'), plt.title('Box filter')
plt.subplot(133), plt.imshow(im_gaussian), plt.axis('off'), plt.title('Gaussian filter')
plt.show()

######################################
## 비선형 필터
######################################
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

##################################
## 실습 11. 메디안 필터 
##################################
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
i=1
plt.figure(figsize=(30,45))
for prop_noise in np.linspace(0.05, 0.3, 3):
    im = Image.open('Mandrill2.jpg')
    n = int(im.width*im.height*prop_noise)
    salt_pepper_noise(im,n) 
    plt.subplot(6,4,i)
    title = str(int(100*prop_noise)) + '% added noise'
    plt.imshow(im), plt.axis('off'), plt.title(title,size=20)
    i += 1
    for sz in [3,7,11]: #[3,5,7]:
        im1 = im.filter(ImageFilter.MedianFilter(size=sz))
        plt.subplot(6,4,i), plt.imshow(im1)
        plt.axis('off'), plt.title("Median with size="+str(sz),size=20)
        i += 1
plt.show()

##################################
## 실습 12. 최대 및 최소 필터 사용
##################################
from PIL import Image, ImageFilter      # For image loading and filtering
import matplotlib.pyplot as plt         # For plotting images
import numpy as np                      # For random number generation and numerical operations
im = Image.open('Mandrill2.jpg')
n = int(im.width*im.height*0.1)
salt_pepper_noise(im,n) 
sz = 3
im1 = im.filter(ImageFilter.MaxFilter(size=sz))
im2 = im.filter(ImageFilter.MinFilter(size=sz))

plt.figure(figsize=(20,35))
plt.subplot(131), plt.imshow(im), plt.axis('off'), plt.title('Original Image with 10% added noise')
plt.subplot(132), plt.imshow(im1), plt.axis('off'), plt.title("Max filter with size= "+str(sz))
plt.subplot(133), plt.imshow(im2), plt.axis('off'), plt.title("Min filter with size= "+str(sz))
plt.show()

##################################################
## 실습 13.양방향 필터
###################################################
# Scikit-image 를 사용한 평활화(잡음제거)
# restoration 모듈에 일련의 비선형 필터를 제공한다.
# 양방향 필터와 비로컬 필터
# 양방향 필터 사용
# 양방향 필터는 에지보존 평활화 필터
# 중심화소는 중심화소와 대략 비슷한밝기를 가진 화솟값 중 일부 화소 값의 가중 평균으로 설정
from skimage import color, img_as_float#, data
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.io import imread
im = color.rgb2gray(img_as_float(imread('Mountain.png')[:,:,:3]))
sigma = 0.155
noisy = skimage.util.random_noise(im,var=sigma**2)
plt.imshow(noisy), plt.axis('off'), plt.title('Noised image'), plt.show()
# sigma_color 과 sigma-spatial
plt.figure(figsize=(20,15))
i=1
for sigma_sp in [5,10,20]:
    for sigma_col in [0.1,0.25,5]:
        plt.subplot(3,3,i), plt.axis('off')
        plt.imshow(skimage.restoration.denoise_bilateral(noisy,
                     sigma_color=sigma_col,
                     sigma_spatial=sigma_sp))
        plt.title(r'$\sigma_r=$'+str(sigma_col)+\
                     r', $\sigma_s=$'+str(sigma_sp),size=20)
        i += 1
plt.show()        
 
###########################################
## 실습 14. 비로컬 필터
###############################################                    
# 비로컬 평균 사용
# 텍스쳐를 보존하는 비선형 잡음 알고리즘
# 임의의 주어진 화소에 대해 관심있는 화소와 유사한 로컬 이웃을 갖는 인근 화소들만의
# 가중 평균이 주어진 화소값을 설정하는 데 사용됨
# 다른 화소를 중심으로 하는 작은 패치는 관심화소를 중심으로 한 패치화 비교됨
# h 매개 변수는 패치 간의 거리의 함수로 패치 가중치의 감소를 제어함
# h가 크면 다른 패치 들 사이에서 더 부드럽게 할 수 있다.

from skimage import img_as_float, restoration
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt

parrot = img_as_float(imread("parrot.png")[:,:,:3])
sigma = 0.5
noisy = parrot + sigma*np.random.standard_normal(parrot.shape)
noisy = np.clip(noisy,0,1)

# 잡음 영상에서 잡음 표준 편차 추정
sigma_est = np.mean(restoration.estimate_sigma(noisy, channel_axis=-1))
sigma_est = sigma
print("Estimated noise standard deviation = {}".format(sigma_est))
# Estimated noise standard deviation = 0.1470

patch_kw = dict(patch_size=5, patch_distance=6) 
# 5*5 패치크기, 13*13 검색영역   

# slow algorithm
denoise = restoration.denoise_nl_means(noisy, h=1.15*sigma_est, fast_mode=False, **patch_kw)

# fast algorithm     
denoise_fast = restoration.denoise_nl_means(noisy, h=1.15*sigma_est, fast_mode=True, **patch_kw)
    
plt.figure(figsize=(12,12))
plt.subplot(221), plt.imshow(noisy), plt.axis('off'), plt.title('Original with noise', size=15)
plt.subplot(222), plt.imshow(denoise), plt.axis('off'), plt.title('non-local slow', size=15)
plt.subplot(223), plt.imshow(parrot), plt.axis('off'), plt.title('Original', size=15)
plt.subplot(224), plt.imshow(denoise_fast), plt.axis('off'), plt.title('non-local fast', size=15)
plt.show()

##############################################
## 실습 15. Scipy.ndimage를 이용한 평활화
#############################################
lena = imread("Lenna.png")
noise = np.random.random(lena.shape)
lena[noise>0.9] = 255
lena[noise<0.1] = 0

plot_image(lena, 'noisy image'), plt.show()

fig = plt.figure(figsize=(20,15))
i = 1
for p in range(25,100,25):
    for k in range(5,25,5):
        plt.subplot(3,4,i)
        filtered = ndimage.percentile_filter(lena,percentile=p, size=(k,k,1))
        plot_image(filtered, str(p)+'percentile, '+ str(k) + 'x' + str(k) + 'kernel')
        plt.axis('off')
        i += 1
plt.show()    