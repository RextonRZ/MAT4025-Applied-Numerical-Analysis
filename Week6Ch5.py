##### Ch5. 미분을 사용한 영상향상
# 영상향상 : 영상의 외관이나 유용성을 개선
# 영상 그래디언트/미분 계산을 위한 공간필터링 기술
# 영상 에지 검출
# 1차 미분, 이산 미분, 2차 미분, 라플라시안
# 영상을 sharpen, unsharp 하는 기술
# 다양한 필터 Sobel, Canny, LoG
# 가우시안/라플라시안 영상 피라미드
# 영상 피라미드를 사용하여 두 영상을 부드럽게 블렌딩하는 방법
# 영상미분-그레디언트, 라플라시안
# 샤퍼닝과 언샤프 매스킹
# 미분과 필터를 사용한 에지 검출
# 영상 피라미드(가우시안과 라플라시안)-블렌딩 영상

## 영상 미분- 그레디언트 및 라플라시안
import numpy as np
from skimage import filters, feature, img_as_float
from skimage.io import imread #, imshow, show
from skimage.color import rgb2gray
from scipy import signal, ndimage
import matplotlib.pyplot as plb

####################################
# 실습 1. 미분과 그래디언트
####################################
# df/dx ~ f(x+1,y)-f(x,y) ~ (f(x+1)-f(x-1))/2
# df/dy ~ (f(x,y+1)-f(x,y-1))/2
# grad f = (df/dx, df/dy)
# magnitude ||grad f|| = sqrt(df/dx**2+ df/dy**2)
# direction theta = atan((df/dy)/(df/dx))
kerx = [[-1,1]]
kery = [[-1],[1]]
im = rgb2gray(imread("Chess.png")[:,:,:3])
imx = signal.convolve2d(im,kerx,mode='same')
imy = signal.convolve2d(im,kery,mode='same')
immag = np.sqrt(imx**2+imy**2)
imang = np.arctan(imy/(imx+0.01)) + np.arctan(imx/(imy+0.01))

plb.figure(figsize=(15,10)), plb.gray()
plb.subplot(231), plb.imshow(im),plb.title('original')
plb.subplot(232), plb.imshow(imx), plb.title('grad x')
plb.subplot(233), plb.imshow(imy), plb.title('grad y')
plb.subplot(234), plb.imshow(immag), plb.title('||grad||')
plb.subplot(235), plb.imshow(imang), plb.title('angle(grad)')
plb.subplot(236) 
plb.plot(range(im.shape[1]), im[0,:],'b-', label=r'$f(x,y)|_{(x=0)}$')
plb.plot(range(im.shape[1]), imx[0,:],'r-', label=r'$gradx(f(x,y))|_{(x=0)}$')
plb.title(r'$gradx(f(x,y))|_{(x=0)}$')
plb.legend()
plb.show()

# # 동일영상에 크기 및 그래디언트 표시
# # g(x,y,R)=grad I |.sin(theta)
# # g(x,y,G)=grad I |.cos(theta)
# # g(c,y,B)=0
# im2 = np.zeros((im.shape[0],im.shape[1],3))
# im2[...,0]= immag*np.sin(imang)
# im2[...,1]= immag*np.cos(imang)
# plb.imshow(im2)

####################################
# 실습 2. 미분과 그래디언트 tiger
####################################
im = rgb2gray(imread("tiger.png")[...,:3])
imx = signal.convolve2d(im,kerx,mode='same')
imy = signal.convolve2d(im,kery,mode='same')
immag = np.sqrt(imx**2+imy**2)
imang = np.arctan(imy/(imx+0.0001))
plb.figure(figsize=(15,10)), plb.gray()
plb.subplot(231), plb.imshow(im),plb.title('original')
plb.subplot(232), plb.imshow(imx), plb.title('grad x')
plb.subplot(233), plb.imshow(imy), plb.title('grad y')
plb.subplot(234), plb.imshow(immag), plb.title('||grad||')
plb.subplot(235), plb.imshow(imang), plb.title('angle(grad)')
#plb.subplot(235), plb.imshow(im+immag), plb.title('ori+|grad|') 
plb.subplot(236), plb.imshow(immag+0.1*imang), plb.title('|grad|+0.1*angle') 
# plb.plot(range(im.shape[1]), im[0,:],'b-', label=r'$f(x,y)|_{(x=0)}$')
# plb.plot(range(im.shape[1]), imx[0,:],'r-', label=r'$gradx(f(x,y))|_{(x=0)}$')
# plb.title(r'$gradx(f(x,y))|_{(x=0)}$')
# plb.legend()
plb.show() 

# 라플라시안
# Rosenfeld와 Kak은 가장 단순한 등방성 비분 연산자가 라플라시안이라는 것을 보임
# 2차 미분, 등방성(회전 불변), 제로 크로싱은 에지 위치 표시
# 1차 미분에서 스파이크/피크 또는 계곡이 있는 위치에서 2차 미분에 대응하는 위치에 제로 크로싱
# d^2f/dx^2 = f(x+1,y)-2f(x,y)+f(x-1,y)
# d^2f/dy^2 = f(x,y+1)-2f(x,y)+f(x,y-1)
# laplace f = d^2f/dx^2 + d^2f/dy^2
# = f(x+1,y)+f(x-1,y)+f(x,y+1)+f(x,y-1)-4f(x,y)
# 라플라시안에 관한 몇가지 주의사항
# Laplace f 는 스칼라이다. 그래디언트는 백터다
# 라플라시안을 계산하기 위해 하나의 커널이 사용된다.
# 그래디언트는 x,y방향 두개의 커널을 사용한다.
# 스칼라이기 때문에 방향이 없으므로 방향 정보가 손실된다.
# 라플라시안은 2차 편미분의 합계이다. 
# 미분의 차수가 높으면 잡음이 증가한다.
# 라플라시안은 잡음에 민감하다.
# 라플라시안 전에 저주파필터를 먼저 수행한다. 
###########################################
## 실습 3. 라플라시안
##########################################
ker_laplacian =[[0,-1,0],[-1,4,-1],[0,-1,0]] 
im = rgb2gray(imread('Chess.png')[:,:,:3])
im1 = np.clip(signal.convolve2d(im,ker_laplacian,mode='same'),0,1)
plb.gray()
plb.figure(figsize=(20,10))
plb.subplot(131), plb.imshow(im), plb.title('Ori',size=20)
plb.subplot(132), plb.imshow(im1), plb.title('Laplacian',size=20)
plb.subplot(133), plb.imshow(im+im1), plb.title('Ori+Lap',size=20)
plb.show()

####################################################
## 실습 4. 그래디언트 계산에서 잡음의 영향
#####################################################
# 유한 차분을 사용하여 계산된 미분 필터는 잡음에 매우 미약
# 이웃들과 매우 다른 밝기 값을 갖는 영상의 화소는 일반적으로 잡음 화소
# 잡음이 많을수록 밝기의 변화가 커지고 필터의 응답이 강함
from skimage.util import random_noise
sigma = 1
im = im + random_noise(im,var=sigma**2)
#plb.imshow(im)
kerx = [[-1,1]]
kery = [[-1],[1]]
imx = signal.convolve2d(im,kerx,mode='same')
imy = signal.convolve2d(im,kery,mode='same')
immag = np.sqrt(imx**2+imy**2)
imang = np.arctan(imy/(imx+0.0001))
plb.figure(figsize=(15,10)), plb.gray()
plb.subplot(231), plb.imshow(im),plb.title('original',size=20)
plb.subplot(232), plb.imshow(imx), plb.title('grad x',size=20)
plb.subplot(233), plb.imshow(imy), plb.title('grad y',size=20)
plb.subplot(234), plb.imshow(immag), plb.title('||grad||',size=20)
plb.subplot(235), plb.imshow(imang), plb.title('angle(grad)',size=20)
#plb.subplot(235), plb.imshow(im+immag), plb.title('ori+|grad|') 
plb.subplot(236) # , plb.imshow(immag+0.1*imang), plb.title('|grad|+0.1*angle') 
plb.plot(range(im.shape[1]), im[0,:],'b-', label=r'$f(x,y)|_{(x=0)}$')
plb.plot(range(im.shape[1]), imx[0,:],'r-', label=r'$gradx(f(x,y))|_{(x=0)}$')
plb.title(r'$gradx(f(x,y))|_{(x=0)}$',size=20)
plb.legend()
plb.show() 

#########################################################
# 미분 필터를 작용하기 전에 영상을 평활화
# 먼저 LPF(예: 가우시안 필터)를 사용하여 입력 영상을 부드럽게
# 평활화된 영상에서 피크(임계치 사용)를 찾는다.
# 이것은 LoG 필터(2차 미분필터를 사용하는경우)의 아이디를 발생

## 샤프닝과 언샤프 마스킹
#########################################################
## 실습 5. 샤프닝 필터
#########################################################
# 샤프닝 목적 : 영상의 세부사항 강조, 흐리게 처리된 세부사항 향상
# 라플라시안을 사용한 샤프닝
# 1. 원본 임의 영상에 라플라시안 적용
# 2. 라플라시안 영상 + 원본 영상
# scikit-image filters 모듈의 laplace()
from skimage.filters import laplace
im = rgb2gray(imread("Mandrill2.jpg"))
im1= np.clip(laplace(im)+im,0,1)

plb.figure(figsize=(15,10))
plb.subplot(121), plb.imshow(im), plb.axis('off'), plb.title('Ori',size=20)
plb.subplot(122), plb.imshow(im1), plb.axis('off'), plb.title('sharpen',size=20)
plb.show()

############################################
## 실습 6. 언샤프 마스킹
############################################
# 영상 - 흐려진 영상
# sharpend = ori + (ori - blurred)*amount
# Scipy ndimage 모듈 사용
def rgb2gray2(im):
    gray = 0.2989*im[...,0]+0.587*im[...,1]+0.114*im[...,2]
    return np.clip(gray,0,1)
im = rgb2gray2(imread("Mandrill2.jpg")/255)
im_blurred = ndimage.gaussian_filter(im,5)
im_detail = np.clip(im-im_blurred,0,1)

fig, axes = plb.subplots(2,3,figsize=(15,10))
axes = axes.ravel()
axes[0].set_title('Original  image',size=20), axes[0].imshow(im)
axes[1].set_title('Blurred sigma=5',size=20), axes[1].imshow(im_blurred)
axes[2].set_title('Detail image',size=20), axes[2].imshow(im_detail)

for i,alpha in enumerate([1,5,10]):
    im_sharp = np.clip(im+alpha*im_detail,0,1)
    axes[i+3].imshow(im_sharp)
    axes[i+3].set_title('Sharpend, alpha='+str(alpha),size=20)

for ax in axes: ax.axis('off')
plb.show()

## 미분과 필터(소벨,캐니 등)를 사용한 에지 검출
# 에지를 구성하는 요소는 영상밝기에 급격한 변화가 있는 화소
# 에지 검출은 입력이 2차원 명암도 영상이고 출력이 곡선 집합(에지)인 전처리 기술
# 에지를 사용한 영상표현은 화소를 사용하는 영상 표현보다 훨씬 간결하다.
# 그래디언트 크기를 임계치로 구분, numpy.clip( , 0,1)은 0보다 작은 것은 0, 1보다 큰 것은 1
# 이진 영상 <==  Otsu 분할 등
# 편미분의 유한 차분 근사법을 사용하여 계산된 그래디언트의 크기로 에지 검출, 소벨 필터

# 편미분을 사용해 계산된 그래디언트 크기
# 에지가 두껍고  다중 화소인 경우
# 한 화소 너비의 각 에지가 이진 영상을 얻으려면 화소 주위의 그래디언트 방향을 따라 
# 로컬 최대가 아닌 화소를 제거하는 비 최대 억제 non maximum supression 알고리즘 적용

#####################################
# 실습 7.Sobel
#####################################
import numpy as np
from skimage import filters, feature, img_as_float
from skimage.io import imread, imshow, show
from skimage.color import rgb2gray
import matplotlib.pylab as plb

im = rgb2gray(imread("victoria_memorial.png")[:,:,:3])
edgex = np.clip(filters.sobel_h(im),0,1)
edgey = np.clip(filters.sobel_v(im),0,1)
edges = filters.sobel(im)

plb.figure(figsize=(20,10)), plb.gray()
plb.subplot(221), plb.imshow(im), plb.axis('off'), plb.title('Original',size=25)
plb.subplot(222), plb.imshow(edgex), plb.axis('off'), plb.title('SobelX',size=25)
plb.subplot(223), plb.imshow(edgey), plb.axis('off'), plb.title('SobelY',size=25)
plb.subplot(224), plb.imshow(edges), plb.axis('off'), plb.title('Sobel',size=25)

##################################################################
## 실습 8. 에지 검출기: 프리윗, 로버츠, 소벨, 스칼(Scharr), 라플라스
#################################################################
# Scikit-image를 사용하는 다른 에지 검출기-프리윗, 로버츠, 소벨, 스칼(Scharr) 및 라플라스
im = rgb2gray(imread('Goldengate.png')[...,:3])
edges = []
edges.append(im)
edges.append(filters.roberts(im))
edges.append(filters.scharr(im))
edges.append(filters.sobel(im))
edges.append(filters.prewitt(im))
edges.append(np.clip(filters.laplace(im),0,1)) 

titles = ['original','Roberts','Scharr','Sobel','Prewitt','Laplace']
plb.figure(figsize=(20,10)), plb.gray()
for i,edge in enumerate(edges):
    plb.subplot(2,3,i+1), plb.imshow(edge), plb.axis('off'), plb.title(titles[i],size=30)
plb.show()    

# Scikit-image를 사용하여 캐니 에지 검출기
# John F. Canny
# 1. 평활화/잡음 감소 : 5*5 가우시안 필터
# 2. 그래디언트의 크기와 방향 계산 : 소벨 수평 및 수직 필터를 영상에 적용하여 
#     에지 그래디언트의 크기와 방향 계산, 
#     계산된 각도는 수평, 수직 및 2개의 대각선 방향을 나타내는 4개 각도 중 하나로 반올림
# 3. 비 최대 억제 : 에지가 얇아진다. 에지를 구성하지 않을 수 있는 임의의 원치않는 화소 제거
#     모든 화소는 그래디언트 방향으로 그 이웃에 있는 로컬 최대 값인지 검사. 
#     결과적으로 얇은 에지를 가진 이진 영상 획득
# 4. 연결과 히스테리시스 hysteresis 임계치 : 검출된 모든 에지가 강한 에지인지 여부 결정.
#     두 가지 임계치 min_val 과 max_val 사용, 확실한 에지는 max_val 보다 높은 그래디언트 값 가진 에지
#     min_val 아래 밝기 그래디언트 값 가진 에지는 버려짐
#     이 두 임계치 사이 에지는 연결성에 따라 에지 또는 비-에지로 분류
#     확실한 화소에 연결되면 에지의 일부, 그렇지 않으면 버려짐, 결과적으로 작은 화소 잡음 제거
# 영상의 강한 에지 출력
#im =rgb2gray(imread())
##############################################
## 실습 9. Canny 에지 검출기
##############################################
from skimage import filters, feature, img_as_float
from scipy import ndimage
im2 = im.copy()         
im2 = ndimage.gaussian_filter(im2,4)
im2 += 0.05 * np.random.random(im2.shape)
# No noise
edges1 = feature.canny(im)
edges2 = feature.canny(im,sigma=0.5)
edgesS = filters.sobel(im)

plb.figure(figsize=(18,12))
plb.subplot(221), plb.imshow(im), plb.axis('off'), plb.title('No noise Image', size=20)
plb.subplot(223), plb.imshow(edges1), plb.axis('off'), plb.title('Canny $\sigma=1$',size=20)
plb.subplot(224), plb.imshow(edges2), plb.axis('off'), plb.title('Canny $\sigma=0.5$',size=20)
plb.subplot(222), plb.imshow(edgesS), plb.axis('off'), plb.title('Sobel',size=20)
# Noise Gaussian 5%
edges1 = feature.canny(im2)
edges2 = feature.canny(im2,sigma=0.5)
edgesS = filters.sobel(im2)

plb.figure(figsize=(18,12))
plb.subplot(221), plb.imshow(im2), plb.axis('off'), plb.title('5% Noisy Image',size=20)
plb.subplot(223), plb.imshow(edges1), plb.axis('off'), plb.title('Canny $\sigma=1$',size=20)
plb.subplot(224), plb.imshow(edges2), plb.axis('off'), plb.title('Canny $\sigma=0.5$',size=20)
plb.subplot(222), plb.imshow(edgesS), plb.axis('off'), plb.title('Sobel',size=20)

# LoG와 DoG 필터들
# Laplacian of Gaussian : 2차 미분은 잡음에 매우 민감하기 때문에 라플라시안 적용 전 평활화하여 잡음 제거
# 컨볼류션의 결합 속성 때문에 가우시안 필터의 2차 미분을 취한 다음, 결과 필터를 영상에 적용
# 다른 스케일(분산)을 가진 두 가우시안의 차이(DoG: Difference of two Gaussian)를 사용
# G_sigma = 1/(2pi sigma^2) exp(-(x^2+y^2)/(2 sigma^2))
# d^2 G_sigma(x,y)/dx^2 = 1/(2pi sigma^4) exp(-(x^2+y^2)/(2 sigma^2))(x^2/sigma^2 -1)
# d^2 G_sigma(x,y)/dy^2 = 1/(2pi sigma^4) exp(-(x^2+y^2)/(2 sigma^2))(y^2/sigma^2 -1)
# LoG(x,y) = Laplace G_sigma(x,y) = d^2 G_sigma/dx^2 + d^2 G_sigma/dy^2 
#          = -1/(pi sigma^2) exp(-(x^2+y^2)/(2 sigma^2)) (1-(x^2+y^2)/(2 sigma^2))
# DoG(x,y) = Laplac G_sigma ~ G_sigma1 - G_sigma2,   sigma_1 ~ sqrt 2 sigma,  sigma_2 ~ sigma/sqrt 2
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from numpy import pi
def plot_kernel(kernel,s,name):
    plb.title(name,size=20),
    plb.imshow(kernel,cmap='YlOrRd') # jet or gray_r
    ax = plb.gca()
    ax.set_xticks(np.arange(-0.5, kernel.shape[0], 2.5))
    ax.set_yticks(np.arange(-0.5, kernel.shape[1], 2.5))
    plb.colorbar()
    
def LOG(k=12,s=3):
    n = 2*k+1
    kernel = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            x,y = j-k, i-k
            dG  =(x**2 + y**2)/(2*s**2)
            kernel[i,j] = (1-dG)*np.exp(-dG)/(pi*s**4)
    kernel = np.round(kernel/np.sqrt((kernel**2).sum()),3) 
    return kernel

def DOG(k=12,s=3):
    n = 2*k+1
    s1,s2 = s*np.sqrt(2), s/np.sqrt(2)
    kernel = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            x,y = j-k,i-k
            dG1 =(x**2 + y**2)/(2*s1**2)
            dG2 =(x**2 + y**2)/(2*s2**2)
            kernel[i,j] = np.exp(-dG1)/(2*pi*s1**2) - np.exp(-dG2)/(2*pi*s2**2)
    kernel = np.round(kernel/np.sqrt((kernel**2).sum()),3)
    return kernel        

im3 = im.copy()
s = 3
kernel1 = LOG()
kernel2 = DOG()
outim1 = convolve2d(im3,kernel1)
outim2 = convolve2d(im3,kernel2)

plb.figure(figsize=(20,18))
plb.subplot(221), plb.axis('off'), plot_kernel(kernel1,s,'LOG')
plb.subplot(222), plb.axis('off'), plb.title('Output with LOG',size=20), plb.imshow(np.clip(outim1,0,1))
# plb.imshow(outim1) 
plb.subplot(223), plb.axis('off'), plot_kernel(kernel2,s,'DOG')
plb.subplot(224), plb.axis('off'), plb.title('Output with DOG',size=20), plb.imshow(np.clip(outim2,0,1))
# plb.imshow(outim2)
plb.show()

#########################################333
## 실습 11. LoG
#############################################
# LoG는 입력영상에서 Bans Pass Filter로 작동한다.(낮은 주파수와 높은 주파수를 모두 차단하므로) 
# LoG의 대역 통과 특성은 또한 DoG 근사법으로 설명할 수 있다.
# LoG/DoG 필터로 얻은 출력영상들이 매우 유사하다는 것을 알 수 있다. 
# LoG 필터는 에지 검출에 매우 유용하다
# LoG는 또한 영상에서 블롭blob 을 찾는데 유용하다.
#SciPy ndimage 모듈을 사용한 LoG 필터
im4 = im.copy()
fig = plb.figure(figsize=(18,12)), plb.gray()
for sigma in range(1,10):
    plb.subplot(3,3,sigma)
    img_log = ndimage.gaussian_laplace(im4,sigma=sigma)
    plb.imshow(np.clip(img_log,0,1)),plb.axis('off')
    plb.title('LoG with sigma='+str(sigma),size=20)
plb.show()    

# LoG 필터를 사용한 에지 검출
# 1. 가우시안 필터로 평활화
# 2. 평활화된 영상을 라플라시안 필터와 컨볼루션
# 3. 제로 크로싱 계산
# Laplace(f*G) = Laplace(G)*f
# Laplacian of Gaussian filtered image = Laplacian of Gaussian(LoG)-filtered image

# 제로 크로싱 계산을 사용한 마르Marr와 힐드레스Hildreth 알고리즘의 에지 검출
# Marr와 Hildreth 는 LoG 컨볼루션된 영상의 제로크로싱을 계산하여 이진영상으로 에지검출
# 1. LoG컨볼루션 영상을 이진 영상으로 변환(양수는 1, 음수는 0)
# 2. 0인 영역의 경계를 살펴봄
# 3. 1이 가까이 있는 0인 화소를 찾음
# 4. 어떤 화소가 0이면서 8이웃 중에 1이 있으면 그 화소는 에지로 간주함. 
def zero_crossing(im): #직접 짜세요.
    tim = img_as_float(im>0)
    outim = np.zeros(list(tim.shape))
    for i in range(tim.shape[0]):
        ipre = max(i-1,0)
        inex = min(i+1,tim.shape[0]-1)+1
        for j in range(tim.shape[1]):
            if tim[i,j]==0:
                jpre = max(j-1,0)
                jnex = min(j+1,tim.shape[1]-1)+1
                ijsum = inex-ipre + jnex-jpre 
                if tim[ipre:inex,jpre:jnex].sum(): #-ijsum:
                    outim[i,j]=1
    return outim

im5 = im.copy()
fig = plb.figure(figsize=(14,20)), plb.gray()
for sigma in range(2,10,2):
    result = ndimage.gaussian_laplace(im5,sigma=sigma)
    plb.subplot(4,2,sigma-1), plb.axis('off'),
    plb.imshow(result), plb.title('LoG, sigma='+str(sigma),size=20) 
    plb.subplot(4,2,sigma), plb.axis('off'),
    plb.imshow(zero_crossing(result))
    plb.title('LoG with zero-crossing, sigma='+str(sigma),size=20)
plb.show()
# zero-crossing은 닫힌 윤곽을 형성한다.

#########################################
# 실습 13. PIL을 사용하여 에지 찾기와 향상
#######################################
# PIL.ImageFilter.filter
import numpy as np
from PIL import Image
from PIL.ImageFilter import FIND_EDGES, EDGE_ENHANCE, EDGE_ENHANCE_MORE
im6 = Image.open("Goldengate.png")
plb.figure(figsize=(18,15))
plb.subplot(221), plb.imshow(im6), plb.title('Golden Gate')

for i,f in enumerate([EDGE_ENHANCE, EDGE_ENHANCE_MORE]): # FINE_EDGES not working
    tim = im6.filter(f)
    #tim.show() 
    plb.subplot(2,2,i+3), plb.axis('off'), plb.imshow(tim), plb.title(str(f),size=20)
    #tim.title(str(f))
plb.show()    
##############################################################################
