##### Ch2. 샘플링, 푸리에 변환, 컨볼루션
# 샘플링
# 업 샘플링
# 영상의 크기를 늘린다.
# 새로운 큰 영상은 원래의 작은 영상에서 해당화소가 없는 일부 화소를 가진다.
# 다음과 같이 미지의 화속값을 추측한다.
# 보간법 : 최근접 이웃법 양선형, 삼선형
# 최근접 이웃 기반 업 샘플링은 영상 품질을 떨어뜨릴 수 있다.
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

#####################################
## 실습 1. 보간법
#####################################
im = imread("Mountain.png")

im = im[:,:,0]
methods = ['none','nearest','bilinear','bicubic','spline16','lanczos']
fig, axes = plt.subplots(nrows=2,ncols=3, figsize=(15,10)) 
for ax, interp_method in zip(axes.flat, methods):
    ax.imshow(im,interpolation=interp_method, cmap='gray')
    ax.set_title(str(interp_method), size=20)
plt.tight_layout()
plt.show()    

im = im[25:75,80:120]
methods = ['none','nearest','bilinear','bicubic','spline16','lanczos']
fig, axes = plt.subplots(nrows=2,ncols=3, figsize=(15,10)) 
for ax, interp_method in zip(axes.flat, methods):
    ax.imshow(im,interpolation=interp_method, cmap='gray')
    ax.set_title(str(interp_method), size=20)
plt.tight_layout()
plt.show() 
# pixelation(nearest), blocking artifact, zig-zag boundary

# 업샘플링 및 보간법
# 양선형 보간법 Bilinear Interpolation
# P: 보간하고자 하는 점
# Q11, Q12, Q21, Q22 : 네  개의 이웃
# 바이큐빅 보간법
# BiCubic Interpolation
# 큐빅 보간법의 확장
# 부드럽다
# 라그랑지 다항식, 3차 스플라인, 3차 회선 알고리즘
# 4*4 환경에서 3차 스플라인 보간 사용

#####################################
## 실습 2. PIL에서의 보간법
#####################################
from PIL import Image, ImageFont, ImageDraw
im = Image.open("messi.jpg") # (110,110)
imS = im.resize((im.width//2,im.height//2)) # (55,55)
imN = imS.resize((500,500), Image.NEAREST) 
imB = imS.resize((500,500), Image.BILINEAR)
imC = imS.resize((500,500), Image.BICUBIC)

plt.figure(figsize=(10,10))
plt.subplot(221), plt.imshow(imS), plt.title('Original')
plt.subplot(222), plt.imshow(imN), plt.title('Nearest')
plt.subplot(223), plt.imshow(imB), plt.title('Bilinear')
plt.subplot(224), plt.imshow(imC), plt.title('Bicubic')
plt.tight_layout()
plt.show()

#####################################
## 실습 3. 다운 샘플링
#####################################
# 다운 샘플링
# 검은 색 패치/아티팩트 및 패턴 포함
# 앨리어싱 : 샘플링 속도가 Nyquist 속도보다 낮기 때문에 발생한다.
#           앨리어싱을 피하는 방법은 샘플링 속도를 Nyquist 속도이상으로 높이는 것이다.
im = Image.open("messi.jpg")
imS = im.resize((im.width//2,im.height//2))

plt.figure(figsize=(10,10))
plt.subplot(121), plt.imshow(im), plt.title('Original'), 
plt.subplot(122), plt.imshow(imS), plt.title('Downsampling')
plt.show()

# 다운 샘플링 및 안티 앨리어싱
# 앨리어싱 효과 때문에 효과가 좋지 않다.
# 패치성 및 불량 출력
# antialias 고품질의 다운 샘플링 필터
im = Image.open("parrot.png")
im = im.resize((200,250))
ims = im.resize((im.width//3,im.height//3))
ima = im.resize((im.width//3, im.height//3),Image.LANCZOS)

plt.figure(figsize=(10,10)) 
plt.subplot(131), plt.imshow(im), plt.title('Original')
plt.subplot(132), plt.imshow(ims), plt.title('Down')
plt.subplot(133), plt.imshow(ima), plt.title('Down with antialiasing')
plt.show()

im = im.resize((541,811))
im1 = im.copy()
plt.figure(figsize=(10,14))
for i in range(4): 
    plt.subplot(2,2,i+1), plt.imshow(im1, cmap='gray') #, plt.axis('off')
    plt.title('image size = '+str(im1.size)) #[0])+'x'+str(im1.size[1]))
    im1 = im.resize((im.width//(2**(i+1)),im.height//(2**(i+1))))
    #im1 = rescale(im1, scale=0.5, multichannel=True, anti_aliasing=False)
    
plt.subplots_adjust(wspace=0.1,hspace=0.1)
plt.show()

# # 안티알리어싱 을 사용하여 코드변경
# im1 = im.copy()
# plt.figure(figsize=(8,14))
# for i in range(4): 
#     plt.subplot(2,2,i+1), plt.imshow(im1, cmap='gray'), plt.axis('off')
#     plt.title('ANTI:image size = '+str(im1.size)) #[0])+'x'+str(im1.size[1]))
#     im1 = im.resize((im.width//(2**(i+1)),im.height//(2**(i+1))),Image.ANTIALIAS)
# #    im1 = rescale(im1, scale=0.5, multichannel=True, anti_aliasing=False)
    
# plt.subplots_adjust(wspace=0.1,hspace=0.1)
# plt.show()


## 양자화
# 영상의 밝기와 관련있음
# 화소당 사용되는 비트수
# 일반적으로 256 명암도
# 화소저장을 위한 비트 수가 감소함에 따라 양자화 에러가 증가
# 인위적인 경계 또는 윤곽 및 모자이크화 
# 영상 품질 저하

#####################################################
## 실습 4. PIL 로 양자화
# P 모드와 색상 인수를 가능한 색상의 최대 개수로 사용
# 색상 양자화를 위한 PIL Image 모듈의 convert() 함수
##################################################
im = Image.open('parrot.png')
plt.figure(figsize=(30,20))
plt.subplot(331), plt.imshow(im), plt.title('Original'), plt.axis('off')
num_colors_list = [2**n for n in range(7,-1,-1)] # 1<<n = 2**n

i = 1
for num_colors in num_colors_list:
    im1 = im.convert('P', palette=Image.ADAPTIVE, colors=num_colors)
    plt.subplot(3,3,i+1), plt.imshow(im1), plt.axis('off')
    title = str(num_colors) 
    plt.title('Image with # colors = '+title, size=20)
    i += 1
plt.subplots_adjust(wspace=0.2, hspace=0.2)
plt.show()

##### Ch2. 이산푸리에 변환 p65
# 푸리에 변환은 긴 수학적 역사를 가지고 있음
# 2D 이산 푸리에 변환 만 다룸
# 영상이 2차원을 따라 사인과 코사인의 가중합으로 표현될 수 있는 2D 함수로 생각할 수 있다.
# 영상의 명암도 화솟값 집합(공간/시간 영역) == DFT ==> 푸리에 계수 집합(주파수 영역)
# 공간 변수와 변환 변수는 일련의 이산 연속 정수 값(영상을 나타내는 2D 배열)을 취할 수 있으므로 불연속
# 주파수 영역 2D 배열 == IDFT ==> 공간 영역
# DFT:  F[u,v] = 1/MN \sum_{x=0}^{M-1}\sum_{y=0}^{N-1} f[x,y]exp(-2pi i(ux/M + vy/N))
# IDFT: f[x,y] = \sum_{u=0}^{M-1}\sum_{v=0}^{N-1} F[u,v]exp(2pi i(ux/M + vy/N))

# DFT가 필요한 이유는 무엇인가?
# 주파수 영역으로 변환하면 영상을 더 잘 이해할 수 있다.
# 저주파는 영상의 평균 총 정보 레벨에 해당
# 높은 주파수는 에지, 잡음 및 보다 자세한 정보
# 대부분의 영상은 소수의 DFT 계수들을 사용하여 표현
# 푸리에 희소 영상 Fourier-sparse image
# JPEG 영상 압축 알고리즘: DCT: Discrete Cosine Transform

# DFT를 계산하는 고속 푸리에 변환 알고리즘
# DFT O(N^2), FFT O(NlogN)
# numpy와 scipy 라이브러리는 모두 FFT 알고리즘 사용

###################################################
## 실습 6. scipy.fftpack 모듈을 사용한 FFT
#import numpy as np
#from PIL import Image
################################################### 
im = np.array(Image.open('Rhino.jpg').convert('L'))
def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis,ddof=ddof)
    return np.where(sd==0, 0, m/sd)
snr = signaltonoise(im,axis=None)
print('SNR for the original image = '+str(snr))
# SNR for the original image = 2.075630009048946

from scipy import fftpack
freq = fftpack.fft2(im)
im1 = fftpack.ifft2(freq).real
snr = signaltonoise(im1,axis=None)
print('SNR for the image obtained after reconstruction = '+str(snr))
#SNR for the image obtained after reconstruction = 2.075630009048946
np.allclose(im,im1) # 두 영상이 거의 같으면 True

# 주파수 스펙트럼 그리기
# 푸리에 계수는 복소수이므로 크기를 직접 볼 수 있다.
# 푸리에 변환의 크기를 표시하는 것을 변환 스펙트럼 이라고 한다.
# F(0,0) : DC 계수
# 변환 계수는 DC 성분이 중심에 있도록 fftshift()로 이동된다.
freq2 = fftpack.fftshift(freq)
am = np.log(np.max(abs(freq2))); cons = 255/np.log(1+am)
np.min(np.abs(freq2))
plt.figure(figsize=(20,10))
plt.subplot(221),plt.imshow(im,cmap='gray'), plt.axis('off'), plt.title('Original')
plt.subplot(222),plt.imshow(im1,cmap='gray'), plt.axis('off'), plt.title('Image after recon')
plt.subplot(223),plt.imshow(freq2.astype(int), cmap='gray'), plt.title('Frequency')
plt.subplot(224),plt.imshow((cons*np.log(1+freq2)).astype(int), cmap='gray'), plt.title('Frequency log scale')
plt.show()

###################################################
## 실습 7. numpy.fft 모듈을 사용한 FFT
####################################################
import numpy as np
import numpy.fft as fp
from skimage.io import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
im1 = rgb2gray(imread('Rhino.jpg'))
im1 = im1[:183,:145]  #(183,275)
freq1 = fp.fft2(im1); am = np.max(abs(freq1)); cons = 255/np.log(1+am)
im1_ = fp.ifft2(freq1).real
mag  = cons*np.log(1+np.abs(fp.fftshift(freq1)))

plt.figure(figsize=(12,10))
plt.subplot(221), plt.imshow(im1,cmap='gray'), plt.title('Original')
plt.subplot(223), plt.imshow(mag,cmap='gray'), plt.title('FFT spectrum magnitude')
plt.subplot(224), plt.imshow(np.angle(fp.fftshift(freq1)),cmap='gray'), plt.title('FFT Phase')
plt.subplot(222), plt.imshow(np.clip(im1_,0,255),cmap='gray'), plt.title('Reconstructed')
plt.show()

im2 = rgb2gray(imread('Mandrill.jpg'))
im2 = im2[:183,:145]
freq2 = fp.fft2(im2); am = np.max(abs(freq2)); cons = 255/np.log(1+am)
im2_ = fp.ifft2(freq2).real
mag  = cons*np.log(1+np.abs(fp.fftshift(freq1)))

plt.figure(figsize=(12,10))
plt.subplot(221), plt.imshow(im2,cmap='gray'), plt.title('Original')
plt.subplot(223), plt.imshow(mag,cmap='gray'), plt.title('FFT spectrum magnitude')
plt.subplot(224), plt.imshow(np.angle(fp.fftshift(freq2)),cmap='gray'), plt.title('FFT Phase')
plt.subplot(222), plt.imshow(np.clip(im2_,0,255),cmap='gray'), plt.title('Reconstructed')
plt.show()

#########################################
## 실습 8. FFT의 위상 정보
# |F(u,v)|는 일반적으로 공간 주파수가 높을수록 감소하고, 
# FFT 위상은 정보가 상대적으로 없는 것처럼 보인다.
# 비록 크기만큼 유익하진 않지만 DFT위상은 중요한 정보이기 때문에
# 위상을 사용할 수 없거나 다른 위상 배열을 사용하면 영상을 제대로 재구성할 수 없다.
# 한 영상의 실수부와 다른 영상의 허수부를 섞으면...
################################################
im1_ = fp.ifft2(np.vectorize(complex)(freq1.real, freq2.imag)).real
im2_ = fp.ifft2(np.vectorize(complex)(freq2.real, freq1.imag)).real

plt.figure(figsize=(20,15))
plt.subplot(211), plt.imshow(np.clip(im1_,0,255),cmap='gray')
plt.title('Reconstructed Re(F1)+Im(F2)')
plt.subplot(212), plt.imshow(np.clip(im2_,0,255),cmap='gray')
plt.title('Reconstructed Re(F2)+Im(F1)')
plt.show()

##### Ch2. 합성곱
## 컨볼루션(회선) 이해
# 마스크, 필터, 커널
# 컨볼루션 필터링은 영상의 공간 주파수 특성을 수정하는 데 사용
# 화소의 새로운 값을 계산하기 위해 이웃하는 모든화소의 가중치를 더해 중앙화소의 값 결정

# 영상을 왜 컨볼루션하나?
# 평활화, 선명화, 엠보싱, 에지 검출

# scipy.signal.convolv2d와 컨볼루션
# 명암도 영상에 컨볼루션 적용
# 라플라스 커널과 컨볼루션을 사용하여 명암도 cameraman.jpg 영상에서 에지를 먼저 검출
# 박스 커널을 사용하여 흐리게 만들자.
# import numpy as np
#################################################
## 실습 9 합성곱
################################################
import numpy as np
from skimage.io import imread, imshow, show
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
im = rgb2gray(imread('cameraman.jpg'))
print(np.max(im)) #1.0
print(im.shape) #(512,512)
#imshow(im)

from scipy import signal#, ndimage
blur_box_kernel = np.ones((3,3))/9
edge_laplace_kernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
im_blurred = signal.convolve2d(im,blur_box_kernel)
im_edges = np.clip(5*signal.convolve2d(im,edge_laplace_kernel),0,1)

plt.figure(figsize=(8,6))
plt.subplot(131), plt.imshow(im,cmap='gray'), plt.axis('off'), plt.title('Original')
plt.subplot(132), plt.imshow(im_blurred,cmap='gray'), plt.axis('off'), plt.title('Blurred')
plt.subplot(133), plt.imshow(im_edges,cmap='gray'), plt.axis('off'), plt.title('Edges')
plt.show()

#####################################################
# 실습 10. 컨볼루션 모드 패드 값 및 경계 조건
# 에지 화소를 수행할 작업에 따라 mode, boundary, fillvalue 의 세가지 인수가 있다.
# mode = 'full' : 출력.shape = 입력.shape + [1,1]
# mode = 'valid" : 출력.shape = 입력.shape + [-1,-1]
# mode = 'same' : 출력.shape = 입력.shape, 'full'출력에 관해서 중심에 있음
########################################################################
ims = im[0:5,0:5]
plt.figure(figsize=(8,7)); print(im[0,0])
plt.subplot(221), plt.imshow(ims,cmap='gray'), plt.title('Original 5*5')
imb = signal.convolve2d(ims,blur_box_kernel,mode='full'); print(imb[0,0])
plt.subplot(222), plt.imshow(imb,cmap='gray'), plt.title('Full 7*7')
imb = signal.convolve2d(ims,blur_box_kernel,mode='valid'); print(imb[0,0])
plt.subplot(223), plt.imshow(imb,cmap='gray'), plt.title('Valid 3*3')
imb = signal.convolve2d(ims,blur_box_kernel,mode='same'); print(imb[0,0])
plt.subplot(224), plt.imshow(imb,cmap='gray'), plt.title('Same 5*5')
plt.show()

###################################################################
# 실습 11. 컬러 영상에 컨볼루션 적용
# embos 커널과 schar 에지 검출 복소수 커널 사용
##################################################################
im = imread('tajmahal.jpg')/255
print(np.max(im))
print(im.shape)   #133,162,3
emboss_kernel = np.array([[-2,-1,0],[-1,1,1],[0,1,2]])
edge_kernel = np.array([[-3-3j,0-10j,+3-3j],[-10+0j,0+0j,+10+0j],[-3+3j,0+10j,+3+3j]])
im_embossed = np.ones(im.shape)
im_edges    = np.ones(im.shape)

for i in range(3):
    conv = signal.convolve2d(im[...,i], emboss_kernel, mode='same', boundary='symm')
    im_embossed[...,i] = np.clip(conv,0,1)
    conv = signal.convolve2d(im[...,i], edge_kernel, mode='same', boundary='symm')
    im_edges[...,i] = np.clip(conv,0,1)

fig, axes = plt.subplots(ncols=3,figsize=(10,18))
axes[0].imshow(im), axes[0].set_title('Original')
axes[1].imshow(im_embossed), axes[1].set_title('Embossed')
axes[2].imshow(im_edges), axes[2].set_title('Schar Edge Detection')
for ax in axes:
    ax.axis('off')
plt.show()

fig, axes = plt.subplots(ncols=3,figsize=(10,18))
axes[0].imshow(im[:,:,0],cmap='gray'), axes[0].set_title('Original')
axes[1].imshow(im_embossed[:,:,0],cmap='gray'), axes[1].set_title('Embossed')
axes[2].imshow(im_edges[:,:,0],cmap='gray'), axes[2].set_title('Schar Edge Detection')
for ax in axes:
    ax.axis('off')
plt.show()

####################################################
# 실습 12.SciPy ndimage.convolve를 이용한 컨볼루션
########################################################
from skimage.io import imread, imshow, show
from scipy import ndimage
import numpy as np
im = imread('victoria_memorial.png').astype(float)
sharpen_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]).reshape((3,3,1))
emboss_kernel = np.array([[-2,-1,0],[-1,1,1],[0,1,2]]).reshape((3,3,1))
im_sharp = ndimage.convolve(im, sharpen_kernel, mode='nearest')
im_sharp = np.clip(im_sharp,0,255).astype(np.uint8)
im_emboss = ndimage.convolve(im,emboss_kernel, mode='nearest')
im_emboss = np.clip(im_emboss,0,255).astype(np.uint8)

plt.figure(figsize=(10,15))
plt.subplot(311), plt.imshow(im.astype(np.uint8)), plt.axis('off'), plt.title('Ori',fontsize=30)
plt.subplot(312), plt.imshow(im_sharp), plt.axis('off'), plt.title('Sharpend',fontsize=30)
plt.subplot(313), plt.imshow(im_emboss), plt.axis('off'), plt.title('Embossed',fontsize=30)
plt.tight_layout(), plt.show()

##################################################
# 실습 13. 상관관계 vs. 컨볼루션
# 영상과 템플릿 간의 교차 상관관계로 템플릿 매칭
###################################################
from scipy import misc, signal
face_image = misc.face(gray=True) - misc.face(gray=True).mean()
#template_image = np.copy(face_image[300:365,670:750])  # 오른쪽 눈 영역 복사
template_image = np.copy(face_image[280:500,550:750])  
template_image -= template_image.mean()
face_image = face_image + np.random.randn(*face_image.shape)*50 # 랜덤 노이즈

correlation = signal.correlate2d(face_image,template_image,boundary='symm',mode='same')
y,x = np.unravel_index(np.argmax(correlation),correlation.shape)

plt.figure(figsize=(10,15)) 
plt.subplot(311), plt.imshow(face_image,cmap='gray') #, plt.axis("off")
plt.subplot(312), plt.imshow(template_image,cmap='gray') #, plt.axis("off")
plt.subplot(313), plt.imshow(correlation,cmap='gray') #, plt.colorbar() #, plt.axis('off')
plt.show()

(fig, ax) = plt.subplots(nrows=3,ncols=1,figsize=(10,10))
ax[0].imshow(face_image,cmap='gray'), ax[0].set_title('Ori') #, ax[0].axis('off')
ax[1].imshow(template_image,cmap='gray'), ax[1].set_title('Template')
ax[2].imshow(correlation, cmap='gray'), ax[2].set_title('Correlation')

















