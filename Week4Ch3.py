#### Ch.3.Convolution and Frequency domin Filtering
####       컨볼루션과 주파수 영역 필터링
## 1. 컨볼루션 정리 및 주파수 영역 가우시안 흐림
## 2. 주파수 영역 필터링(scipy ndimage 모듈 및 scikit-image 사용)
## 컨볼루션 정리 f(x,y)*h(x,y)=F(u,v)H(u,v)
## f,h  커널, 필터, 마스크, 윈도우, 열화 함수. 향상 함수 

############################################
## 실습 1. 가우시안 필터 합성곱과 주파수 영역 변환
############################################
import numpy as np
import numpy.fft as fp
import matplotlib.pyplot as plt
# from scipy import signal
from skimage.io import imread
from scipy.signal.windows import gaussian
im2 = imread('Lenna.png') #RGBA, (220,220,4)
im = np.mean(im2, axis=2) # (220,220)

freq = fp.fft2(im)
gauss_kernel = np.outer(gaussian(im.shape[0], 5), gaussian(im.shape[1], 5))
# gauss_kernel = np.outer(signal.gaussian(im.shape[0],5), signal.gaussian(im.shape[1],5)) 
# outer product  ou = np.outer(a,b) : ou[i,j]=a[i]*b[j]
assert(freq.shape == gauss_kernel.shape) # shape이 같은 지 검사, False 면 stop
freq_kernel = fp.fft2(fp.ifftshift(gauss_kernel)) #fp.fftshift([1,2,3])=[3,1,2], fp.ifftshift([1,2,3])=[2,3,1]
convolved = freq * freq_kernel
iml = fp.ifft2(convolved).real

mag1 = 20*np.log10(1+fp.fftshift(freq))
mag2 = 20*np.log10(1+fp.fftshift(freq_kernel))
mag3 = 20*np.log10(1+fp.fftshift(convolved))

plt.figure(figsize=(20,15)), plt.gray()
plt.subplot(231), plt.imshow(im),               plt.title('Original',size=20),          plt.axis('off')
plt.subplot(232), plt.imshow(gauss_kernel),     plt.title('Gaussian Kernel',size=20),   plt.axis('off')
plt.subplot(233), plt.imshow(iml),              plt.title('Output image',size=20),      plt.axis('off')
plt.subplot(234), plt.imshow(mag1.astype(int)), plt.title('Original Spectrum',size=20), plt.axis('off')
plt.subplot(235), plt.imshow(mag2.astype(int)), plt.title('Gaussian Spectrum',size=20), plt.axis('off')
plt.subplot(236), plt.imshow(mag3.astype(int)), plt.title('Output Spectrum',size=20),   plt.axis('off')
plt.subplots_adjust(wspace=0.2, hspace=0)
plt.show()

#############################################
## 실습 2. 주파수 영역에서의 가우시안 LPF 
#############################################
import numpy as np
import numpy.fft as fp
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.signal.windows import gaussian

gauss_kernel4 = np.outer(gaussian(im.shape[0],1), gaussian(im.shape[1],1)) # 위 sigma 5, vs. sigma 1
freq_kernel4 = fp.fft2(fp.ifftshift(gauss_kernel4))
mag4 = 20*np.log10(1+fp.fftshift(freq_kernel4))
convolved4 = freq * freq_kernel4
iml4 = fp.ifft2(convolved4).real

gauss_kernel5 = np.outer(gaussian(im.shape[0],3), gaussian(im.shape[1],3)) # 위 sigma 5, vs. sigma 1
freq_kernel5 = fp.fft2(fp.ifftshift(gauss_kernel5))
mag5 = 20*np.log10(1+fp.fftshift(freq_kernel5))
convolved5 = freq * freq_kernel5
iml5 = fp.ifft2(convolved5).real

plt.figure(figsize=(20,15))
plt.subplot(241), plt.imshow(mag2.astype(int), cmap='coolwarm')
plt.title('sigma=5',size=20), plt.axis('off'), plt.clim(0, 30), plt.colorbar() # plt.clim(0,30), plt.clim(0,40)
plt.subplot(242), plt.imshow(mag5.astype(int), cmap='coolwarm')
plt.title('sigma=3',size=20), plt.axis('off'), plt.clim(0, 30), plt.colorbar()
plt.subplot(243), plt.imshow(mag4.astype(int), cmap='coolwarm')
plt.title('sigma=1',size=20), plt.axis('off'), plt.clim(0, 30), plt.colorbar()
plt.subplot(245), plt.imshow(iml)
plt.title('Image,sigma=5',size=20), plt.axis('off') 
plt.subplot(246), plt.imshow(iml5)
plt.title('Image,sigma=3',size=20), plt.axis('off')
plt.subplot(247), plt.imshow(iml4)
plt.title('Image,sigma=1',size=20), plt.axis('off')
plt.subplot(248), plt.imshow(im)
plt.title('Original Image',size=20), plt.axis('off')
plt.show()

## 3D, p95 : HW

#########################################################
# 실습 3. scipy signal.fftconvolve를 사용한 가우시안 필터
########################################################
from scipy import signal
im = np.mean(imread('Mandrill2.jpg'), axis=2)
print(im.shape) #(400,400)
gauss_kernel = np.outer(gaussian(11,5),gaussian(11,5))
imb = signal.fftconvolve(im,gauss_kernel, mode='same')

fig, ax = plt.subplots(1,3,figsize=(20,8))
ax[0].imshow(im, cmap='gray'), ax[0].set_title('Original',size=20),        ax[0].set_axis_off()
ax[1].imshow(gauss_kernel),    ax[1].set_title('Gaussian Kernel',size=20), ax[1].set_axis_off()
ax[2].imshow(imb, cmap='gray'),ax[2].set_title('Blurred',size=20),         ax[2].set_axis_off()
fig.show()

from scipy import fftpack
F1 = fftpack.fft2(im.astype(float))
F2 = fftpack.fftshift(F1)
mag1 = (20*np.log10(1+F2)).astype(int)

F1 = fftpack.fft2(imb.astype(float))
F2 = fftpack.fftshift(F1)
mag2 = (20*np.log10(1+F2)).astype(int)

plt.figure(figsize=(15,8))
plt.subplot(121), plt.imshow(mag1,cmap='gray'), plt.title('Original Spectrum',size=20)
plt.subplot(122), plt.imshow(mag2,cmap='gray'), plt.title('Blur Spectrum',size=20)
plt.show()

##########################################
## 실습 4. 시간 비교
###########################################
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from skimage.io import imread   
from scipy.signal.windows import gaussian

im = np.mean(imread('Mandrill2.jpg'), axis=2)
gauss_kernel = np.outer(gaussian(50,5),gaussian(50,5))
imb1 = signal.convolve(im,gauss_kernel, mode='same')
imb2 = signal.fftconvolve(im,gauss_kernel, mode='same')

plt.figure(figsize=(15,8))
plt.subplot(131), plt.imshow(im,cmap='gray'),   plt.title('Original',size=20)
plt.subplot(132), plt.imshow(imb1,cmap='gray'), plt.title('Convolve',size=20)
plt.subplot(133), plt.imshow(imb2,cmap='gray'), plt.title('FFTconvolve',size=20)
plt.show()

import timeit
def wrapper_convolve(func):
    def wrapped_convolve():
        return func(im, gauss_kernel, mode="same")
    return wrapped_convolve

wrapped_convolve = wrapper_convolve(signal.convolve)
wrapped_fftconvolve = wrapper_convolve(signal.fftconvolve)
times1 = timeit.repeat(wrapped_convolve, number=1, repeat=100)
times2 = timeit.repeat(wrapped_fftconvolve, number=1, repeat=100)
data = [times1, times2]

plt.figure(figsize=(8,6))
box = plt.boxplot(data, patch_artist=True)
colors = ['cyan','pink']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    
plt.xticks(np.arange(3), ("",'convolve','fftconvolve'), size=15)
plt.yticks(fontsize=15)
plt.xlabel('scipy.signal convolution methods', size=15)
plt.ylabel('time taken to run',size=15)
plt.show()

## 필터의 효과
## 대비(Contrast)를 향상(Enhancement)
## 잡음 제거(Smoothing)
## 알려진 패턴을 감지(Template Matching)

# 고역통과 필터 HPF, Highpass filter
# 고주파 성분 edge, detail, noise
# HPF 구현
# 1. scipy.fftpack fft2로 2D FFT를 수행하고 영상의 주파수 영역표현 구함
# 2. 고주파 성분 유지: 저주파 성분 제거
# 3. 역 FFT를 수행하여 영상을 재구성

######################################################
## 실습 5. High Pass Filtering
######################################################
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numpy.fft as fp
from scipy import fftpack

im = np.array(Image.open('Rhino.jpg').convert('L')) 
plt.figure(figsize=(10,10))
plt.imshow(im,cmap='gray'), plt.axis('off')
plt.show()

freq = fp.fft2(im)
freq1 = np.copy(freq)
freq2 = fp.fftshift(freq1)
mag = (20*np.log10(1+freq2)).astype(int)
plt.figure(figsize=(10,10))
plt.imshow(mag,cmap='gray')
plt.show()

(w,h)=freq.shape
half_w, half_h = np.int16(w/2), np.int16(h/2)
## 20*20 크기 저주파 영역 계수값 제거
freq2[half_w-10:half_w+11, half_h-10:half_h+11] = 0
HPF = (20*np.log10(1+freq2)).astype(int)
plt.figure(figsize=(10,10))
plt.imshow(HPF,cmap='gray')
plt.show()

im1 = fp.ifft2(fftpack.ifftshift(freq2))
im2 = np.clip(im1.real,0,255)
plt.figure(figsize=(10,10))
plt.imshow(im2,cmap='gray'), plt.axis('off')
plt.show()

plt.figure(figsize=(10,10))
plt.subplot(221), plt.imshow(im,cmap='gray')
plt.axis('off'), plt.title('Original Image')
plt.subplot(222), plt.imshow(mag,cmap='gray')
plt.axis('off'), plt.title('Original Frequency')
plt.subplot(223), plt.imshow(im2,cmap='gray')
plt.axis('off'), plt.title('High Pass Image')
plt.subplot(224), plt.imshow(HPF,cmap='gray')
plt.axis('off'), plt.title('High Pass Frequency')
plt.subplots_adjust(wspace=0.1, hspace=0)
plt.tight_layout()
plt.show()

#############################
## 실습 6. Cameramen HPF
#############################
import numpy.fft as fp
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy import fftpack

im = np.array(Image.open("cameraman.jpg").convert('L'))
freq = fp.fft2(im)
(w,h) = freq.shape
half_w, half_h = w//2, h//2
snrs_hp = []
lbs = list(range(24))

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis,ddof=ddof)
    return np.where(sd==0, 0, m/sd)

plt.figure(figsize=(20,20))
for i in lbs:
    freq1 = np.copy(freq)
    freq2 = fftpack.fftshift(freq1)
    freq2[half_w-i:half_w+i, half_h-i:half_h+i] = 0
    mag = (20*np.log10(1+freq2)).astype(int)
    iml = np.clip(fp.ifft2(fftpack.ifftshift(freq2)).real,0,255)
    snrs_hp.append(signaltonoise(iml,axis=None))
    
    plt.subplot(6,8,2*i+1), plt.imshow(iml, cmap='gray'), plt.axis('off')
    plt.title('F= '+str(i+1), size=20)
    plt.subplot(6,8,2*i+2), plt.imshow(mag, cmap='gray'), plt.axis('off')
    #plt.title('Frequency with F= '+str(i+1), size=20)
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
plt.show()

plt.plot(lbs, snrs_hp, 'b.-')
plt.xlabel('Cufoff Frequency for HPF',size=20)
plt.ylabel('SNR',size=20)
plt.axis([-1, 25, 0.4, 0.6])
plt.grid()
plt.show() 

##################
## 실습 7. LPF  
##################
from scipy import ndimage 
im = np.mean(imread('Lenna.png'), axis=2)
freq = fp.fft2(im)
freq_gaussian = ndimage.fourier_gaussian(freq, sigma=4)
mag0= 20*np.log10(1+fp.fftshift(freq)).real
mag = 20*np.log10(1+fp.fftshift(freq_gaussian)).real
iml = fp.ifft2(freq_gaussian)

plt.figure(figsize=(12,10))
plt.subplot(221), plt.imshow(im,cmap='gray'), plt.axis('off')
plt.subplot(222), plt.imshow(iml.real,cmap='gray'), plt.axis('off')
plt.subplot(223), plt.imshow(mag0,cmap='gray'), plt.axis('off')
plt.subplot(224), plt.imshow(mag,cmap='gray'), plt.axis('off')
plt.show()

# another plotting
fig,ax=plt.subplots(2,2,figsize=(12,10))
#plt.gray()
ax[0][0].imshow(im,cmap='gray'),       ax[0][0].axis('off')
ax[0][1].imshow(iml.real,cmap='gray'), ax[0][1].axis('off')
ax[1][0].imshow(np.int32(mag0)),     ax[1][0].axis('off')
ax[1][1].imshow(np.int32(mag)),      ax[1][1].axis('off')
plt.show()

#############################################
## 실습 8. 코뿔소
############################################
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numpy.fft as fp
from scipy import fftpack

im = np.array(Image.open('Rhino.jpg').convert('L')) 
freq = fp.fft2(im)
freq1 = np.copy(freq)
freq2 = fp.fftshift(freq1)
freq2h = np.copy(freq2)

(w,h)=freq.shape
half_w, half_h = np.int16(w/2), np.int16(h/2)
## 20*20 크기 저주파 영역 계수값 제거
freq2h[half_w-10:half_w+11, half_h-10:half_h+11] = 0
freq2l = freq2 - freq2h
mag = (20*np.log10(1+freq2)).astype(int)
HPF = (20*np.log10(1+freq2h)).astype(int)
LPF = (20*np.log10(1+freq2l)).astype(int)

iml = fp.ifft2(fftpack.ifftshift(freq2l))
iml = np.clip(iml.real,0,255)
imh = fp.ifft2(fftpack.ifftshift(freq2h))
imh = np.clip(imh.real,0,255)

plt.figure(figsize=(10,10))
plt.subplot(321), plt.imshow(im,cmap='gray')
plt.axis('off'), plt.title('Original Image')
plt.subplot(322), plt.imshow(mag,cmap='gray')
plt.axis('off'), plt.title('Original Frequency')
plt.subplot(323), plt.imshow(iml,cmap='gray')
plt.axis('off'), plt.title('Low Pass Image')
plt.subplot(324), plt.imshow(LPF,cmap='gray')
plt.axis('off'), plt.title('Low Pass Frequency')
plt.subplot(325), plt.imshow(imh,cmap='gray')
plt.axis('off'), plt.title('High Pass Image')
plt.subplot(326), plt.imshow(HPF,cmap='gray')
plt.axis('off'), plt.title('High Pass Frequency')
plt.subplots_adjust(wspace=0.1, hspace=0)
plt.tight_layout()
plt.show()

#########################
## 실습 9. LPF Cameramen
#########################
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numpy.fft as fp
from scipy import fftpack

im = np.array(Image.open("cameraman.jpg").convert('L'))
freq = fp.fft2(im)
(w,h) = freq.shape
half_w, half_h = w//2, h//2
snrs_hp = []
lbs = list(range(24))

plt.figure(figsize=(20,20))
for i in lbs:
    freq1 = np.copy(freq)
    freq2 = fftpack.fftshift(freq1)
    freqh = np.copy(freq2)
    freqh[half_w-i:half_w+i, half_h-i:half_h+i] = 0
    freql = freq2 - freqh
    mag = (20*np.log10(1+freql)).astype(int)
    iml = np.clip(fp.ifft2(fftpack.ifftshift(freql)).real,0,255)
    snrs_hp.append(signaltonoise(iml,axis=None))
    
    plt.subplot(6,8,2*i+1), plt.imshow(iml, cmap='gray'), plt.axis('off')
    plt.title('F= '+str(i+1), size=20)
    plt.subplot(6,8,2*i+2), plt.imshow(mag, cmap='gray'), plt.axis('off')
    #plt.title('Frequency with F= '+str(i+1), size=20)
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
plt.show()

plt.plot(lbs, snrs_hp, 'b.-')
plt.xlabel('Cufoff Frequency for LPF',size=20)
plt.ylabel('SNR',size=20)
plt.axis([-1, 25, 2, 3])
plt.grid()
plt.show() 

########################################################
## 실습 10. DoG(Difference of Gaussian)를 사용한 대역통과 필터 BPF
###########################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from skimage.io import imread
from skimage import img_as_float

im = img_as_float(imread('Dinosaur.jpg'))
x = np.linspace(-10,10,15)
kernel_1d = np.exp(-0.5*x**2)
kernel_1d /= np.trapz(kernel_1d)
gauss_kernel1 = kernel_1d[:,np.newaxis]* kernel_1d[np.newaxis,:]

kernel_1d = np.exp(-5*x**2)
kernel_1d /= np.trapz(kernel_1d)
gauss_kernel2 = kernel_1d[:,np.newaxis]* kernel_1d[np.newaxis,:]

DoGKernel = gauss_kernel1[:,:,np.newaxis] - gauss_kernel2[:,:,np.newaxis]
im1 = signal.fftconvolve(im, DoGKernel, mode="same")
print(np.max(im1))

ma = np.max(im1); mi = np.min(im1); im1 = (im1 - mi)/(ma-mi)
plt.figure(figsize=(12,10))
plt.subplot(331), plt.imshow(im), plt.axis('off'), plt.title('Original')
plt.subplot(332), plt.imshow(DoGKernel[:,:,0],cmap='gray'), plt.axis('off'), plt.title('DOG')
plt.subplot(333), plt.imshow(1-im1), plt.axis('off'), plt.title('Recon[0 1]')
ratio = 0.3
mm = ma*ratio + mi*(1-ratio); imm = (im1 - mm)/(ma-mm)
plt.subplot(334), plt.imshow(1-imm), plt.axis('off'), plt.title('[0.3 1]')
ratio = 0.4
mm = ma*ratio + mi*(1-ratio); imm = (im1 - mm)/(ma-mm)
plt.subplot(335), plt.imshow(1-imm), plt.axis('off'), plt.title('[0.4 1]')
ratio = 0.5
mm = ma*ratio + mi*(1-ratio); imm = (im1 - mm)/(ma-mm)
plt.subplot(336), plt.imshow(1-imm), plt.axis('off'), plt.title('[0.5 1]')
ratio = 0.6
mm = ma*ratio + mi*(1-ratio); imm = (im1 - mm)/(ma-mm)
plt.subplot(337), plt.imshow(1-imm), plt.axis('off'), plt.title('[0.6 1]')
ratio = 0.65
mm = ma*ratio + mi*(1-ratio); imm = (im1 - mm)/(ma-mm)
plt.subplot(338), plt.imshow(1-imm), plt.axis('off'), plt.title('[0.65 1]')
ratio = 0.8
mm = ma*ratio + mi*(1-ratio); imm = (im1 - mm)/(ma-mm)
plt.subplot(339), plt.imshow(1-imm), plt.axis('off'), plt.title('[0.8 1]')
# ratio = 0.9
# mm = ma*ratio + mi*(1-ratio); imm = (im1 - mm)/(ma-mm)
# plt.subplot(339), plt.imshow(1-imm), plt.axis('off'), plt.title('[0.9 1]')
#  #1-10*np.clip(im1,0,1)), plt.axis('off')
plt.show()

# kernel = np.ones([15,15])
# kernel[3:12,3:12]=1; kernel[7:8,7:8]=0; kernel=kernel[:,:,np.newaxis]
# im1 = signal.fftconvolve(im, kernel, mode="same")

# plt.figure(figsize=(20,10))
# plt.subplot(131), plt.imshow(im), plt.axis('off')
# plt.subplot(132), plt.imshow(10*np.clip(im1,0,1)), plt.axis('off')
# plt.subplot(133), plt.imshow(kernel[:,:,0],cmap='gray'), plt.axis('off')
# plt.show()

######################################
## 실습 11. 대역차단 필터 : Notch filter
######################################
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack   
from skimage.io import imread

im = np.mean(imread("parrot.png"),axis=2)/255
iml = np.copy(im)
for n in range(im.shape[1]):
    iml[:,n] += np.cos(0.1*np.pi*n)
    
F1 = fftpack.fft2(im.astype(float))
F2 = fftpack.fftshift(F1)
F3 = fftpack.fft2(iml.astype(float))
F4 = fftpack.fftshift(F3)  

mag1 = (20*np.log10(1+F2)).astype(int)
mag2 = (20*np.log10(1+F4)).astype(int)

plt.figure(figsize=(12,10))
plt.subplot(321), plt.imshow(im,cmap='gray'), plt.axis('off')
plt.title('Original Image')  
plt.subplot(322), plt.imshow(mag1,cmap='gray'), plt.axis('off')
#plt.xticks(np.arange(0,im.shape[1],25)), plt.yticks(np.arange(0,im.shape[0],25))
plt.title('Original Image Spectrum')
plt.subplot(323), plt.imshow(iml,cmap='gray'), plt.axis('off')
plt.title('Image after adding Sinusoidal Noise')
plt.subplot(324), plt.imshow(mag2,cmap='gray'), plt.axis('off')
#plt.xticks(np.arange(0,im.shape[1],25)), plt.yticks(np.arange(0,im.shape[0],25))
plt.title('Noisy Image Spectrum')
plt.subplot(326), plt.imshow(mag2-mag1,cmap='gray'), plt.axis('off')
#plt.xticks(np.arange(0,im.shape[1],25)), plt.yticks(np.arange(0,im.shape[0],25))
plt.title('Spectrum Difference')
plt.tight_layout()
plt.show()

##########################
## 실습 12. 설계.....
##########################
F5 = np.copy(F4); F6 = np.copy(F4); F7 = np.copy(F4)
h,w = F5.shape

plt.figure(figsize=(15,10))
plt.gray()
plt.subplot(241), plt.imshow(iml), plt.axis('off'), plt.title("Blurred")
plt.subplot(242), plt.imshow(mag2), plt.axis('off')
F5[h//2-1:h//2+1,:w//2] = F5[h//2-1:h//2+1,w//2+1:] = 0
iml = fftpack.ifft2(fftpack.ifftshift(F5)).real
mag2 = (20*np.log10(1+F5)).astype(int)
plt.subplot(243), plt.imshow(iml), plt.axis('off'), plt.title('width 1')
plt.subplot(244), plt.imshow(mag2[h//2-20:h//2+20,w//2-20:w//2+20]), plt.axis('off')
F6[h//2-3:h//2+3,:w//2-2] = F6[h//2-3:h//2+3,w//2+3:] = 0
iml = fftpack.ifft2(fftpack.ifftshift(F6)).real
mag2 = (20*np.log10(1+F6)).astype(int)
plt.subplot(245), plt.imshow(iml), plt.axis('off'), plt.title('width 3')
plt.subplot(246), plt.imshow(mag2[h//2-20:h//2+20,w//2-20:w//2+20]), plt.axis('off')
F7[h//2-5:h//2+5,:w//2-4] = F7[h//2-5:h//2+5,w//2+5:] = 0
iml = fftpack.ifft2(fftpack.ifftshift(F7)).real
mag2 = (20*np.log10(1+F7)).astype(int)
plt.subplot(247), plt.imshow(iml), plt.axis('off'), plt.title('width 5')
plt.subplot(248), plt.imshow(mag2[h//2-20:h//2+20,w//2-20:w//2+20]), plt.axis('off')
plt.tight_layout()
plt.show()

##########################################
## 실습 13. 영상 복원 => 화질저하 Degradation
## Deconvolution : 역 필터 및 Wiener 필터
###########################################
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy import fftpack as fp
from scipy.signal.windows import gaussian

im = np.mean(imread('Lenna.png'),axis=2)
gauss_kernel = np.outer(gaussian(im.shape[0],10), gaussian(im.shape[1],10))
freq = fp.fft2(im)
freq_kernel = fp.fft2(fp.ifftshift(gauss_kernel))
convolved = freq * freq_kernel

im_blur = fp.ifft2(convolved).real
im_blur = 255*im_blur/np.max(im_blur)

epsilon = 1.e-6
freq = fp.fft2(im_blur)
freq_kernel = 1/(epsilon + freq_kernel) # inverse filter
convolved = freq*freq_kernel
im_restored = fp.ifft2(convolved).real
im_restored = 255*im_restored / np.max(im_restored)

plt.figure(figsize=(10,10)), plt.gray()
plt.subplot(221), plt.imshow(im)
plt.title('Original image'), plt.axis('off')
plt.subplot(222), plt.imshow(im_blur)
plt.title('Blurred image'), plt.axis('off')
plt.subplot(223), plt.imshow(im_restored)
plt.title('Restored image with inverse filter'), plt.axis('off')
plt.subplot(224), plt.imshow(im_restored - im)
plt.title('Restored - Original image'), plt.axis('off')
plt.show()

#############################################
## 실습 14
##############################################

mblur_kernel = np.zeros((im.shape[0],im.shape[1]))
mblur_kernel[int((im.shape[0]-1)/2)-5:int((im.shape[0]-1)/2)+6,:] = np.ones([11,im.shape[1]])
##mblur_kernel[:,int((im.shape[1]-1)/2)-20:int((im.shape[1]-1)/2)+21] = np.ones([im.shape[0],41])
#mblur_kernel = mblur_kernel / (41*(im.shape[1]+im.shape[0]))
freq = fp.fft2(im)
convolved = freq * fp.ifftshift(mblur_kernel)
im_blur = fp.ifft2(convolved).real
im_blur = 255*im_blur/np.max(im_blur)

epsilon = 1.e-6
freq = fp.fft2(im_blur)
freq_kernel = 1/(epsilon + fp.ifftshift(mblur_kernel))
convolved = freq* fp.ifftshift(freq_kernel)
im_restored = fp.ifft2(convolved).real
im_restored = 255*im_restored / np.max(im_restored)

plt.figure(figsize=(10,10)), plt.gray()
plt.subplot(331), plt.imshow(im)
plt.title('Original image'), plt.axis('off')
plt.subplot(332), plt.imshow((20*np.log10(1+fp.fft2(im)).astype(int)))
plt.title('Original frquency'), plt.axis('off')
plt.subplot(333), plt.imshow(mblur_kernel)
plt.title('Bandpass filter'), plt.axis('off')
plt.subplot(334), plt.imshow(im_blur)
plt.title('Blurred image'), plt.axis('off')
plt.subplot(335), plt.imshow((20*np.log10(1+fp.fft2(im_blur)).astype(int)))
plt.title('Blurred frquency'), plt.axis('off')
plt.subplot(337), plt.imshow(im_restored)
plt.title('Restored image with inverse filter'), plt.axis('off')
plt.subplot(338), plt.imshow((20*np.log10(1+fp.fft2(im_restored)).astype(int)))
plt.title('Restored frquency'), plt.axis('off')
plt.subplot(339), plt.imshow(im_restored - im)
plt.title('Restored - Original image'), plt.axis('off')
plt.show()

##############################################
## 실습 15. Wiener 필터를 이용한 Deconvolution => Image Restration
#############################################
from skimage import color, data, restoration
from scipy.signal import convolve2d as conv2
import matplotlib.pyplot as plt
import numpy as np

im = color.rgb2gray(imread('tiger.png')[:,:,:3])
n = 7
psf = np.ones((n,n))/n**2
im1 = conv2(im, psf, 'same')
im2, _ = restoration.unsupervised_wiener(im1, psf)

fig, axs = plt.subplots(1,3,figsize=(12,4))
plt.gray()
axs[0].imshow(im), axs[0].axis('off')
axs[0].set_title('Original image',size=20)
axs[1].imshow(im1), axs[1].axis('off')
axs[1].set_title('Noisy blurred image',size=20)
axs[2].imshow(im), axs[2].axis('off')
axs[2].set_title('Self tuned restoration',size=20)
plt.tight_layout(), plt.show()

####################################################
## 실습 16. FFT로 영상잡음 제거
############################################
im = imread('moonlanding.jpg')
plt.figure(figsize=(10,10))
plt.gray()
plt.imshow(im[:,:,0])
plt.axis('off')
plt.show()

from scipy import fftpack
from matplotlib.colors import LogNorm
im_fft = fftpack.fft2(im[:,:,0])

def plot_spectrum(im_fft):
    plt.figure(figsize=(10,10))
    plt.gray()
    imm = np.log(np.abs(im_fft)+1.e-5)
    imm = np.min(imm) + imm
    imm = imm/np.max(imm)
    plt.imshow(imm) #, norm=LogNorm(vmin=5), cmap='afmhot')
    plt.colorbar()
    
plt.figure()
plot_spectrum(im_fft)
plt.title('Spectrum with Fourier transform',size=20)
plt.show()

## fft로 필터링
keep_fraction = 0.1
im_fft2 = im_fft.copy()
r,c = im_fft2.shape

im_fft2[int(r*keep_fraction):int(r*(1-keep_fraction))]=0
im_fft2[:,int(c*keep_fraction):int(c*(1-keep_fraction))] = 0

plt.figure()
plt.gray()
plot_spectrum(fftpack.fftshift(im_fft2))
plt.title('Filtered Spectrum')
plt.show()  

# 최종 영상 재구성
im_new = fftpack.ifft2(im_fft2).real
plt.figure(figsize=(10,10))
plt.gray()
plt.imshow(im_new)
plt.axis('off')
plt.title('Reconstructed Image',size=20) 
plt.show()
        
    
    




