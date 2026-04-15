# import numpy as np
# from PIL import Image, ImageFont, ImageDraw
# from PIL.ImageChops import add, subtract, multiply, difference, screen
# import PIL.ImageChops as stat 
# from skimage.io import imread, imsave, imshow, show
# from skimage import color, viewer, img_as_float, data
# from skimage.transform import SimilarityTransform, warp, swirl
# from skimage.util import random_noise
# import matplotlib.image as mpimg
# import matplotlib.pylab as plt
# from scipy import misc
#######################################################################
# # 파이썬에서 이 '#'다음에 오는 부분은 파이썬 프로그램이 인식하지 않는 부분이라
# 프로그램에는 아무 영향도 주지 않습니다. 
# 그러나, 내가 짠 프로그램을 내가 잘 모를 때, 
# 혹은 이 프로그램을 딴 사람들이 잘 이해시킬 때
# 이렇게 '#'뒤에 코멘트를 답니다.
# Written by Kiwoon Kwon
#######################################################################

######################################################################
# 실습 1 : 기초
######################################################################

# Print
# 명령 탭으로 가서 print("재미있는 영상과 파이썬") 을 입력해 보세요.
print("재미있는 영상과 파이썬")  # 커서를 ) 뒤에 놓고 위의 I 를 눌러 보세요.
print("제 이름은 권 입니다.")

# 사칙연산
12+25
12-25
12*25
12/25

# 변수 탭에서 생성되는 거 확인해 보세요. 
a = 1
g = 1.1
b = [1,2,3,4,5]
c = (1,2,3,4,5)
e = {14,15}
import numpy as np #Module
d = np.array((11,12,13))
f = np.array([[1,2,3],[4,5,6]])

# 리스트, 튜플과 array의 차이를 확인해보세요.
[1,2,3]+[4,5,6]
(1,2,3)+(4,5,6)
np.array((1,2,3))+np.array([4,5,6])

[1,2]+[3,4,5]
np.array([1,2])+np.array([3,4,5])

######################################################################
# 실습 2 : 그림 그리기
######################################################################
import matplotlib.pyplot as plt

x = np.linspace(0,2,21)
y = np.cos(np.pi*x)
z = 2*np.sin(np.pi*x)

plt.plot(y); plt.show() #(a)

plt.plot(x,y); plt.show() #(b)

plt.plot(y); plt.plot(x,y) #(c)

plt.plot(x,y,'b-*'); plt.show() #(d), b:blue, -실선, * 별모양

plt.plot(x,y,'b--*',x,z,'ro'); plt.show() #(e)

plt.plot(x,y,'b--*',x,z,'ro') 
plt.xlabel('x (unit:pi)'); plt.title('Cosine and 2*Sine functions') 
plt.legend(['cosine','2*sine']); plt.show()  #(f)

plt.plot(y,z); plt.show() #(h)

plt.plot(y,z); plt.axis('image'); plt.show() #(g)

########################################################################
# 실습 3 : 사진 불러오기
########################################################################
from skimage.io import imread
im = imread("MyungJin.jpg")
print(im.shape, im.dtype, type(im))

plt.figure(figsize=(10,10))
plt.imshow(im) 
#plt.axis('off')
plt.show()

imr = im[:,:,0]
plt.imshow(imr, cmap='gray')
plt.axis('off')
plt.show()

# colormap
# autumn(), bone(), cool(), copper(), flag(), gray(), hot(), hsv(), inferno(), jet(), magma(), nipy_spectral(),
# pink(), plasma(), prism(), spring(), summer(), viridis(), winter().
plt.cool(); plt.imshow(imr); plt.axis('off')
plt.flag(); plt.imshow(imr); plt.axis('off')
plt.jet(); plt.imshow(imr); plt.axis('off')

#####################################################
# 실습 4: 밝은 영상, 어두운 영상
#####################################################
from copy import copy 
from skimage import img_as_float, data

im = imread("Mountain.png") # im.flags WRITABLE=TRUE 이어야 함
imr = img_as_float(im[:,:,1]); ratio = 0.5
imblack = copy(imr); imwhite = copy(imr)
imblack[imr<ratio] = 0
imwhite[imr>ratio] = 1
    
plt.figure(figsize=(8,6)); print(im[0,0])
plt.subplot(221), plt.imshow(im), plt.title('Original')
plt.subplot(222), plt.imshow(imr,cmap='gray'), plt.title('Gray')
plt.subplot(223), plt.imshow(imblack,cmap='gray'), plt.title('Black')
plt.subplot(224), plt.imshow(imwhite,cmap='gray'), plt.title('White')
plt.show()    
    
###########################################
## 실습 5 : 보간법
###########################################
im = imr
methods = ['none','nearest','bilinear','bicubic','spline16','lanczos']
fig, axes = plt.subplots(nrows=2,ncols=3, figsize=(15,10)) 
for ax, interp_method in zip(axes.flat, methods):
    ax.imshow(im,interpolation=interp_method, cmap='gray')
    ax.set_title(str(interp_method), size=20)
plt.tight_layout()
plt.show()    

im = imr[25:75,80:120]
methods = ['none','nearest','bilinear','bicubic','spline16','lanczos']
fig, axes = plt.subplots(nrows=2,ncols=3, figsize=(15,10)) 
for ax, interp_method in zip(axes.flat, methods):
    ax.imshow(im,interpolation=interp_method, cmap='gray')
    ax.set_title(str(interp_method), size=20)
plt.tight_layout()
plt.show() 

############################################
## 실습 6: RGB, CMY
############################################
from skimage import color 
im = imread(r"C:\Users\ooiru\Sohn.png"); im = im[:,:,0:3]
hsv = im
#from copy import copy

hsv1 = copy(hsv); hsv2 = copy(hsv); hsv3 = copy(hsv) 
plt.figure(figsize=(8,6))
plt.subplot(221), plt.imshow(im), plt.title('Original')
hsv1[:,:,1]=0; hsv1[:,:,2]=0
plt.subplot(222), plt.imshow(hsv1), plt.title('RED')
hsv2[:,:,0]=0; hsv2[:,:,2]=0
plt.subplot(223), plt.imshow(hsv2), plt.title('GREEN')
hsv3[:,:,0]=0; hsv3[:,:,1]=0
plt.subplot(224), plt.imshow(hsv3), plt.title('BLUE')
plt.tight_layout(); plt.show() 

hsv1 = copy(hsv); hsv2 = copy(hsv); hsv3 = copy(hsv) 
plt.figure(figsize=(8,6))
plt.subplot(221), plt.imshow(im), plt.title('Original')
hsv1[:,:,2]=0
plt.subplot(222), plt.imshow(hsv1), plt.title('R+G=YELLOW')
hsv2[:,:,0]=0
plt.subplot(223), plt.imshow(hsv2), plt.title('G+B=CYAN')
hsv3[:,:,1]=0
plt.subplot(224), plt.imshow(hsv3), plt.title('R+B=MAGENTA')
plt.tight_layout(); plt.show()

hsv1 = copy(hsv); hsv2 = copy(hsv); hsv3 = copy(hsv) ;hsv4 = copy(hsv);
hsv5 = copy(hsv); hsv6 = copy(hsv); hsv7 = copy(hsv);    
plt.subplot(241), plt.imshow(im), plt.title('Original')
hsv1[:,:,1]=0; hsv1[:,:,2]=0
plt.subplot(242), plt.imshow(hsv1), plt.title('RED')
hsv2[:,:,0]=0; hsv2[:,:,2]=0
plt.subplot(243), plt.imshow(hsv2), plt.title('GREEN')
hsv3[:,:,0]=0; hsv3[:,:,1]=0
plt.subplot(244), plt.imshow(hsv3), plt.title('BLUE')
hsv4[:,:,2]=0
plt.subplot(245), plt.imshow(hsv4), plt.title('R+G=YELLOW')
hsv5[:,:,0]=0
plt.subplot(246), plt.imshow(hsv5), plt.title('G+B=CYAN')
hsv6[:,:,1]=0
plt.subplot(247), plt.imshow(hsv6), plt.title('R+B=MAGENTA')
hsv7[:, :, :] = 0
plt.subplot(248), plt.imshow(hsv7), plt.title('BLACK')
plt.tight_layout()
plt.show()

####################################
## ETC.
## scikit-image의 데이터셋과 misc 데이터셋
####################################
from skimage import data
from skimage.io import imshow, show

im = data.astronaut()
imshow(im)
show()

im = data.camera()
imshow(im)
show()

im = data.coins()
imshow(im)
show()

im = data.moon()
imshow(im)
show()

im = data.coffee()
imshow(im), show()

im = data.chelsea()
imshow(im), show()

im = data.checkerboard()
imshow(im)
show()

from scipy.datasets import face
im = face()
print(im.shape, im.dtype, type(im))
plt.imshow(im), plt.axis('off')
plt.show()

##############################################
# 실습 7 : 영상 조작 
##############################################
import numpy as np
from skimage.io import imread
import matplotlib.pylab as plt
from copy import copy
lena = imread("Lenna.png")
lena2 = copy(lena)
lx,ly,_ = lena.shape
X,Y=np.ogrid[0:lx,0:ly]
mask = (X-lx/2)**2 + (Y-ly/2)**2> lx*ly/4
lena[mask,:]=0
plt.figure(figsize=(10,10))
plt.subplot(121), plt.imshow(lena2), plt.title('Original'), plt.axis('off')
plt.subplot(122), plt.imshow(lena), plt.title('Mask'), plt.axis('off')
plt.show()

##############################################
# 실습 8 : alpha 블렌딩 : 메시에서 호나우두
# (1-alpha)*메시 + alpha*호나우두
##############################################
import matplotlib.image as mpimg
im1 = mpimg.imread("messi.jpg")/255
im2 = mpimg.imread("ronaldo.jpg")/255


i=1
plt.figure(figsize=(18,15))
for alpha in np.linspace(0,1,20):
    plt.subplot(4,5,i)
    plt.imshow((1-alpha)*im1 + alpha*im2)
    plt.axis('off')
    i += 1
plt.subplots_adjust(wspace=0.05,hspace=0.05)
plt.show()




##############################################
# size not same so change code slightly
import matplotlib.image as mpimg
from skimage.transform import resize
im1 = mpimg.imread("messi.jpg")
im2 = mpimg.imread("ronaldo.jpg")

target_shape = (min(im1.shape[0], im2.shape[0]), min(im1.shape[1], im2.shape[1]))
im1 = resize(im1, target_shape, anti_aliasing=True)
im2 = resize(im2, target_shape, anti_aliasing=True)

i=1
plt.figure(figsize=(18,15))
for alpha in np.linspace(0,1,20):
    plt.subplot(4,5,i)
    plt.imshow((1-alpha)*im1 + alpha*im2)
    plt.axis('off')
    i += 1
plt.subplots_adjust(wspace=0.05,hspace=0.05)
plt.show()





##############################################
# 실습 9 : 영상 반전 
##############################################
im = imread("parrot.png")  #RGBA
imRGB = im[:,:,0:3]    #RGBA ==> RGB
imR = im[:,:,0]
plt.figure(figsize=(10,10))
plt.subplot(221), plt.imshow(imRGB), plt.title('RGB'), plt.axis('off')
plt.subplot(222), plt.imshow(255-imRGB), plt.title('RGB:Complement'), plt.axis('off')
plt.subplot(223), plt.imshow(imR,cmap='gray'), plt.title('Red'), plt.axis('off')
plt.subplot(224), plt.imshow(255-imR, cmap='gray'), plt.title('Red:Complement'), plt.axis('off')
plt.show()


##Extra
from skimage.transform import SimilarityTransform, warp, swirl

im = imread("parrot.png")
tform = SimilarityTransform(scale=0.9, rotation=np.pi/4, translation=(im.shape[0]/2,-100)) #im.shape[1]/2))#,-100)

warped = warp(im,tform)

plt.imshow(warped), plt.axis('off')
plt.show()


