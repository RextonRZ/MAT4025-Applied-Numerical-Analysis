# 영상 불러오기 및 그리기
import numpy as np

import matplotlib.pyplot as plt  # pylab  #plt.imshow, plt.show
import matplotlib.image as mpimg

from skimage.io import imread #, imsave, imshow, show
from skimage import color, data #, viewer, img_as_float
from skimage.transform import SimilarityTransform, warp, swirl, rescale
from skimage.util import random_noise

from PIL import Image #, ImageFont, ImageDraw
from PIL.ImageChops import add, subtract, multiply, difference, screen

from scipy import misc, ndimage, signal, stats, fftpack as fp

## PIL을 이용한 영상조작
# 많은 함수 제공
# 점 변환을 사용하여 화소값 변경
# 영상에서 기하학적 변환 수행

################################################
## 실습 1 RGB  채널에 대한 화소 값 히스토 그램 그리기
###################################################
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
im = Image.open("clock.jpg")

plt.figure(figsize=(18,6)) 
plt.subplot(121), plt.imshow(im)
plt.subplot(122)
pl = im.histogram()
plt.bar(range(256),  pl[:256], color='r', alpha = 0.5)
plt.bar(range(256), pl[256:2*256], color='g', alpha = 0.4)
plt.bar(range(256),  pl[2*256:], color='b', alpha = 0.3)
plt.show()

# RGB 채널 분리
ch_r, ch_g, ch_b = im.split()
im2 = Image.merge('RGB',(ch_b,ch_g,ch_r))
plt.figure(figsize=(10,6))
plt.subplot(2,2,1), plt.imshow(ch_r, cmap='Reds'), plt.axis('off')
plt.subplot(2,2,2), plt.imshow(ch_g, cmap='Greens'), plt.axis('off')
plt.subplot(2,2,3), plt.imshow(ch_b, cmap='Blues'), plt.axis('off')
plt.subplot(2,2,4), plt.imshow(im2), plt.axis('off')
plt.tight_layout(), plt.show()

################################################
## 실습 2. 영상 자르기와 크기 조정
###################################################
# parrot : width, height, mode, format
im = Image.open("parrot.png")
print(im.width, im.height, im.mode, im.format),

# 영상 자르기 crop, 영상 크기 조정 resize
im.show()
im_c = im.crop((200,50,700,400)) #left, top, right, bottom
im_c.show() 
im2 = im.resize((100,100))
im2.show()

plt.figure(figsize=(18,6))
plt.subplot(1,3,1), plt.imshow(im)
plt.subplot(1,3,2), plt.imshow(im_c)
plt.subplot(1,3,3), plt.imshow(im2)
plt.show()

################################################
## 실습 3. 영상 크기 축소, 확대, 업샘플링, 다운 샘플링
###################################################
im = Image.open("clock.jpg")
print(im.width, im.height) #259 194

ims = im.resize((im.width//5,im.height//5),Image.LANCZOS) # 앤티 앨리어싱, 고품질 다운 샘플링 기술
print(ims.width, ims.height)

# Use Image.NEAREST (0), Image.LANCZOS (1), Image.BILINEAR (2), Image.BICUBIC (3), Image.BOX (4) or Image.HAMMING (5)
im_bilinear = ims.resize((ims.width*5,ims.height*5),Image.BILINEAR) # 양선형 보간법
im_nearest = ims.resize((ims.width*5,ims.height*5),Image.NEAREST)
im_lanczos = ims.resize((ims.width*5,ims.height*5),Image.LANCZOS)
im_bicubic = ims.resize((ims.width*5,ims.height*5),Image.BICUBIC)
im_box = ims.resize((ims.width*5,ims.height*5),Image.BOX)
im_hamming = ims.resize((ims.width*5,ims.height*5),Image.HAMMING)

plt.figure(figsize=(8,8))
plt.subplot(3,3,1), plt.imshow(im), plt.title('Original')
plt.subplot(3,3,2), plt.imshow(ims), plt.title('Small')
plt.subplot(3,3,3), plt.imshow(im_nearest), plt.title('Nearest')
plt.subplot(3,3,4), plt.imshow(im_lanczos), plt.title('LANCZOS')
plt.subplot(3,3,5), plt.imshow(im_bilinear), plt.title('Bilinear')
plt.subplot(3,3,6), plt.imshow(im_bicubic), plt.title('BICUBIC')
plt.subplot(3,3,7), plt.imshow(im_box), plt.title('BOX')
plt.subplot(3,3,8), plt.imshow(im_hamming), plt.title('HAMMING')
plt.show()

########################################################
im = Image.open("victoria_memorial.png")
print(im.width, im.height) #313, 145

ims = im.resize((im.width//5,im.height//5), Image.LANCZOS) # 앤티 앨리어싱, 고품질 다운 샘플링 기술
print(ims.width, ims.height)

# Use Image.NEAREST (0), Image.LANCZOS (1), Image.BILINEAR (2), Image.BICUBIC (3), Image.BOX (4) or Image.HAMMING (5)
im_bilinear = ims.resize((ims.width*5,ims.height*5),Image.BILINEAR) # 양선형 보간법
im_nearest = ims.resize((ims.width*5,ims.height*5),Image.NEAREST)
im_lanczos = ims.resize((ims.width*5,ims.height*5),Image.LANCZOS)
im_bicubic = ims.resize((ims.width*5,ims.height*5),Image.BICUBIC)
im_box = ims.resize((ims.width*5,ims.height*5),Image.BOX)
im_hamming = ims.resize((ims.width*5,ims.height*5),Image.HAMMING)

plt.figure(figsize=(8,8))
plt.subplot(3,3,1), plt.imshow(im), plt.title('Original')
plt.subplot(3,3,2), plt.imshow(ims), plt.title('Small')
plt.subplot(3,3,3), plt.imshow(im_nearest), plt.title('Nearest')
plt.subplot(3,3,4), plt.imshow(im_lanczos), plt.title('LANCZOS')
plt.subplot(3,3,5), plt.imshow(im_bilinear), plt.title('Bilinear')
plt.subplot(3,3,6), plt.imshow(im_bicubic), plt.title('BICUBIC')
plt.subplot(3,3,7), plt.imshow(im_box), plt.title('BOX')
plt.subplot(3,3,8), plt.imshow(im_hamming), plt.title('HAMMING')
plt.show()

################################################
## 실습 4. 밝기 영상과 로그/파워-로우 변환
###################################################
# 영상을 명암도로 변환, point(), 로그 변환과 파워-로우 변환
im = Image.open("clock.jpg")
im_g = im.convert('L')
# im_g.show()
# 명암도 변환   : 
# 로그 변환
#im_g.point(lambda x: 255*np.log(1+x/255)).show()
#np.log(2.7)
255/np.log(2) # 367
367*np.log(2) # 255
im_log = im_g.point(lambda x: 367*np.log(1+x/255))
im_3   = im_g.point(lambda x: 255*(x/255)**3)
im_06  = im_g.point(lambda x: 255*(x/255)**0.6)

plt.figure(figsize=(8,8))
plt.subplot(2,2,1), plt.imshow(im_g,cmap='gray'), plt.title('Original')
plt.subplot(2,2,2), plt.imshow(im_log,cmap='gray'), plt.title('log')
plt.subplot(2,2,3), plt.imshow(im_3,cmap='gray'), plt.title('3')
plt.subplot(2,2,4), plt.imshow(im_06,cmap='gray'), plt.title('0.6')
plt.tight_layout(), plt.show()

################################################
## 실습 5. 기하학적 변환, 영상반사
###################################################
imlr = im.transpose(Image.FLIP_LEFT_RIGHT)
imtb = im.transpose(Image.FLIP_TOP_BOTTOM)
#im2 = im.transpose(Image.FLIP_LEFT_RIGHT)
imdiag = imlr.transpose(Image.FLIP_TOP_BOTTOM)

plt.figure(figsize=(8,8))
plt.subplot(2,2,1), plt.imshow(im,cmap='gray'), plt.axis('off'),plt.title('Original')
plt.subplot(2,2,2), plt.imshow(imlr,cmap='gray'), plt.axis('off'),plt.title('LeftRight')
plt.subplot(2,2,3), plt.imshow(imtb,cmap='gray'), plt.axis('off'),plt.title('TopBottom')
plt.subplot(2,2,4), plt.imshow(imdiag,cmap='gray'), plt.axis('off'),plt.title('Diagonal')
plt.tight_layout(), plt.show()

################################################
## 실습 6. 영상회전
###################################################
im45 = im.rotate(45)
img45 = im_g.rotate(45)
img45c = im_g.point(lambda x: 255-x).rotate(45)
img30 = im_g.rotate(30)
img60 = im_g.rotate(60)
img90 = im_g.rotate(90)

plt.figure(figsize=(12,8))
plt.subplot(2,3,1), plt.imshow(im), plt.axis('off'), plt.title('Original',fontsize=30)
plt.subplot(2,3,2), plt.imshow(im45), plt.axis('off'), plt.title('45color',fontsize=30)
plt.subplot(2,3,3), plt.imshow(img45c,cmap='gray'), plt.axis('off'), plt.title('45Complement',fontsize=30)
plt.subplot(2,3,4), plt.imshow(img30,cmap='gray'), plt.axis('off'), plt.title('30',fontsize=30)
plt.subplot(2,3,5), plt.imshow(img60,cmap='gray'), plt.axis('off'), plt.title('60',fontsize=30)
plt.subplot(2,3,6), plt.imshow(img90,cmap='gray'), plt.axis('off'), plt.title('90',fontsize=30)
plt.tight_layout(), plt.show()

################################################
## 실습 7. 영상 Affine 변환
###################################################
# 영상에 어파인 변환 적용
# T : 역 매핑으로 구현되는 경우가 많다.
# 전단행렬
# 6개의 튜플 (a,b,c,d,e,f)
# (ax+by+c, dx+ey+f)
imb = im_g
size = (int(imb.width), imb.height)    
imt0 = imb.transform(size, Image.AFFINE, data=(1,-0.5,0,0,1,0))

size = (int(1.4*imb.width), imb.height)   
imt1 = imb.transform(size, Image.AFFINE, data=(1,-0.5,0,0,1,0))

a = 1; b = -0.4; c = 0 #-300
d = -0.2; e = 0.7; f = 0 #+100
det = a*e - b*d
w = (e-b)/det 
h = (a-d)/det
size = (int(w*imb.width)-c, int(h*imb.height)-f)    
imt2 = imb.transform(size, Image.AFFINE, data=(a,b,c,d,e,f))

c = -300; f = 100
imt3 = imb.transform(size, Image.AFFINE, data=(a,b,c,d,e,f))

size = (int(w*imb.width)-c, int(h*imb.height)-f)   
imt4 = imb.transform(size, Image.AFFINE, data=(a,b,c,d,e,f))

plt.figure(figsize=(12,6))
plt.subplot(2,3,1), plt.imshow(imb,cmap='gray'), plt.title('Original',fontsize=30)
plt.subplot(2,3,2), plt.imshow(imt0,cmap='gray'), plt.title('(x-0.5y,y)',fontsize=20)
plt.subplot(2,3,3), plt.imshow(imt1,cmap='gray'), plt.title('(x-0.5y,y),1.4width',fontsize=20)
plt.subplot(2,3,4), plt.imshow(imt2,cmap='gray'), plt.title('(1,-0.4,0,-0.2,0.7,0)',fontsize=20)
plt.subplot(2,3,5), plt.imshow(imt3,cmap='gray'), plt.title('(1,-0.4,-300,-0.2,0.7,100)',fontsize=20)
plt.subplot(2,3,6), plt.imshow(imt4,cmap='gray'), plt.title('(1,-0.4,-300,-0.2,0.7,100),size',fontsize=20)
plt.tight_layout(), plt.show()

################################################
## 실습 8. 사영변환
###################################################
img = im_g
width, height = img.size
m = -0.5
xshift = abs(m) * width
new_width = width + int(round(xshift))
imaffine = img.transform((new_width, height), Image.AFFINE,
        (1, m, -xshift if m > 0 else 0, 0, 1, 0), Image.BICUBIC)

def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float64)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

coeffs1 = find_coeffs(
        [(0, 0), (width, 0), (width, height), (0, height)],
        [(0, 0), (width, 0), (new_width, height), (xshift, height)])
im1 = img.transform((width, height), Image.PERSPECTIVE, coeffs1, Image.BICUBIC)

coeffs2 = find_coeffs(
        [(0, 0), (width, 0), (width, height), (0, height)],
        [(0, 0), (width, 0), (int(width*3), height), (0, height)] )
im2 = img.transform((width, height), Image.PERSPECTIVE, coeffs2, Image.BICUBIC)

coeffs3 = find_coeffs(
        [(0, 0), (width, 0), (width, height), (0, height)],
        [(0, 0), (width, 0), (int(width*3), height), (-width/2, height)] )
im3 = img.transform((width, height), Image.PERSPECTIVE, coeffs3, Image.BICUBIC)

coeffs4 = find_coeffs(
        [(0, 0), (width, 0), (width, height), (0, height)],
        [(0, 0), (width, 0), (int(width*3), height), (-width/2, height*3)] )
im4 = img.transform((width, height), Image.PERSPECTIVE, coeffs4, Image.BICUBIC)

plt.figure(figsize=(10,8))
plt.subplot(3,2,1), plt.imshow(img,cmap='gray'), plt.title('Original')
plt.subplot(3,2,2), plt.imshow(imaffine,cmap='gray'), plt.title('Affine')
plt.subplot(3,2,3), plt.imshow(im1,cmap='gray'), plt.title('(0.5W,H,1.5W,H)')
plt.subplot(3,2,4), plt.imshow(im2,cmap='gray'), plt.title('(0,H,3W,H)')
plt.subplot(3,2,5), plt.imshow(im3,cmap='gray'), plt.title('(-0.5W,H,3W,H)')
plt.subplot(3,2,6), plt.imshow(im4,cmap='gray'), plt.title('(-0.5W,3H,3W,H)')
plt.tight_layout(), plt.show()

# 원근 변환
# params1 = [1,0.1,0,-0.1,0.5,0,0,-0.005,-0.001]
# params2 = [1,0.5,0,-0.1,0.5,0,0,-0.005,1]
# params3 = [1,1,0,-0.1,0.5,0,0,-0.005,10]
# size = (int(1.5*imb.width), 2*imb.height )
# im1 = im_g.transform(size, Image.PERSPECTIVE, params1, Image.BICUBIC)
# im2 = im_g.transform(size, Image.PERSPECTIVE, params2, Image.BICUBIC)
# im3 = im_g.transform(size, Image.PERSPECTIVE, params3, Image.BICUBIC)

# plt.figure(figsize=(8,8))
# plt.subplot(2,2,1), plt.imshow(im_g,cmap='gray'), plt.title('Original')
# plt.subplot(2,2,2), plt.imshow(im1,cmap='gray') 
# plt.subplot(2,2,3), plt.imshow(im2,cmap='gray') 
# plt.subplot(2,2,4), plt.imshow(im3,cmap='gray') 
# plt.tight_layout(), plt.show()

################################################
## 실습 9. 영상의 화소값 변경 putpixel(), 소금후추 잡음, 영상에 글자/타원 그리기
###################################################
im = Image.open("parrotRGB.png") # RGB 저장
iml = im.copy()
n = 50000
x = np.random.randint(0,im.width,n)
y = np.random.randint(0,im.height,n)
for (x,y) in zip(x,y):
    new_pix = (0,0,0) if np.random.rand() < 0.5 else (255,255,255)
    iml.putpixel((x,y),new_pix)
iml.show()

# 영상에 그리기 : ellipse() , PIL.ImageDraw
from PIL import ImageDraw, ImageFont
iml2 = im.copy()
draw = ImageDraw.Draw(iml2)
draw.ellipse((500,700,650,750), fill=(255,255,255))
del draw
iml2.show()

iml3 = iml2.copy()
draw = ImageDraw.Draw(iml3)
draw.ellipse((500,500,550,650), fill=(255,255,255))
del draw
iml3.show()

# 영상에 글자 쓰기
iml4 = im.copy()
draw = ImageDraw.Draw(iml4)
font = ImageFont.truetype("arial.ttf",23)
draw.text((100,20), 'Welcone to Image Processing with python', font=font)
del draw
iml4.show()

plt.figure(figsize=(8,8))
plt.subplot(2,2,1), plt.imshow(iml), plt.axis('off'), plt.title('Noise')
plt.subplot(2,2,2), plt.imshow(iml2), plt.axis('off'), plt.title('Ellipse')
plt.subplot(2,2,3), plt.imshow(iml3), plt.axis('off'), plt.title('Two Ellipses')
plt.subplot(2,2,4), plt.imshow(iml4), plt.axis('off'), plt.title('Text')
plt.tight_layout(), plt.show()

# # 섬네일 생성
im = Image.open("parrot.png") # RGB 저장
im_thumbnail = im.copy()
im_thumbnail.thumbnail((100,100))
im_thumbnail.show()

################################################
## 실습 10. alpha 블렌딩 : blend()
# out = image1*(1-alpha) + image2*alpha
###################################################
from PIL import Image, ImageFont, ImageDraw
im1 = Image.open("parrot.png")
im2 = Image.open("clock.jpg")
im3 = Image.open("victoria_memorial.png")
im1.mode  
im2.mode 
im3.mode 
im2 = im2.convert('RGBA') # 두영상의 컬러 모드 다름. 같은 모드 변경
im2 = im2.resize((im1.width,im1.height), Image.BILINEAR)
im3 = im3.resize((im1.width,im1.height), Image.BILINEAR)
im02 = Image.blend(im1,im2,alpha=0.2)
im04 = Image.blend(im1,im2,alpha=0.4)
im06 = Image.blend(im1,im2,alpha=0.6)
im08 = Image.blend(im1,im2,alpha=0.8)


plt.figure(figsize=(8,6))
plt.subplot(2,3,1), plt.imshow(im1), plt.axis('off'), plt.title('0',fontsize=30)
plt.subplot(2,3,2), plt.imshow(im02), plt.axis('off'),plt.title('0.2',fontsize=30)
plt.subplot(2,3,3), plt.imshow(im04), plt.axis('off'),plt.title('0.4',fontsize=30)
plt.subplot(2,3,4), plt.imshow(im06), plt.axis('off'),plt.title('0.6',fontsize=30)
plt.subplot(2,3,5), plt.imshow(im08), plt.axis('off'),plt.title('0.8',fontsize=30)
plt.subplot(2,3,6), plt.imshow(im2), plt.axis('off'),plt.title('1',fontsize=30)
plt.tight_layout(), plt.show()

################################################
## 실습 11. 두 영상에 대한 사칙연산
#################################################
from PIL.ImageChops import add, subtract, multiply, difference, screen
E = multiply(im1,im2)
A = add(im1,im2) # ((image1 + image2) / scale + offset)
B = difference(im1,im2)
C = subtract(im2,im1)
C2 = subtract(im1,im2)
D = screen(im1,im2) #MAX - ((MAX - image1) * (MAX - image2) / MAX) 

plt.figure(figsize=(8,8))
plt.subplot(3,3,1), plt.imshow(im1), plt.axis('off'), plt.title('Parrot')
plt.subplot(3,3,2), plt.imshow(im2), plt.axis('off'),plt.title('Clock')
plt.subplot(3,3,3), plt.imshow(A), plt.axis('off'),plt.title('P+C')
plt.subplot(3,3,4), plt.imshow(B), plt.axis('off'),plt.title('|P-C|')
plt.subplot(3,3,5), plt.imshow(C2), plt.axis('off'),plt.title('P-C')
plt.subplot(3,3,6), plt.imshow(C), plt.axis('off'),plt.title('C-P')
plt.subplot(3,3,7), plt.imshow(D), plt.axis('off'),plt.title('screen')
plt.subplot(3,3,8), plt.imshow(E), plt.axis('off'),plt.title('multiply')
plt.tight_layout(), plt.show()

E = multiply(im1,im3)
A = add(im1,im3)
B = difference(im1,im3)
C = subtract(im3,im1)
C2 = subtract(im1,im3)
D = screen(im1,im3)

plt.figure(figsize=(8,8))
plt.subplot(3,3,1), plt.imshow(im1), plt.axis('off'),plt.title('Parrot')
plt.subplot(3,3,2), plt.imshow(im3), plt.axis('off'),plt.title('Memorial')
plt.subplot(3,3,3), plt.imshow(A), plt.axis('off'),plt.title('P+M')
plt.subplot(3,3,4), plt.imshow(B), plt.axis('off'),plt.title('|P-M|')
plt.subplot(3,3,5), plt.imshow(C2), plt.axis('off'),plt.title('P-M')
plt.subplot(3,3,6), plt.imshow(C), plt.axis('off'),plt.title('M-P')
plt.subplot(3,3,7), plt.imshow(D), plt.axis('off'),plt.title('screen')
plt.subplot(3,3,8), plt.imshow(E), plt.axis('off'),plt.title('multiply')
plt.tight_layout(), plt.show()

E = multiply(im2,im3)
A = add(im2,im3)
B = difference(im3,im2)
C = subtract(im3,im2)
C2 = subtract(im2,im3)
D = screen(im2,im3)

plt.figure(figsize=(8,8))
plt.subplot(3,3,1), plt.imshow(im2), plt.axis('off'),plt.title('Clock')
plt.subplot(3,3,2), plt.imshow(im3), plt.axis('off'),plt.title('Memorial')
plt.subplot(3,3,3), plt.imshow(A), plt.axis('off'),plt.title('C+M')
plt.subplot(3,3,4), plt.imshow(B), plt.axis('off'),plt.title('|C-M|')
plt.subplot(3,3,5), plt.imshow(C2), plt.axis('off'),plt.title('C-M')
plt.subplot(3,3,6), plt.imshow(C), plt.axis('off'),plt.title('M-C')
plt.subplot(3,3,7), plt.imshow(D), plt.axis('off'),plt.title('screen')
plt.subplot(3,3,8), plt.imshow(E), plt.axis('off'),plt.title('multiply')
plt.tight_layout(), plt.show()

############################################
# 실습11. 영상의 윤곽선 그리기
# 영상의 등고선과 채워진 윤곽선
#############################################
from skimage import color, img_as_float, data
from skimage.io import imread
im = color.rgb2gray(imread("parrotRGB.png"))

plt.figure()
plt.subplot(131), plt.imshow(im, cmap='gray'), 
plt.axis('off'), plt.title('Original Image',size=7)
plt.subplot(132)
plt.contour(np.flipud(im), colors='k', levels=np.logspace(-15,15,100))
plt.axis('off'), plt.axis('equal'), plt.title('Image Contour Lines', size=7)
plt.subplot(133), 
plt.contour(np.flipud(im), cmap='hot')
plt.axis('off'), plt.axis('equal'), plt.title('Image Filled Contour', size=7)
#plt.colorbar()
plt.show() 

##########################################
## 실습 12. 영상에 랜덤 가우시안 잡음 추가 : random_noise()
############################################
from skimage.util import random_noise
from skimage import color, img_as_float, data
im = img_as_float(imread("parrot.png"))
sigmas = [0.1, 0.25, 0.4, 1]

plt.figure(figsize=(15,15))    
for i in range(4):
    noisy = random_noise(im, var=sigmas[i]**2)
    plt.subplot(2,2,i+1), plt.imshow(noisy), plt.axis('off')
    plt.title("Gaussian noise with sigma="+str(sigmas[i]), size=20)
plt.tight_layout()
plt.show()

############################################
# 실습13 : 
# warp 함수를 이용한 Affine 기하변환, transform 모듈
# SimilarityTransform() 함수를 사용하여 변환행렬 계산
# swirl 비선형 변환 :
# strength : swirl 크기
# radius: swril 화소단위의 범위
# rotation : 회전 각도
#################################################
from skimage.transform import SimilarityTransform, warp, swirl

im = imread("parrotRGB.png")
tform = SimilarityTransform(scale=0.9, rotation=np.pi/4, translation=(im.shape[0]/2,-200))#im.shape[1]/2))#,-100))
warped = warp(im, tform) 

swirled = swirl(im, rotation=np.pi/4, strength=50, radius=500)

plt.figure()
plt.subplot(1,2,1), plt.imshow(warped), plt.axis('off'), plt.title('Warp',fontsize=30)
plt.subplot(1,2,2), plt.imshow(swirled), plt.axis('off'), plt.title('Swirl',fontsize=30)
plt.show()




