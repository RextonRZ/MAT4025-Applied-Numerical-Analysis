##### Ch8.  영상 분할
# scikit-image 사용
# 허프 변환 : 영상에서 원과 직선 검출
# 임계치와 오쓰 분할
# 에지 기반/영역 기반 분할
#Felzenszwalb, SLIC, QuickShift, Compact Watershed 알고리즘
# 능동 윤곽선, 형태학적 스네이크와 GrabCut 알고리즘 (python open cv 사용)

## 영상분할이란 무엇인가?
# 서로 다른 객체 또는 객체의 일부에 해당하는 별개의 영역 또는 범주로 영상을 분할
# 각 영역에는 유사한 속성을 가진 화소가 포함됨
# 영상의 각 화소는 이러한 범주 중 하나에 할당 ==> 영상 분류
# 좋은 분할은 동일한 범주내의 화솟값들이 유사
# 사이한 범주에 이웃하는 화솟값은 다른 발ㄲ기값을 갖는다. 
# 분할의 목표는 영상의 표현을 보다 의미있고 쉽게 분석할 수 있도록 단순화거나 변경하는 것
# 분할의 품질과 신뢰성은 영상분석의 성공여부를 결정
# 영상을 올바르게 분할하는 것은 매우 어려운 문제
# noncontextual : 화소의 공간적 관계를 고려하지 않음
# contextual : 공간적 관계를 고려함
# scima-image, python-opencv(cv2), SimpleITK

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.pyplot import cm
import cv2

from skimage.io import imread
from skimage.color import rgb2gray,label2rgb
from skimage.feature import canny, corner_harris, corner_subpix, corner_peaks
from skimage.transform import warp, AffineTransform, hough_line, hough_line_peaks, hough_circle, hough_circle_peaks
from skimage.exposure import rescale_intensity
from skimage.measure import ransac
from skimage.draw import ellipse, circle_perimeter
from skimage import data, img_as_float #data.astronaut

#################################### 
# Example 1.허프 변환-직선과 원 검출
#####################################
# 매개 변수 공간에서 수행되는 투표 절차를 사용하여 특정모양 객체의 인스턴스를 잦는 것을 목표로 하는 특징 추출 기술
# 고전적인 허프변환은 영상에서 직선을 검출
# 극좌표(rho, theta)를 사용하여 직선을 나타냄, rho: 선분의 길이, theta: 직선과 x축 사이의 각도
# 먼저 2차원 히스토그램을 만든다.
# rho와 theta이 각 값에 대해 입력 영상에서 해당 직선에 가가운 0이 아닌 화소의 수 계산
# 이에 따라 (rho,theta) 배열을 증가시킨다.
# 따라서 각각의 0이 아닌 화소는 잠재적 직선 후보를 투표하는 것으로 간주한다.
# 가장 가능성 높은 직선은 가장높은 득표를 얻은 매개변수 값, 즉 2차원 히스토그램의 로컬 최대 값에 해당

# 이 방법은 원을 감지하도록 확장된다. 
# 유사한 투표방법을 사용하여 원의 매개 변수 공간에서 최대값을 찾는다.
# 곡선의 매개변수가 많을수록 허프 변환을 사용하여 곡선을 감지하는데 공간적으로나 계산적으로 많은 비용이 든다.
# hough_line(), hough_line_peaks : 직선 검출
# hough_circle(), hough_circle_peaks : 원 검출

# 영상 보기 함수
# Lines 
def plot_image(image,title, cmap='gray'):
    plt.imshow(image, cmap=cmap), plt.title(title, size=20), plt.axis('off')
def plot_ax_image(ax,image,title,cmap='gray',size=20):
    ax.imshow(image,cmap=cmap), ax.set_title(title), ax.axis('off')    

gray = imread("TriangleCircle.png")
# image = gray2rgb(imread("TriangleCircle.png"))
# gray = rgb2gray(image)
image = np.zeros((200, 200))
idx = np.arange(25, 175)
image[idx, idx] = 255
# cv2.line(image,(45, 25), (25, 175),(0,255,0),2)
# image[line(25, 135, 175, 155)] = 255
#gray = rgb2gray(rgba2rgb(imread("TriangleCircle.jpg")))

fig, axes = plt.subplots(1,3,figsize=(10,10))
axes = axes.ravel()
plot_ax_image(axes[0], 1-image, 'Input image')

h, theta, d = hough_line(image)
extent = [np.rad2deg(theta[0]), np.rad2deg(theta[-1]), d[0],d[-1]]
axes[1].imshow(1-h, extent=extent, cmap=cm.hot, aspect=1.5)
axes[1].set_title("Hough transform", size=20)
axes[1].set_xlabel('Angles (degrees)', size=10)
axes[1].set_ylabel('Distance (pixels)', size=10)
axes[1].axis('image')      

hough = hough_line_peaks(h,theta,d)
axes[2].imshow(1-image,cmap='gray') 

for _, angle, dist in zip(*hough):
    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
    axes[2].axline((x0, y0), slope=np.tan(angle + np.pi/2))  
axes[2].set_ylim((image.shape[0],0))
axes[2].set_axis_off(), axes[2].set_title('Detected lines',size=20)   
plt.show() 

## Circles
#bgr_img = cv2.imread('TriangleCircle.png') # read as it is
bgr_img = cv2.imread('us_coins.jpg') # read as it is
#bgr_img = cv2.imread('Eagle_coins.jpg') # read as it is

if bgr_img.shape[-1] == 3:           # color image
    b,g,r = cv2.split(bgr_img)       # get b,g,r
    rgb_img = cv2.merge([r,g,b])     # switch it to rgb
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
else:
    gray_img = bgr_img

#img = cv2.medianBlur(gray_img, 5) #TriangleCircle
img = cv2.medianBlur(gray_img, 31) #us_coins
#img = cv2.medianBlur(gray_img, 151) #Eagle_coins

cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
            param1=50,param2=30,minRadius=0,maxRadius=0)

circles = np.uint16(np.around(circles))

for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

plt.subplot(121),plt.imshow(rgb_img)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(cimg)
plt.title('Hough Transform'), plt.xticks([]), plt.yticks([])
plt.show()
      
#################################################
# Example 2. 에지 기반 분할
#################################################
# 에지 기반 분할을 사용하여 동전 윤곽선 묘사
# 첫번째 단계 : 캐니 에지 검출기
#coin = imread('us_coins.jpg')
import matplotlib.pyplot as plt
from skimage import data
from skimage.feature import canny
from skimage import morphology
from scipy import ndimage as ndi

# Helper function to plot images
def plot_ax_image(ax, image, title, cmap='gray', size=20):
    ax.imshow(image, cmap=cmap)
    ax.set_title(title, size=size)
    ax.axis('off')

coins = data.coins()
edges = canny(coins, sigma=2)
fig, axes = plt.subplots(figsize=(7,7))
plot_ax_image(axes, edges, 'Canny detector')
plt.show()

# filling the holes
from scipy import ndimage as ndi

fill_coins = ndi.binary_fill_holes(edges)
_, axes = plt.subplots(figsize=(10,6))
plot_ax_image(axes, fill_coins, 'Filling the holes')
plt.show()

# Removing small objects
from skimage import morphology

coins_cleaned = morphology.remove_small_objects(fill_coins, 50)
_, axes = plt.subplots(figsize=(10,6))
plot_ax_image(axes, coins_cleaned, 'Removing small objects')
plt.show()

####################################################
# Example 3. Region based Segmentation
####################################################
from skimage.filters import sobel

elevation_map = sobel(coins)
plot_image(elevation_map,'Elevation map')
plt.show()

# Marker
markers = np.zeros_like(coins)
markers[coins<30] = 1
markers[coins>150] = 2
print(np.max(markers), np.min(markers))

plt.figure(figsize=(10,6))
plot_image(markers, 'Markers', cmap='hot')
plt.colorbar()
plt.show()   

# Watershed
import skimage.segmentation as segm
seg = segm.watershed(elevation_map, markers)
fig, axes = plt.subplots(figsize=(10,6))
plot_ax_image(axes, seg, 'Watershed') 
plt.show()

####################################################
# Example 4. Labelling
#################################################### 
labeled_coins, _ = ndi.label(fill_coins) # line 146
image_label_overlay = label2rgb(labeled_coins, image=coins) 

fig, axes = plt.subplots(1,2, figsize=(14,6), sharey=True)
axes[0].imshow(coins, cmap=plt.cm.gray, interpolation='nearest')
axes[0].contour(fill_coins, [0.5], linewidths=1.2, colors='y')
axes[1].imshow(image_label_overlay, interpolation='nearest')
axes[0].axis('off'), axes[1].axis('off')
plt.show()   

#######################################################
# Example 5. Felzenszwalb
######################################################### 
from matplotlib.colors import LinearSegmentedColormap

for imfile in ['Sohn.png','Mountain.png','monalisa.jpg','butterfly.jpg']:
    img = img_as_float(imread(imfile)[::2,::2,:3])
    segments_fz = segm.felzenszwalb(img, scale=100, sigma=0.5, min_size=400)
    borders = segm.find_boundaries(segments_fz)
    unique_colors = np.unique(segments_fz.ravel())
    segments_fz[borders] = -1

    colors = [np.zeros(3)]
    for color in unique_colors:
        colors.append(np.mean(img[segments_fz == color], axis=0))
        
    cm = LinearSegmentedColormap.from_list('pallete',colors, N=len(colors))

    plt.figure(figsize=(20,10))
    plt.subplot(121), plot_image(img, 'Original')
    plt.subplot(122)
    plot_image(segments_fz, 'Felzenszwalb Segmentation',cmap=cm)
    plt.show()    
    
# Scale paramete
img = imread('Mountain.jpg')[::2,::2,:3] # ['Sohn.png','Mountain.png','monalisa.jpg','butterfly.jpg']:
plt.figure(figsize=(10,10))
plt.subplot(321), plt.imshow(img), plt.axis('off'), plt.title('Fish',size=20)
for i,scale in enumerate([50,100,200,400,800]):                     
    segments_fz = segm.felzenszwalb(img, scale=scale, sigma=0.5, min_size=200) 
    seg = segm.mark_boundaries(img, segments_fz, color=(0,1,0))
    plt.subplot(3,2,i+2),  plot_image(seg, 'scale='+str(scale))
plt.suptitle('Felzenszwalbs Segmentation',size=30)
plt.tight_layout(rect=[0,0.03,1,0.95])
plt.show() 

