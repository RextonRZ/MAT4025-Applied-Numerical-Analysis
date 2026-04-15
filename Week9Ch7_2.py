#### Ch 7.영상 특징과 서술자 추출
# 특징 검출기 : 해리스 코너 Harris Corner, SIFT,  HOG
# scikit-image 및 python-opencv (cv2) 라이브러리 함수 사용
# 영삼 매칭 및 객체 검출
# * 특징 검출기와 서술자를 비교하여 영상에서 특징/서술자 검출
# * 해리스 코너 검출기와 영상 매칭에서 해리스 코너 특징 적용
# * LoG, DoG, DoH 가 있는 블롭 blob 검출기
# * HoG 특징 추출
# * SIFT, ORB, BRIEF 특징 및 영상 매칭에서의 응용
# * 유사-하르 Haar-like 특징과 얼굴검출에서의 응용
##################################################
# 특징 검출기와 서술자
# 특징 feature : 영상 처리 작업과 관련된 주요 점들 points의 그룹 또는 정보
#                영상의 추상적이고 보다 일반적인 종종 강건한 표현을 생성
# 특징 검출기 feature detector/ 추출기 extractor
#          : 코너, 로컬 최대/최소, 영상의 특징을 검출/추출 등을 기반으로 영상에서
#            관심점 그룹을 선택하는 알고리즘
# 서술자 : 특징/관심 지점(예:HOG 특징)으로 영상을 나타내기 위한 값 모음
# 특징 추출은 영상을 특징 서술자 집합으로 변환하는 연산
#            특별한 형태의 차원감소
# 로컬 특징은 일반적으로 관심 점과 그것의 서술자로 구성
# 전체 영상의 전역 특징은 종종 바람직하지 않다.
# 실제적으로 영상을 코너, 에지, 블롭 같은 관심 영역에 해당하는 로컬 특징으로 묘사
#     밝기 및 그래디언트 같은 특정 광도 속성(Photometric property)의 로컬분포를 
#        포착하는 서술자
# 로컬 특징의 일부 속성
#     반복적, 각 영역의 동일한 점들을 독립적 검출
#     변환, 회전, 스케일 (어파인 변환)에 불변
#     잡음/블러/폐색(Occulsion)/클러터(Clutter)/조명 변화 에 대해 견고
#     관심있는 구조, 독특성
# 로컬 특징을 사용하는 곳: 영상 등록, 영상 매칭, 영상 스티칭(파노라마)
#                        객체 인식/검출
# Python-opencv (cv2 라이브러리)를 사용



##########################################
## 실습 1. 해리스 코너 검출기
# 윈도우가 영상 내에서 위치를 변경함에 따라 윈도우 내의 밝기 변화를 탐색
# 모든 방향에서 밝기 값이 크게 변경 cf. 에지: 한 방향에서만 급격히 변화
# 윈도우가 코너에서 어느 방향으로 움직이던지 밝기 값이 크게 변화
# 회전에는 불변, 크기에는 의존
#########################################
# Scikit-image 이용
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread #, imshow, show
from skimage.color import rgb2gray
#from scipy import signal, ndimage

image = rgb2gray(imread("Chess.png")[:,:,:3])
image2 = rgb2gray(imread("images.jpg"))
plt.figure()
plt.subplot(121), plt.imshow(image,cmap="gray")
plt.subplot(122), plt.imshow(image2,cmap="gray")
plt.show()
# square = np.zeros([10,10])
# square[2:8,2:8] = 1

from skimage.feature import corner_harris, corner_subpix, corner_peaks
plt.figure()
plt.subplot(121), plt.title('Checkerboard',size=20)
cp = corner_peaks(corner_harris(image), min_distance=5)
plt.imshow(image,cmap='gray'), plt.plot(cp[:,0],cp[:,1],'r*') #, plt.show() 

plt.subplot(122), plt.title('Football',size=20)
cp = corner_peaks(corner_harris(image2), min_distance=20)
plt.imshow(image2,cmap='gray'), plt.plot(cp[:,0],cp[:,1],'r*')
plt.show() 

# coordinates = corner_harris(image) #, k=0.001)
# image[coordinates>0.28] = [255,0,0,255] #01*coordinates.max()] = [255,0,0,255]
# plt.imshow(image,cmap="gray")

from skimage import data
from skimage.transform import warp, AffineTransform
from skimage.draw import ellipse
# Sheared checkerboard
tform = AffineTransform(scale=(1.3, 1.1), rotation=1, shear=0.7, translation=(110, 30))
image = warp(data.checkerboard()[:90, :90], tform.inverse, output_shape=(200, 310))
# Ellipse
rr, cc = ellipse(160, 175, 10, 100)
image[rr, cc] = 1
# Two squares
image[30:80, 200:250] = 1
image[80:130, 250:300] = 1

coords = corner_peaks(corner_harris(image), min_distance=5, threshold_rel=0.02)
coords_subpix = corner_subpix(image, coords, window_size=13)

fig, ax = plt.subplots()
ax.imshow(image, cmap=plt.cm.gray)
ax.plot(coords[:, 1], coords[:, 0], color='cyan', marker='o', linestyle='None', markersize=6)
ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)
ax.axis((0, 310, 200, 0))
plt.show()

####################################
## 서브화소 정확도 사용
##################################
# 코너를 최고 정확도로 찾고자 할 때 subpix 사용
# scikit-image.corner_subpix() : 검출된 코너가 서브 화소 정확도로 정밀해짐
# 통계적 테스트를 사용해서 corner_peaks() 함수로 이전에 계산된 코너지점을
#    허용/거부할지를 결정
#함수가 코너를 검색하는 데 사용할 인접 영역(윈도우)의 크기를 정의해야 함
im = imread("pyramid.jpg")
img = rgb2gray(im)
coordinates = corner_harris(img,k=0.001)
coordinates[coordinates>0.03*coordinates.max()] = 255
corners = corner_peaks(coordinates)
subpixs = corner_subpix(img, corners, window_size = 11)

plt.figure(figsize=(20,20))
plt.subplot(121), plt.imshow(coordinates,cmap='inferno')
plt.plot(subpixs[:,1],subpixs[:,0],'r.',marker=5)
plt.axis('off')
plt.subplot(122), plt.imshow(im,interpolation='nearest')
plt.plot(corners[:,1],corners[:,0],'bo', marker=3)
plt.plot(subpixs[:,1],subpixs[:,0],'r+',marker=5)
plt.axis('off')
plt.tigt_layout(), plt.show()

#############################
## 응용 프로그램 : 영상 매칭
#############################
# 영상에서 관심 점들을 검출하면, 동일한 객체의 여러 영상들에서 그 점을 매칭 시키는 방법
# 두 영상을 매칭시키는 일반적인 방법
#  - 관심 점(point of interest) 계산 : 해리스 코너 검출기 사용
#  - 각 키 점(Key point) 주변 영역(Window) 고려
#  - 이 영역에서 각 영상 각 키 점들에 대한 로컬 특징 서술자 계산하고 표준화
#  - 두 영상에 계산된 로컬 서술자를 매칭 : 유클리드 거리 이용
##############################################################
## 실습 3. Ransack 알고리즘과 Harris Conrner 특징을 사용한 강건한 영상매칭
############################################################
# 영상을 어파인 변환된 버전을 사용하여 매칭, 마치 영상을 다른 관점에서 찍은 듯
# 영상 매칭 알고리즘
#  1. 두 영상에서 관심 점들 또는 해리스 코너를 계산
#  2. 점들 주위 작은 공간 고려. 점들의 일치 정도는 제곱 차의 가중치 합을 사용.
#     측정값이 견고하지 않으면 약간의 관점 변경으로 이용
#  3. 일치함이 발견되면 소스와 대응된 목적지 좌표 집합을 얻음. 
#      두 영상 사이의 기하학적 변환 추정하는 데 사용
#  4. 좌표를 가지고 매개 변수를 간단하게 추정하는 것으로는 충분하지 않음. 잘못 가능성 고려
#  5. RANSAC(RANdom SAmple Consensus) 알고리즘. (매개변수를 견고하게 추정)
#     점들을 inlier와 outlier로 분류, outlier를 무시하면서 모델을 inlier에 맞추어 
#      어파인 변환과 매칭하는 것을 찾음 
#temple = rgb2gray(imread('temple.jpg'))

# generate synthetic checkerboard image and add gradient for the later matching
from skimage import filters, feature, img_as_float
from skimage.exposure import rescale_intensity
checkerboard = img_as_float(data.checkerboard())
#checkerboard = img
img_orig = np.zeros(list(checkerboard.shape) + [3])
img_orig[..., 0] = checkerboard
gradient_r, gradient_c = np.mgrid[0 : img_orig.shape[0], 0 : img_orig.shape[1]] / float(
    img_orig.shape[0]
)
img_orig[..., 1] = gradient_r
img_orig[..., 2] = gradient_c
img_orig = rescale_intensity(img_orig)
img_orig_gray = rgb2gray(img_orig)

# warp synthetic image
tform = AffineTransform(scale=(0.9, 0.9), rotation=0.2, translation=(20, -10))
img_warped = warp(img_orig, tform.inverse, output_shape=(200, 200))
img_warped_gray = rgb2gray(img_warped)

# extract corners using Harris' corner measure
coords_orig = corner_peaks(
    corner_harris(img_orig_gray), threshold_rel=0.001, min_distance=5
)
coords_warped = corner_peaks(
    corner_harris(img_warped_gray), threshold_rel=0.001, min_distance=5
)

# determine sub-pixel corner position
coords_orig_subpix = corner_subpix(img_orig_gray, coords_orig, window_size=9)
coords_warped_subpix = corner_subpix(img_warped_gray, coords_warped, window_size=9)


def gaussian_weights(window_ext, sigma=1):
    y, x = np.mgrid[-window_ext : window_ext + 1, -window_ext : window_ext + 1]
    g = np.zeros(y.shape, dtype=np.double)
    g[:] = np.exp(-0.5 * (x**2 / sigma**2 + y**2 / sigma**2))
    g /= 2 * np.pi * sigma * sigma
    return g


def match_corner(coord, window_ext=5):
    r, c = np.round(coord).astype(np.intp)
    window_orig = img_orig[r-window_ext:r+window_ext+1, c-window_ext:c+window_ext+1,:]

    # weight pixels depending on distance to center pixel
    weights = gaussian_weights(window_ext, 3)
    weights = np.dstack((weights, weights, weights))

    # compute sum of squared differences to all corners in warped image
    SSDs = []
    for cr, cc in coords_warped:
        window_warped = img_warped[
            cr - window_ext : cr + window_ext + 1,
            cc - window_ext : cc + window_ext + 1,
            :]
        SSD = np.sum(weights * (window_orig - window_warped) ** 2)
        SSDs.append(SSD)

    # use corner with minimum SSD as correspondence
    min_idx = np.argmin(SSDs)
    return coords_warped_subpix[min_idx]


# find correspondences using simple weighted sum of squared differences
src = []
dst = []
for coord in coords_orig_subpix:
    if any(coord) and len(coord)>0 and not all(np.isnan(coord)):
        src.append(coord)
        dst.append(match_corner(coord))
src = np.array(src)
dst = np.array(dst)


# estimate affine transform model using all coordinates
model = AffineTransform()
model.estimate(src, dst)

# robustly estimate affine transform model with RANSAC
from skimage.measure import ransac
model_robust, inliers = ransac(
    (src, dst), AffineTransform, min_samples=3, residual_threshold=2, max_trials=100
)
outliers = inliers == False


# compare "true" and estimated transform parameters
print("Ground truth:")
print(
    f'Scale: ({tform.scale[1]:.4f}, {tform.scale[0]:.4f}), '
    f'Translation: ({tform.translation[1]:.4f}, '
    f'{tform.translation[0]:.4f}), '
    f'Rotation: {-tform.rotation:.4f}'
)
print("Affine transform:")
print(
    f'Scale: ({model.scale[0]:.4f}, {model.scale[1]:.4f}), '
    f'Translation: ({model.translation[0]:.4f}, '
    f'{model.translation[1]:.4f}), '
    f'Rotation: {model.rotation:.4f}'
)
print("RANSAC:")
print(
    f'Scale: ({model_robust.scale[0]:.4f}, {model_robust.scale[1]:.4f}), '
    f'Translation: ({model_robust.translation[0]:.4f}, '
    f'{model_robust.translation[1]:.4f}), '
    f'Rotation: {model_robust.rotation:.4f}'
)

# visualize correspondence
from skimage.feature import plot_matches
fig, ax = plt.subplots(nrows=2, ncols=1)
plt.gray()
inlier_idxs = np.nonzero(inliers)[0]
plot_matches(
    ax[0],
    img_orig_gray,
    img_warped_gray,
    src,
    dst,
    np.column_stack((inlier_idxs, inlier_idxs)),
    matches_color='b',
)
ax[0].axis('off'), ax[0].set_title('Correct correspondences')

outlier_idxs = np.nonzero(outliers)[0]
plot_matches(
    ax[1],
    img_orig_gray,
    img_warped_gray,
    src,
    dst,
    np.column_stack((outlier_idxs, outlier_idxs)),
    matches_color='r',
)
ax[1].axis('off'), ax[1].set_title('Faulty correspondences')
plt.show()

###########################################
## 실습 4. LoG, DoG, DoH를 사용한 블롭 검출기
###########################################
# Blob : 어두운 영역에 있는 밝은 영역 또는 밝은 영역에 있는 어두운 영역
# LoG : Laplacian of Gaussian
#  에지 검출을 위해 LoG필터의 제로 크로싱을 하였음
#  찾고자 하는 작은 탬플릿 영상을 영상의 모든 지역과 비교
#  스케일 공간 개념으로 LoG의 3차원(위치+스케일) 극한을 검색하여 스케일 불변 영역 찾을 수도 있음
#  LoG필터의 sigma이 블롭의 스케일과 매칭되면, 라플라스 응답의 크기는 블롭의 중심에서 최대
#  LoG 컨볼루션된 영상이 점진적으로 증가하는 sigma에 대해 계산되어 큐브cube에 쌓인다.
#  블롭은 이 큐브의 로컬 최대값
#  어두운 배경의 밝은 블롭 감지
#  정확하지만 속도는 느리다. 특히 더 큰 블롭을 사용하는 경우
# DoG : Difference of Gausssina
#  LoG 접근법은 DoG 접근법에 의해 근사되며 따라서 더 빠르다.  
#  sigma가 증가하면서 영상은 평활화되고, 연속된 두 평활화 영상사이의 차가 큐브에 쌓인다.
#  어두운 배경에서 밝은 블롭을 다시 검출
#  큰 블롭 검출이 여전히 시간이 많이 소요되지만, LoG보다 빠르고 정확성은 떨어진다. 
# DoH : Determinant of Hessian
#  가장 빠르다.
#  영상의 Hessian에서 최대값을 계산하여 블롭을 검출
#  블롭의 크기는 검출 속도에 영향을 주지 않는다.
#  작은 블롭은 정확하게 검출되지 않는다.
################################################################################
from skimage.feature import blob_dog, blob_log, blob_doh
im = imread('butterfly.jpg')
img = rgb2gray(im)

log_blobs = blob_log(img, max_sigma=30, num_sigma=10, threshold=.1)
dog_blobs = blob_dog(img, max_sigma=30, threshold=.1)
log_blobs[:,2] = np.sqrt(2) * log_blobs[:,2]
dog_blobs[:,2] = np.sqrt(2) * dog_blobs[:,2]
doh_blobs = blob_doh(img, max_sigma=30, threshold=.005)
list_blobs = [log_blobs, dog_blobs, doh_blobs]

colors, titles = ['blue', 'lime', 'red'], ['LoG', 'DoG', 'DoH']
fig, axes = plt.subplots(2,2,figsize=(20,20))
axes = axes.ravel()
axes[0].imshow(im, interpolation='nearest')
axes[0].set_title('Original image',size=30), axes[0].set_axis_off()
for i,blobs in enumerate(list_blobs):
    axes[i+1].imshow(im), axes[i+1].set_title('Blobs with'+titles[i],size=30)
    for (y,x,row) in blobs:
        col = plt.Circle((x,y), row, color=colors[i], linewidth=2, fill=False)
        axes[i+1].add_patch(col), axes[i+1].set_axis_off()
plt.tight_layout(), plt.show()        

###########################################################
## 실습 5. HOG : Histogram of Oriented Gradients
#########################################################
# 객체 검출을 위한 대표적 서술자 HOG
# HOG 서술자 계산 법
#  1. 원하는 경우 영상을 전체적으로 정규화
#  2. 수평 및 수직 그래디언트 계산
#  3. 그래디언트 히스토그램 계산
#  4. 블록 Block 기준으로 정규화
#  5. 특징 서술자 벡터로 전개
###############################################################
from skimage.feature import hog

im = rgb2gray(imread('cameraman.jpg'))
fd, hog_image = hog(im, orientations=8, pixels_per_cell=(16,16),\
                    cells_per_block=(1,1), visualize=True)
hog_image_rescaled = rescale_intensity(hog_image, in_range=(0,10))
print(im.shape, len(fd)) 

fig, axes = plt.subplots(2,2,figsize=(15,15))
axes = axes.ravel()
axes[0].axis('off'), axes[0].imshow(im, cmap='gray')
axes[0].set_title('Input image',size=20)   
axes[1].axis('off'), axes[1].imshow(hog_image_rescaled, cmap='gray')
axes[1].set_title('Histogram of Oriented Gradients, r=1',size=20)
axes[2].axis('off'), axes[2].imshow(hog_image_rescaled**0.8, cmap='gray')
axes[2].set_title('Histogram of Oriented Gradients, r=0.8',size=20)
axes[3].axis('off'), axes[3].imshow(hog_image_rescaled**0.3, cmap='gray')
axes[3].set_title('Histogram of Oriented Gradients, r=0.5',size=20)
plt.tight_layout(), plt.show()

###############################################################3
## 실습 6. SIFT : Scale-Invariant Feature Transform
###############################################################
# 스케일 불변 특징 변환
# 서술자, 영상 영역에 대한 대체 표현
# 간단한 코너 검출기는 일치시킬 영상이 본질적으로 비슷랗 때(스케일, 방향) 잘 작동
# 영상의 스케일이나 방향/회전이 다른 경우, SIFT를 사용하여 매칭
# SIFT는 스케일 불변 뿐 아니라 영상의 회전, 조면 및 시점이 변경될 때에도 여전히 우수
# 이동, 회전, 스케일 및 기타 영상 매개 변수에 영향을 받지 않는 로컬 특징좌표로 영상을 변환
# sIFT 알고리즘
#  * 스케일 공간 극한값 검출(Scale-space extrema detection) : 다중 스케일  및
#      영상 위치를 검색, 위치 및 특성 스케일은 DoG 검출기 사용
#  * 키포인트의 지역화(keypoint localization) : 안정성 기준에 따라 키포인트 선택,
#      낮은 콘트라스트와 에지 키포인트를   제거하여 강한 관심점들만 유지
#  * 방향 지정(Orientation Assignment) : 각 키포인트 영역에 대해 최상의 방향을 
#      계산하여 매칭의 안정성에 기여
#  * 키포인트 서술자 계산(Keypoint descriptor computation) : 각 키포인트 영역을
#      잘 설명하기 위해 선택된 스케일과 회전에서 로컬 영상 그래디언트 사용
# SIFT는 다음에 대해 강건하다.
#  * 조명의 작은 변화 : 그래디언트와 정규화로 인한 변화
#  * 자세 : 뱡향 히스토그램으로 인한 작은 어파인 변화
#  * 스케일 : DoG
#  * 클래스 내 변화 : 히스토그램으로 인한 작은 변화  
###################################################################
# Opencv와 opencv-contrib 사용
# SIFT 객체 생성
# detect() 메서드를 시요앟여 영상의 키포인트 계산
# 모든 키포인트는 특별한 특징이며 여러가지 속성이 있다.
# x,y, 각도(방향), 응답(키포인트 밝기), 의미있는 이웃의 크기 등
# cv2의 drawKeyPoints() 함수를 사용하여 검출된 키포인트 주위에 작은 원을 그림
# 이 함수에 cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS 를 적용하염 키포인트 방향과 원이 그려짐
# 키포인트와 서술자를 함께 계산 : detectAndCompute() 함수 사용  
#####################################################################  
import cv2     
im = cv2.imread('monalisa.jpg')
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()

kp = sift.detect(gray,None)
flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
im1 = cv2.drawKeypoints(im,kp, None,flags=flags)

fig, axes = plt.subplots(1,2,figsize=(15,10))
axes = axes.ravel()
axes[0].axis('off'), axes[0].imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
axes[0].set_title('Original',size=20)
axes[1].axis('off'), axes[1].imshow(cv2.cvtColor(im1,cv2.COLOR_BGR2RGB))
axes[1].set_title('Image with SIFT Keypoints',size=20)
plt.show()

######################################################
## 응용프로그램 - BRIEF, SIFT 및 ORB를 사용한 영상 매칭
###########################################################
##################################################################
# 실습 7. BRIEF를 사용한 영상 매칭
###############################################################
# BRIEF : Binary Robust Independent Elementary Features : 짧은 이진 서술자
# 서술자 => 영상 매출과 객체 검출에 사용됨
# BRIEF 서술자는 상대적으로 적은 비트 수를 가짐. 밝기 차이 테스트 집합을 사용하여 계산
# 메모리를 차지하는 공간이 적음. 해밍 거리 측정법, 효율적
# 회전 불변성은 제공하지 않음.
# 다른 스케일에서 특성을 검출하여 원하는 스케일 불변성을 얻을 수 있음
from skimage import transform
from skimage.feature import match_descriptors, BRIEF,plot_matches

img1 = rgb2gray(imread('Lenna.png')[...,:3])
affine_trans = AffineTransform(scale=(1.2,1.2), translation=(0,-100))
img2 = transform.warp(img1, affine_trans)
img3 = transform.rotate(img1,25)

coords1, coords2, coords3 = corner_harris(img1), corner_harris(img2), corner_harris(img3)
coords1[coords1>coords1.max()] = 1
coords2[coords2>coords2.max()] = 1
coords3[coords3>coords3.max()] = 1
keypoints1 = corner_peaks(coords1, min_distance=5)
keypoints2 = corner_peaks(coords2, min_distance=5)
keypoints3 = corner_peaks(coords3, min_distance=5)

extractor = BRIEF()
extractor.extract(img1, keypoints1)
keypoints1, descriptors1 = keypoints1[extractor.mask], extractor.descriptors
extractor.extract(img2, keypoints2)
keypoints2, descriptors2 = keypoints2[extractor.mask], extractor.descriptors
extractor.extract(img3, keypoints3)
keypoints3, descriptors3 = keypoints3[extractor.mask], extractor.descriptors
matches12 = match_descriptors(descriptors1, descriptors2, cross_check = True)
matches13 = match_descriptors(descriptors1, descriptors3, cross_check = True)

# plt.figure()
# plt.subplot(211), plt.imshow(descriptors1)
# plt.subplot(212), plt.imshow(descriptors2)
# plt.show()

fig, (axes1, axes2) = plt.subplots(2,1,figsize=(10,10))
plot_matches(axes1, img1, img2, keypoints1, keypoints2, matches12)
axes1.axis('off'), axes1.set_title("Original vs Scaled\Translated")
plot_matches(axes2, img1, img3, keypoints1, keypoints3, matches13)
axes2.axis('off'), axes2.set_title('Original vs Rotated')
plt.show()

##########################################################
# 실습 8. ORB
##########################################################
# ORB : Oriented FAST and Rotated BRIEF : SIFT애 대한 효율적 대안
# ORB 특징 검출과 이진 서술자 알고리즘
# 지향적 FAST 검출방법과 회전된 BRIEF 서술자
# BRIEF와 비교할 때, ORB는 스케일과 회전 불변에 더 뛰어남.
#  매칭을 위해 해밍 거리를 더 뛰어나게 사용함.
#  실시간 응용을 고려할 때, BRIEF보다 더 선호됨
from skimage.feature import ORB

img1 = rgb2gray(imread('monalisa.jpg'))
img4 = rgb2gray(imread('monalisa2.jpg'))
img2 = transform.rotate(img1,180)
affine_trans = AffineTransform(scale=(1.3,1.1), rotation=0.5, translation=(0,-200))
img3 = transform.warp(img1, affine_trans)
img4 = transform.resize(img4, img1.shape, anti_aliasing=True)

descriptor_extractor = ORB(n_keypoints=200)
descriptor_extractor.detect_and_extract(img1)
keypoints1 = descriptor_extractor.keypoints
descriptors1 = descriptor_extractor.descriptors
descriptor_extractor.detect_and_extract(img2)
keypoints2 = descriptor_extractor.keypoints
descriptors2 = descriptor_extractor.descriptors
descriptor_extractor.detect_and_extract(img3)
keypoints3 = descriptor_extractor.keypoints
descriptors3 = descriptor_extractor.descriptors
descriptor_extractor.detect_and_extract(img4)
keypoints4 = descriptor_extractor.keypoints
descriptors4 = descriptor_extractor.descriptors

matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)
matches13 = match_descriptors(descriptors1, descriptors3, cross_check=True)
matches14 = match_descriptors(descriptors1, descriptors4, cross_check=True)

fig, axe = plt.subplots(2,2,figsize=(10,15))
axe = axe.ravel()
axe[0].imshow(img1,cmap='gray'), axe[0].axis('off'), axe[0].set_title('Original',size=20)
axe[1].imshow(img2,cmap='gray'), axe[1].axis('off'), axe[1].set_title('Rotation',size=20)
axe[2].imshow(img3,cmap='gray'), axe[2].axis('off'), axe[2].set_title('Transformed',size=20)
axe[3].imshow(img4,cmap='gray'), axe[3].axis('off'), axe[3].set_title('Different Background',size=20)
plt.show()

fig, axe = plt.subplots(3,1,figsize=(10,15))
axe = axe.ravel()
plot_matches(axe[0], img1, img2, keypoints1, keypoints2, matches12)
axe[0].axis('off'), axe[0].set_title('Original vs Rotation',size=20)
plot_matches(axe[1], img1, img3, keypoints1, keypoints3, matches13)
axe[1].axis('off'), axe[1].set_title('Original vs Transformed',size=20)
plot_matches(axe[2], img1, img4, keypoints1, keypoints4, matches14)
axe[2].axis('off'), axe[2].set_title('Original vs Different background',size=20)
plt.show()

###############################################
# 실습 9. 무차별 매칭을 사용한 ORB 특징의 매칭
###############################################
# 두 개의 영상 서술자가 opencv의 무차별 매칭 brute-force matcher를 사용하여 매칭
# 한 영상의 특징 서술자가 다른 영상의 모든 특징과 매칭되고
#   거리 측도를 사용하여 가장 가까운 것이 반환된다.
# BFMatcher() 함수를 사용하여 ORB 서술자와 함께 사용하여 매칭
# 코드 블록에 의해 계산된 상위 20개의 ORB 키포인트
img1 = cv2.imread('Queiry.jpg',0)
img2 = cv2.imread('Train.jpg',0)

orb = cv2.ORB_create() # ORB 객체 생성
kp1, des1 = orb.detectAndCompute(img1,None) # 키포인트와 서술자 검색
kp2, des2 = orb.detectAndCompute(img2,None)

# BFMatcher객체 생성
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)  # 서술자 매칭
matches = sorted(matches, key = lambda x:x.distance) #거리순 정렬

# 첫 20개 매치 그리기
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=2)

plt.figure(figsize=(10,10))
plt.subplot(121), plt.imshow(img1,cmap='gray')#plt.axis('off'), 
plt.title('Query',size=20)
plt.subplot(122), plt.imshow(img2,cmap='gray')#plt.axis('off'), 
plt.title('Train',size=20)
plt.show()

plt.figure(figsize=(20,10))
plt.imshow(img3), plt.axis('off')
plt.show()

################################################
## 실습 10.SIFT 서술자의 무차별 매칭과 비율 테스트
################################################
# 두 영상 사이의 SIFT 키포인트는 최근접이웃을 식별하여 매칭
# 그러나, 잡음과 같은 요소로 인해 두번째 가까운 매칭이 첫번째보다 더 가깝게 보일 수 있다.
#   이 경우, 두번째 가까운 거리와 가장 가까운 거리의 비율이 0.8 이상이면 거부
# 이렇게 하면 잘못된 매칭의 약 90%를 제거하고, 약 5%만 매칭
# knnMatch() 함수를 사용하여 키포인트에 대해 k=2의 가장 잘 매칭된 항목을 얻음
#   비율테스트를 이용할 것
sift = cv2.xfeatures2d.SIFT_create() #SIFT 검출기 객체 생성
kp1, des1 = sift.detectAndCompute(img1,None) #SIFT 키포인트와 서술자 검색
kp2, des2 = sift.detectAndCompute(img2,None)
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)

good_matches = []
for m1, m2 in matches: # Apply ratio test
    if m1.distance < 0.75*m2.distance:
        good_matches.append([m1])
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2, good_matches[:], None, flags=2) 

plt.figure(figsize=(20,10))
plt.imshow(img3)
plt.axis('off')
plt.show() 

########################################
## 유사 하르 특징
########################################
# 객체 검출에 사용되는 매우 유용한 특징
# 적분 영상 integral images을 사용하면 일정 크기(스케일)의 유사  하르 특징을 효율적으로 계산
# 계산 속도가 빠름, 컨볼루션 커널과 같음
# 얼굴 검출 : 눈이 코, 볼, 콧등보다 더 어둡다.
########################################
# 실습 11. 유사 하르 특징 서술자
#######################################
# 5개의 다른 유사하르 특징 서술자
# scikit-image.feature 모듈의 haar_like_feature_coord, draw_haar_like_feature
#  을 이용하여 여려 유형의 Haar 특징 서술자 시각화
from skimage.feature import haar_like_feature_coord, draw_haar_like_feature

images = [np.zeros((2,2)), np.zeros((2,2)), np.zeros((3,3)), np.zeros((3,3)), np.zeros((2,2))]
feature_types = ['type-2-x','type-2-y', 'type-3-x', 'type-3-y','type-4']

fig,axes = plt.subplots(3,2,figsize=(5,7))
for axes,img,feat_t in zip(np.ravel(axes), images, feature_types):
    coordinates, _ = haar_like_feature_coord(img.shape[0], img.shape[1], feat_t)
    haar_feature = draw_haar_like_feature(img,0,0,img.shape[0],img.shape[1],\
                                          coordinates, max_n_features=1,random_state=42,\
                                          color_positive_block=(0.0,1.0,1.0),\
                                          color_negative_block=(1.0,1.0,0.0),alpha=0.8)
    axes.imshow(1-haar_feature), axes.set_title(feat_t), axes.set_axis_off()
fig.suptitle('Different Haar-like feature descriptors')
plt.axis('off'), plt.tight_layout()
plt.show()    

#########################################################
# 실습 12. 유사 하르 특징 서술자를 이용하여 얼굴 검출하기
#########################################################
# Viola-Jones 얼굴 검출 알고리즘
# 각각의 유사하르 특징은 약한 분류기일 뿐이므로 정확성을 높이기 위해 많은 유사하르 특징 필요
# 1.적분 영상 integral images을 사용하여 각 유사하르 커널의 가능한 크기와 위치에 대해
#   많은 수의 유사하르 특징들이 계산됨
# 2. AdaBoost 앙상블 분류기를 이용하여 많은 특징 중 중요한 특징 선택
# 3. 강한 분류기 학습 모델 사용
# 4. 학습된 모델 사용하여 선택한 특징으로 얼굴 영역 분류
# 영상 대부분이 비얼굴 영역미므로 먼저 윈도우가 얼굴영역인지 아닌지 검사
#   - 비얼굴 영역 : 버림
#   - 가능성 있음 :  다시 검사
# ==> 분류기의 Cascade 개념 : 방대한 양의 특징들을 적용하는 대신 
#         특징을 여러단계로 그룹화하고 하나씩 적용
#          윈도우가 첫번째 단계에서 실패하면 버림
#           통과되면 두번재 단계 적용 => 전통적인 가계학습 과 연관
# Haar-cascade 특징을 갖춘 사전 학습 분류기 사용 OpenCV로 얼굴/눈 검출
#  Opencv에서 학습기와 검출기가 함께 제공
#  얼굴, 눈, 미소에 대한 사전 학습 분류기를 사용하여 모델학습을 생략하고 검출하는 방법
#  OpenCV에는 이미 학습된 많은 모델이 포함되어 있음
#  사전학습분류기는 XML화일로 직렬화되며, OpenCV설치와 함꼐 제공(opencv/data/haarcascases 폴더)
#  얼굴 검출 : XML 분류기 로드, 입력영상을 명암도 모드로 로드, detectMultiScale() 함수 사용 
#    detectMultiScale() 매개 변수
#    scaleFactor: 각 영상 스케일에서 영상 크기를 줄이고, 스케일 피라미드를 만드는데 사용하는 양을 지정
#                 1.2 : 크기를 20% 줄임,  작을수록 검출모델과 매칭하는 크기가 더 많이 발견
#    minNeighbors : 각 후보 사각형을 유지해야하는 이웃 수 지정
#                   검출된 얼굴의 품질에 영향,  값이 클수록 검출은 적지만 품질은 높음
#    minSize,maxSize : 가능한 최소와 최대 객체의 크기, 이 값을 넘어서는 크기의 객체는 무시
# 얼굴이 발견되면 얼굴의 위치 => Rect(x,y,w,h)
#  => 얼굴 관심영역(ROI) 생성, 눈 검출, 
#  정면 얼굴, 상체 => 안경 착용/미착용 눈 검출

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

img = cv2.imread('Lenna.png')
img1 = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 얼굴 찾기
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # 눈 찾기
    roi_color = img[y:y + h, x:x + w]
    roi_gray = gray[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

# 영상 출력
plt.figure(figsize=(10,10))
plt.subplot(121), plt.imshow(img1), plt.axis('off')
plt.subplot(122), plt.imshow(img), plt.axis('off') #,cmap='gray')      
plt.show()
