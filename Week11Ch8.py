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
import cv2

from skimage.io import imread
from skimage.color import rgb2gray, label2rgb
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.transform import warp, AffineTransform
from skimage.util import img_as_float
from skimage.exposure import rescale_intensity
from skimage.measure import ransac
from skimage.draw import ellipse
from skimage import data

def plot_image(image,title, cmap='gray'):
    plt.imshow(image, cmap=cmap), plt.title(title, size=20), plt.axis('off')
def plot_ax_image(ax,image,title,cmap='gray',size=20):
    ax.imshow(image,cmap=cmap), ax.set_title(title), ax.axis('off')   
    
###################################################
# Example 6. SLIC
###################################################
import skimage.segmentation as segm

img = imread('fish.jpg')[::2,::2,:3]
plt.figure(figsize=(15,10))
for i,compactness in enumerate([0.1,1,10,100]):
    segments_slic = segm.slic(img, n_segments=10, compactness=compactness, sigma=1)
    seg = segm.mark_boundaries(img, segments_slic, color=(0,1,0))
    plt.subplot(2,2,i+1), plot_image(seg,'Compactness='+str(compactness))
plt.suptitle('SLIC',size=30), plt.tight_layout(rect=[0,0.03,1,0.95])
plt.show() 

##################################################
# Example 7. RAG (Region Adjacency Graph) 
##################################################
from skimage import graph                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        

def _weight_mean_color(graph, src, dst, n):
    diff = graph.nodes[dst]['mean color']-graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return{'weight':diff}
def merge_mean_color(graph, src, dst):
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color']  = (graph.nodes[dst]['total color']/
                                      graph.nodes[dst]['pixel count'])
labels = segm.slic(img, compactness=50, n_segments=10)
g = graph.rag_mean_color(img, labels)
labels2 = graph.merge_hierarchical(labels, g, thresh=35, rag_copy=False,
             in_place_merge=True, merge_func=merge_mean_color, weight_func=_weight_mean_color)
out = label2rgb(labels2, img, kind='avg')
out = segm.mark_boundaries(out, labels2, (0,0,0))

plt.figure(figsize=(20,10))
plt.subplot(121), plt.imshow(img), plt.axis('off')
plt.subplot(122), plt.imshow(out), plt.axis('off')
plt.show() 

##################################################
# Example 8. QuickShift
##################################################
plt.figure(figsize=(12,12))
i=1
for max_dist in [5,500]:
    for ratio in [0.1, 0.9]:
        segments_quick = segm.quickshift(img, kernel_size=3, 
                                    max_dist=max_dist, ratio=ratio)
        boundary = segm.mark_boundaries(img, segments_quick, color=(0,1,0))
        title = 'Maximum distance is '+str(max_dist)+',ratio = '+str(ratio)
        plt.subplot(2,2,i), plot_image(boundary, title)
        i += 1
plt.suptitle('Quickshift',size=30)
plt.tight_layout(rect=[0,0.03,1,0.95])
plt.show()        

##################################################
# Example 9. Compact Watershed
##################################################
from scipy import ndimage as ndi
#from skimage import morphology
gradient = ndi.sobel(rgb2gray(img))
plt.figure(figsize=(18,15))
i=1
for markers in [20,100]:
    for compactness in [0.01, 0.001, 0.0001]:
        segments_watershed = segm.watershed(gradient, markers=markers,
                                      compactness=compactness)
        boundary = segm.mark_boundaries(img, segments_watershed, color=(0,1,0))
        title = 'Markers = '+str(markers)+', Compactness='+str(compactness)
        plt.subplot(2,3,i), plot_image(boundary,title)
        i += 1
plt.suptitle('Compact watershed',size=30)
plt.tight_layout(rect=[0,0.03,1,0.95])
plt.show()        
        
###############################################
# Example 10. SimpleITK를 사용한 영역 확장
##################################################
import SimpleITK as sitk

def show_image(img, title=None):
    nda = sitk.GetArrayViewFromImage(img)
    plt.imshow(nda,cmap='gray'), plt.axis('off')
    if (title): plt.title(title, size=30)
    
img = 255*rgb2gray(imread('mri.jpg'))
img_T1 = sitk.GetImageFromArray(img)    
img_T1_255 = sitk.Cast(sitk.RescaleIntensity(img_T1),sitk.sitkUInt8)

seed = (100,80)

plt.figure(figsize=(20,30))
plt.subplot(421)
show_image(img_T1, "Original")
plt.scatter(seed[0], seed[1], color='red', s=50)

i = 1
for upper in [80,82,84]:
    seg1 = sitk.ConnectedThreshold(img_T1, seedList=[seed], 
                                   lower=20, upper=upper)
    seg2 = sitk.LabelOverlay(img_T1_255, seg1)
    
    plt.subplot(4,2,2*i+1), show_image(seg1, "Region Growing [20,"+str(upper)+"]")
    plt.scatter(seed[0], seed[1], color='red', s=50)
    plt.subplot(4,2,2*i+2), show_image(seg2, "Connected Threshold")
    plt.scatter(seed[0], seed[1], color='red', s=50)
    i += 1
    
#########################################################
# Example 11.능동 윤곽선    
#########################################################
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage import data

img = data.astronaut()
img_gray = rgb2gray(img)
s = np.linspace(0, 2*np.pi, 400)
x = 220 + 100*np.cos(s)
y = 100 + 100*np.sin(s)
init = np.array([y,x]).T

i=1
plt.figure(figsize=(20,20))
for max_it in [20,30,50,200]:
    snake = active_contour(gaussian(img_gray,3,preserve_range=False), 
                           init, alpha=0.015,
                           beta=10, gamma=0.001,  max_num_iter=max_it)
    plt.subplot(2,2,i), plt.imshow(img)
    plt.plot(init[:,1], init[:,0], '--b', lw=3)
    plt.plot(snake[:,1], snake[:,0],'--r', lw=3)
    plt.axis('off'), plt.title('max_iteration='+str(max_it),size=20)
    i += 1
plt.tight_layout(),  plt.show()   

#########################################################
# Example 12.Morphological Snake : Chan Vese  
#########################################################
from skimage.segmentation import morphological_chan_vese, morphological_geodesic_active_contour
from skimage.segmentation import inverse_gaussian_gradient, checkerboard_level_set

def store_evolution_in(lst):
    # 지정된 리스트에 레벨 세트의 진화를 저장하는 콜백함수를 반환
    def _store(x):
        lst.append(np.copy(x))
        
    return _store

# 모폴리지 ACWE
image = imread('Lenna.png')
image_gray = rgb2gray(image[:,:,:3])
init_lvl_set = checkerboard_level_set(image_gray.shape, 6) # 초기 레벨 설정

evolution = []
callback = store_evolution_in(evolution)
lvl_set = morphological_chan_vese(image_gray, 30, smoothing=3,
            init_level_set=init_lvl_set, iter_callback=callback) 

fig, axes = plt.subplots(2,2,figsize=(8,6))
axes = axes.flatten()
axes[0].imshow(image, cmap='gray'), axes[0].set_axis_off()
axes[0].contour(lvl_set, [0.5], colors='g')
axes[0].set_title("Morphological ACWE segmentation",fontsize=12)

axes[1].imshow(lvl_set, cmap='gray'), axes[1].set_axis_off()
axes[1].contour(evolution[5], [0.5], colors='r')
axes[1].set_title("Morphological ACWE evolution 5", fontsize=12)

axes[2].imshow(lvl_set, cmap='gray'), axes[2].set_axis_off()
axes[2].contour(evolution[10], [0.5], colors='r')
axes[2].set_title("Morphological ACWE evolution 10", fontsize=12)

axes[3].imshow(lvl_set, cmap='gray'), axes[3].set_axis_off()
axes[3].contour(evolution[-1], [0.5], colors='r')
axes[3].set_title("Morphological ACWE evolution Final", fontsize=12)
 
#########################################################
# Example 13.Morphological Snake : Geodesic Active Contour  
#########################################################
# 모톨로지 GAC
image = imread('fish66.png')
image_gray = rgb2gray(image[:,:,:3])
gimage = inverse_gaussian_gradient(image_gray)
init_lvl_set = np.zeros(image_gray.shape, dtype=np.int8)
init_lvl_set[5:-5, 5:-5]=1
#init_lvl_set = checkerboard_level_set(image_gray.shape, 1)

evolution = []
callback = store_evolution_in(evolution)
lvl_set = morphological_geodesic_active_contour(gimage, 400, init_lvl_set,
           smoothing=1, balloon=-1, threshold=0.7, iter_callback=callback)                                     

fig, axes = plt.subplots(2,2,figsize=(8,6))
axes = axes.flatten()
axes[0].imshow(image, cmap='gray'), axes[0].set_axis_off()
axes[0].contour(lvl_set, [0.5], colors='g')
axes[0].set_title("Morphological GAC segmentation",fontsize=12)

axes[1].imshow(gimage, cmap='gray'), axes[1].set_axis_off()
#contour = axes[1].contour(evolution[10],[0.5],colors='g')
axes[1].set_title("Inverse Gaussian Gradient",fontsize=12)

# axes[2].imshow(init_lvl_set, cmap='gray'), axes[2].set_axis_off()
# #contour = axes[2].contour(evolution[20],[0.5],colors='g')
# axes[2].set_title("Init_level_set",fontsize=12)

axes[2].imshow(lvl_set, cmap='gray'), axes[2].set_axis_off()
contour = axes[2].contour(evolution[50],[0.5],colors='g')
axes[2].set_title("Morphological GAC evolution 50",fontsize=12)

axes[3].imshow(lvl_set, cmap='gray'), axes[3].set_axis_off()
contour = axes[3].contour(evolution[-1],[0.5],colors='g')
axes[3].set_title("Morphological GAC evolution",fontsize=12)

fig.tight_layout()
plt.show()

####################################################33
# Example 14. Grabcut
#########################################################
# 그래프 이론의 max-flow min-cut 알고리즘 이용
# 영상의 매칭에서 전경을 추출하는 상호작용 Interactive 분할
# 사용자가 입력영상에서 전경 영역을 대략 지정함으로써 힌트 제공
# 전경 영역 주위에 주로 사각형을 그림
# 최상의 결과를 엉기 위해 영상을 반복적으로 분할
# 원하는 분할이 아닌 경우 : 전경과 배경이 뒤바뀜
# 잘못 분할된 화소 주위에 약간의 스트로크를 해서 미세하게 수정=> 다음 반복에서 나은 결과
# 마스크 영상에 플래그 사용
# 확실한 배경/전경(0/1), 가능한 배경/전경(2/3)
# 마스크 영상 0으로 초기화
# cv2.grabCut()의 mode 인수에 
# cv2.GC_INIT_WITH_RECT 지정 : 사각형 안 1, 사각형 밖 0  
#   grabCut 이 후, 마스크 픽셀 값 0 또는 2 => 0으로 지정,  1또는 3 => 1로 지정 :  mask2
#   입력영상과 곱하여 최종 결과
# cv2.GC_INIT_WITH_MASK 지정 : 위 결과가 안 좋을 경우 마우스로 마스크 직접 강제 지정
#   grabCut 이 후, 마스크 픽셀 값 0 또는 2 => 0으로 지정,  1또는 3 => 1로 지정 :  mask2
#   입력영상과 곱하여 최종 결과 

################################################
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

src = cv2.imread('whale.jpg')
# 사각형 지정을 통한 초기 분할
mask = np.zeros(src.shape[:2], np.uint8) # 마스크
bgdModel = np.zeros((1, 65), np.float64) # 배경 모델 무조건 1행 65열, float64
fgdModel = np.zeros((1, 65), np.float64) # 전경 모델 무조건 1행 65열, float64

rc = cv2.selectROI(img=src)

# RECT는 사용자가 사각형 지정. 이 값에서 계속 업데이트
cv2.grabCut(src, mask, rc, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)

# mask 4개 값을 2개로 변환
mask = np.where((mask == 0) | (mask == 2), 0, 1).astype('uint8')
dst = src * mask[:, :, np.newaxis]

# 초기 분할 결과 출력
cv2.imshow('dst', dst)

src2 = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
dst2 = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

plt.figure()
plt.subplot(221), plt.imshow(src2), plt.axis('off'), plt.title('RGB')
plt.subplot(223), plt.imshow(dst2), plt.axis('off'), plt.title('Segmentation')
plt.subplot(222), plt.imshow(src), plt.axis('off'), plt.title('BGR')
plt.subplot(224), plt.imshow(1-dst2), plt.axis('off'), plt.title('1-Segmentation')
plt.suptitle('GrabCut',size=30)
plt.tight_layout(rect=[0,0.03,1,0.95])
plt.show()

# 마우스 이벤트 처리 함수 등록
def on_mouse(event, x, y, flags, param):
    global mask,dst
    if event == cv2.EVENT_LBUTTONDOWN: # 왼쪽 버튼은 전경
        cv2.circle(dst, (x, y), 3, (255, 0, 0), -1) # 파랑색 색칠
        cv2.circle(mask, (x, y), 3, cv2.GC_FGD, -1) # 마스크에 전경 강제 지정
        cv2.imshow('dst', dst)
    elif event == cv2.EVENT_RBUTTONDOWN: # 오른쪽 버튼은 배경
        cv2.circle(dst, (x, y), 3, (0, 0, 255), -1) # 빨강색 원
        cv2.circle(mask, (x, y), 3, cv2.GC_BGD, -1) # 마스크에 배경 강제 지정
        cv2.imshow('dst', dst)
        
    elif event == cv2.EVENT_MOUSEMOVE: # 마우스 움직임
        if flags & cv2.EVENT_FLAG_LBUTTON: # 왼쪽 누르고 움직이면 전경
            cv2.circle(dst, (x, y), 3, (255, 0, 0), -1)
            cv2.circle(mask, (x, y), 3, cv2.GC_FGD, -1)
            cv2.imshow('dst', dst)
        elif flags & cv2.EVENT_FLAG_RBUTTON: # 오른쪽 누르고 움직이면 배경
            cv2.circle(dst, (x, y), 3, (0, 0, 255), -1)
            cv2.circle(mask, (x, y), 3, cv2.GC_BGD, -1)
            cv2.imshow('dst', dst)

cv2.imshow('dst',dst)
cv2.setMouseCallback('dst', on_mouse)

while True:
    key = cv2.waitKey(10) & 0xFF
    if key == 13:
        cv2.grabCut(src, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK) # 마스크 초기화
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        dst = src * mask2[:, :, np.newaxis]
        cv2.imshow('dst', dst)

    elif key == 27:
        break

cv2.destroyAllWindows()

plt.figure()
plt.subplot(121), plt.imshow(dst), plt.axis('off')
plt.subplot(122), plt.imshow(1-dst), plt.axis('off')
plt.suptitle('After MouseCallback')
plt.tight_layout(rect=[0,0.03,1,0.95])
plt.show()
