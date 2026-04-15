# Ch9 Machine Learning
# Supervised Learning vs. Unsupervised Learning
# Unsupervised Learning : Clusering, PCA, Eigenface
# K-means clustering : Color quantization 
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.io import imread

# from sklearn.cluster import KMeans
# from sklearn.metrics import pairwise_distances_argmin
# from sklearn.utils   import shuffle
# from skimage         import img_as_float #0~1

# from sklearn import cluster
# from skimage.color import rgb2gray
# from sklearn.datasets import fetch_olivetti_faces
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline

##########################################3
# Example 8. MNIST
##########################################
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784',version=1, as_frame=False, parser='auto')
data,labels = mnist["data"],mnist["target"]
train_data, test_data, train_labels, test_labels \
    = data[:60000],data[60000:],labels[:60000],labels[60000:]

def show_digit(x,label):
    plt.axis('off')
    plt.imshow(x.reshape((28,28)), cmap='gray')
    plt.title('Label '+str(label))

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    show_digit(1-test_data[i,:],test_labels[i])
plt.tight_layout() 
plt.show()

##############################################
# Example 9. kNN
##############################################
import time
from sklearn.neighbors import BallTree

# 학습 데이터에서 최근정 이웃 구조 생성
t_before = time.time()
ball_tree = BallTree(train_data)   
t_after = time.time()
t_training = t_after - t_before 
print("Time to build data structure : ", t_training, "seconds") #7.7

# 테스트 데이터에서 최근접이웃 예측
t_before = time.time()
test_neighbors = np.squeeze(ball_tree.query(test_data, k=1, return_distance=False))
test_predictions = train_labels[test_neighbors]
t_after = time.time()
t_testing = t_after - t_before
print("Time to classify test set : ", t_testing, "seconds") #614.9

# 분류기 성능 평가
t_accuracy = sum(test_predictions == test_labels) / len(test_labels)
print("The accuracy is ", t_accuracy) # 96.91%

import pandas as pd
import seaborn as sn
from sklearn import metrics

cm = metrics.confusion_matrix(test_labels, test_predictions)
df_cm = pd.DataFrame(cm, range(10), range(10))
sn.set(font_scale=1.2)
sn.heatmap(df_cm, annot=True, fmt="g") #annot_kws=("size":16))#, fmt='g')

# Errors
ind = np.squeeze(np.nonzero(np.int32(test_predictions) - np.int32(test_labels)))
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(1-test_data[ind[i],:].reshape((28,28)),cmap='gray')
    plt.title( test_predictions[ind[i]] + '/' +  test_labels[ind[i]])
    plt.axis('off')
plt.show() 

##############################################
# Example 10. Maximum Likelihood Estimates
##############################################
def fit_generative_model(x,y):
    k=10 # 레이블 갯수
    d=(x.shape)[1] # 숫자 데이터의 차원 수
    mu = np.zeros((k,d))
    sigma = np.zeros((k,d,d))
    pi = np.zeros(k)
    c = 100  # 정규화를 위한 매개변수, 교차검증을 통해 최적  C 찾기
    for label in range(k):
        indices = (np.int32(y)==label)
        pi[label] = sum(indices)/float(len(y))
        mu[label] = np.mean(x[indices,:], axis=0)
        sigma[label] = np.cov(x[indices,:], rowvar=0, bias=1) + c*np.eye(d) #정규화
    return mu,sigma,pi  # 매개변수 변환

def displaychar(image):
    plt.imshow(1-np.reshape(image,(28,28)),cmap='gray')
    plt.axis('off')

mu,sigma, pi = fit_generative_model(train_data, train_labels)

# 평균 이미지
plt.figure()
for i in range(10):
    plt.subplot(4,3,i+1) #, displaychar(mu[i]) 
    plt.imshow(1-np.reshape(mu[i],(28,28)),cmap='gray')
    plt.axis('off')
plt.show()   

# # 각 [test_image, label] 쌍에 대한 log Pr(label|image) 계산
# import scipy
# import numpy.random
# k=10
# score = np.zeros((len(test_labels),k))
# for label in range(k):
#     rv = scipy.stats.multivariate_normal(mean=mu[label], cov=sigma[label])
#     for i in range(len(test_labels)):
#         score[i,label] = np.log(pi[label]) + np.log(rv.pdf(test_data[i,:]))
# test_predictions = np.argmax(score,axis=1)

# 최종 에러 수 및 정확도 집계
# errors = np.sum(test_predictions != test_labels)
# print("The generative model makes "+str(errors)+" errors out of 10000")
# t_accuracy = sum(test_predictions == test_labels)/len(test_labels)
# print("The accuracy is ", t_accuracy)

#######################################333
# Example 11. SVM 분류기
############################################
from sklearn.svm import SVC
clf = SVC(C=1, kernel='poly', degree=2)
t_before = time.time()
clf.fit(train_data, train_labels)
print(time.time()-t_before, ' seconds') # 138 seconds
print(clf.score(test_data,test_labels))

test_predictions = clf.predict(test_data)
cm = metrics.confusion_matrix(test_labels,test_predictions)
df_cm = pd.DataFrame(cm, range(10), range(10))
sn.set(font_scale=1.2)
sn.heatmap(df_cm, annot=True, fmt="g")

wrong_indices = test_predictions != test_labels
wrong_digits  = test_data[wrong_indices]
wrong_preds   = test_predictions[wrong_indices]
correct_labs  = test_labels[wrong_indices]
print(len(wrong_preds))

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.title(str(wrong_preds[i])+ '/' +str(correct_labs[i]))
    displaychar(wrong_digits[i])
plt.show()  

#######################################333
# Example 12. 유사 하르 특징 서술자를 이용한 얼굴 분류
############################################  
from dask import delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from skimage.data import lfw_subset
from skimage.transform import integral_image
from skimage.feature import haar_like_feature
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature

@delayed
def extract_feature_image(img, feature_type, feature_coord=None):
    # 현재 영상 위해 하르 특징 추출
    ii = integral_image(img)
    return haar_like_feature(ii,0,0,ii.shape[0], ii.shape[1],\
                 feature_type=feature_type, feature_coord=feature_coord)
images = lfw_subset()
print(images.shape)   

# 얼굴영상 25개
fig = plt.figure(figsize=(5,5))
fig.subplots_adjust(left=0, right=0.9, bottom=0, top=0.9, hspace=0.05, wspace=0.05)
for i in range(25):
    plt.subplot(5,5,i+1), plt.imshow(images[i], cmap='bone')
    plt.axis('off')
plt.suptitle('Faces')
plt.show()     

# 얼굴 아닌 영상
fig = plt.figure(figsize=(5,5))
fig.subplots_adjust(left=0, right=0.9, bottom=0, top=0.9, hspace=0.05, wspace=0.05)
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(images[i+100],cmap='bone'), plt.axis('off')
plt.suptitle('Non-faces')
plt.show() 

# 유사하르특징 추출
feature_types = ['type-2-x','type-2-y'] 
# dask 모듈의 delayed() 함수를 이용하여 계산 그래프 작성
# 계산 단계를 위해 다중 CPU사용 권장
Xt = delayed(extract_feature_image(img, feature_types) for img in images[:5])
Xt.visualize()

X = delayed(extract_feature_image(img, feature_types) for img in images)

t_start = time.time()
X = np.array(X.compute(scheduler='processes'))
time_full_feature_comp = time.time() - t_start
y = np.array([1]*100 + [0]*100)
X_train, X_test, y_train, y_test = \
    train_test_split(X,y,train_size=150, random_state=0, stratify=y)
print(time_full_feature_comp) #14.4 초
print(X.shape, X_train.shape) #(200, 101400) (150, 101400)

from sklearn.metrics import roc_curve, auc, roc_auc_score
 
# 가장 중요한 특징을 선택하기 위한 가능한 모든 특징 추출
_, h, w = images.shape
feature_coord, feature_type = haar_like_feature_coord(w,h,feature_types)
# 랜덤 포레스트 분류기 학습 및 성능 확인
clf = RandomForestClassifier(n_estimators=1000, max_depth=None, \
                      max_features=100, n_jobs=-1, random_state=0)

t_start = time.time()
clf.fit(X_train, y_train)
time_full_train = time.time()-t_start  
print(time_full_train)  # 1.2
auc_full_features = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]) #(50,2)
print(auc_full_features) # 1.0

# 가장 중요한 순서대로 특징 정렬, 가장 중요한 25개 표시
idx_sorted = np.argsort(clf.feature_importances_)[::-1]
fig, axes = plt.subplots(5,5,figsize=(10,10))
for i,ax in enumerate(axes.ravel()):
    image = draw_haar_like_feature(images[1],0,0,w,h,\
                                   [feature_coord[idx_sorted[i]]])
    ax.imshow(image), ax.set_xticks([]), ax.set_yticks([])
fig.suptitle('The most important features',size=30)
plt.show()     

#######################################333
# Example 13. HOG 특징을 사용하여 SVM으로 객체 검출
############################################  
img = cv2.imread('BeachPeople.jpg')
# 기본 사람(보행자) 검출기를 사용하여 HOG 서술자 생성
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# 4화소 윈도우 보폭, 1.02의 확대, 그룹핑 없이 검출실행
# (HOG가 확대 매개 변수에서 여러 위치에서 감지함을 보이기 위해)
(foundBoundingBoxes, weights) = \
    hog.detectMultiScale(img, winStride=(4,4), padding=(8,8),\
                         scale=1.02, finalThreshold=0)
print(len(foundBoundingBoxes))  #108

# 원본 영상 복사하여 경계 박스 그림 - 원본 영상은 다시 사용 가능
imgWithRawBoxes = img.copy()
for (hx, hy,  hw, hh) in foundBoundingBoxes:
    cv2.rectangle(imgWithRawBoxes, (hx,hy), (hx+hy, hy+hh), (0,0,255), 1)

plt.figure(figsize=(20,12))
plt.subplot(121), plt.imshow(img), plt.axis('off')
imgWithRawBoxes = cv2.cvtColor(imgWithRawBoxes, cv2.COLOR_BGR2RGB)
plt.subplot(122), plt.imshow(imgWithRawBoxes, aspect='auto'), plt.axis('off')
plt.show()  

#######################################333
# Example 14. 비최대억제
############################################      
# 윈도우 10에서 imutils 라이브러리는 아나콘다에 기본 포함이 아님
# pip install imutils 를 통해 라이브러리 설치 필요
from imutils.object_detection import non_max_suppression

# 경계 박스 형식 변환 : (x,y,w,h) ==> (x1,y1,x2,y2)
rects = np.array([[x,y,x+w,y+h] for (x,y,w,h) in foundBoundingBoxes])
# 오버레이 연산 65% 기준으로 비최대 억제 수행
nmsBoundingBoxes = non_max_suppression(rects, probs=None, overlapThresh=0.65)
print(len(rects), len(nmsBoundingBoxes)) # 전체 박스 vs. 억제 박스 108:1
# 영상에 최종 경계 박스 그리기
for (x1,y1,x2,y2) in nmsBoundingBoxes:
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)

plt.figure(figsize=(20,12))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img, aspect='auto'), plt.axis('off')
plt.show()