# Ch10
# 인기있는 심층 CNN
# VGG-16
# 개와 고양이 영상을 https://www.kaggle.com/c/dogs-vs-cats 에서 다운로드 TRAIN_DIR에 저장
import cv2  # 영상 크기변경 함수 사용
import numpy as np # 베열 처리 함수 사용 - reshape(), array(), save()
import matplotlib.pyplot as plt
import os  # 디렉토리 지정

import tensorflow as tf
from tensorflow.keras.utils import to_categorical # one-hot encoding
from random import shuffle # 네트워크가 잘 학습되도록 학습 데이터 섞음
from tqdm import tqdm # 작업에 대한 깔끔한 백분율 바 표시

TRAIN_DIR = './train/train'
TEST_DIR  = './test/test1'
IMG_SIZE = 50  # 정규화 영상 크기
LR = 1e-6   # 학습률 learning rate - 경사하강 스텝
MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR,'2conv-basic') # 모델 및 학습률 지정

def label_img(img):
    word_label = img.split('.')[-3]  # 오른쪽에서 3번째 원소
    # conversion to one-hot array [cat,dog]
    if word_label == 'cat' : return 0  #[many cats, no dog]
    elif word_label == 'dog': return 1 #[no cat, many dogs]
    
def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path  = os.path.join(TRAIN_DIR, img)
        img   = cv2.imread(path)
        img   = cv2.resize(img, (IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_CUBIC)
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data) # 학습 데이터 파일로 저장 
    return training_data

train_data = create_train_data()

from tensorflow.keras.applications.vgg16 import VGG16 # VGG16 함수 임포트
from tensorflow.keras.optimizers import Adam # Adam 최적화 함수 임포트

train = train_data[:-5000]  # 뒤 5000번째까지 (0~19999)
test  = train_data[-5000:]  # 뒤 5000번부터 마지막까지 (20000~24999)
# 중간계층 시각화 문제의 메모리 해결을 위해 데이터 1/10으로 줄임
train = train_data[:2000] #%-5000]  # 뒤 5000번째까지 (0~19999)
test  = train_data[20000:20500] #-5000:]  # 뒤 5000번부터 마지막까지 (20000~24999)
X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y_train = [i[1] for i in train]
y_train = to_categorical(y_train) # 학습 데이터 레이블 one-hot encoding
print(X_train.shape, y_train.shape)

X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y_test = [i[1] for i in test]
y_test = to_categorical(y_test)

# 학습 모델 구축
model = VGG16(weights=None, input_shape=(IMG_SIZE,IMG_SIZE,3), classes=y_test.shape[1]) 
# 초기 가중치 없이 시작, 입력 데이터 형태, 구분할 그룹 개수

# Adam 최적화, 학습률 지정
from tensorflow.keras import optimizers
Adam = optimizers.Adam(learning_rate=LR) 
# beta_1=0.9, beta_2=0.999, epsilon=1e-07
model.compile(Adam, "categorical_crossentropy", metrics=["accuracy"])
model.summary() # 모델 학습 정보
model.fit(X_train, y_train, validation_data=(X_test, y_test),\
          epochs=20, batch_size=256, verbose=2) # 모델 학습 수행

# 모델 평가
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: {} \nError: {} %".format(scores[1], 100-scores[1]*100))

# 영상 시각화
from tensorflow.keras.models import Model

outputs = model.get_layer('block1_conv2').output
outputs = model.get_layer('block2_conv2').output
intermediate_layer_model = Model(inputs=model.input, outputs=outputs)
intermediate_output = intermediate_layer_model.predict(X_train)

fig = plt.figure(figsize=(10,10))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
plt.gray()
i=3
for c in range(64):
    plt.subplot(8,8,c+1), plt.axis('off')
    plt.imshow(intermediate_output[i,:,:,c])
plt.show()   

# 테스팅 단계
def process_test_data():
    testing_data = []
    for img in os.listdir(TEST_DIR):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_COLOR) #GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])
    np.save('test_data.npy', testing_data)
    return testing_data

test_data = process_test_data()
# test_data = np.load('test_data.npy') 저장된 화일에서 읽기
len(test_data)

plt.figure(figsize=(20,20))
for i in range(100):
    plt.subplot(10,10,i+1)
    plt.imshow(X_test[i])
    plt.axis('off')
    result = 'cat' if probs[i][1] < 0.5 else 'dog'
    plt.title("{}, prob={:0.2f}".format(result, max(probs[i][0],probs[i][1])))
plt.show()     