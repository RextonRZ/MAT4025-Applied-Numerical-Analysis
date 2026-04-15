# Ch 10. Deep learning : Image classification
################################################
# Example 2. Kerase 를 이용한 영상분류
##################################################
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input

(X_train,y_train),(X_test,y_test) = mnist.load_data()
print(X_train.shape, X_test.shape)
# 4차원 형태로 변환 [samples][width][height][1]
X_train = X_train.reshape(X_train.shape[0],28,28,1).astype('float32')
X_test  = X_test.reshape(X_test.shape[0],28,28,1).astype('float32')
X_train = X_train/255  # 정규화
X_test  = X_test/255
y_train = to_categorical(y_train)  # one hot encoding
y_test  = to_categorical(y_test)
num_classes = y_test.shape[1] # 분류 클래스 개수

model = Sequential([
    Input(shape=(28, 28, 1)),
    Flatten(),  #2차원 벡터화 후 추가
    Dense(200, activation='relu'), #relu 활성화 추가
    Dropout(0.15), #드롭아웃 레이트 : 0.15
    Dense(200, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# model = Sequential()
# model.add(Flatten(input_shape=(28,28,1))) 
# model.add(Dense(200,activation='relu'))   
# model.add(Dropout(0.15))                  
# model.add(Dense(200,activation='relu'))
# model.add(Dense(num_classes,activation='softmax'))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
   
model.summary()    # 모델 정보 요약 출력 
    
model.fit(X_train, y_train, validation_data=(X_test,y_test), \
          epochs=10, batch_size=200, verbose=2)
    
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: {}\nError: {} %".format(scores[1], 100-scores[1]*100))    

# Keras 시각화
# pydot_ng 설치 필요: pip install pydot_ng
# GraphVviz 라이브러리 설치 : conda install graphviz
import pydot_ng as pydot
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model.png')

# 중간 계층에서 가중치 시각화
# Existing layers are: ['flatten', 'dense', 'dropout', 'dense_1', 'dense_2'].
W = model.get_layer('dense').get_weights()
print(W[0].shape) # 첫번째 은닉층의 가중치
print(W[1].shape) # 첫번째 은닉층의 바이아스

fig = plt.figure(figsize=(20,20))
fig.subplots_adjust(left=0, right=1, bottom=0, hspace=0.05, wspace=0.05)
plt.gray()
for i in range(200):
    plt.subplot(15,14,i+1), plt.axis('off')
    plt.imshow(np.reshape(W[0][:,i],(28,28))) #786개 1차원 벡터 -> 28*28 영상
plt.suptitle('First Dense layer Weights (200 hidden units)',size=30)
plt.show()  

W = model.get_layer('dense_2').get_weights()
print(W[0].shape) # 마지막 은닉층의 가중치
print(W[1].shape) # 마지막 은닉층의 바이아스

fig = plt.figure(figsize=(20,20))
fig.subplots_adjust(left=0, right=1, bottom=0, hspace=0.05, wspace=0.05)
plt.gray()
for i in range(10):
    plt.subplot(4,3,i+1), plt.axis('off')
    plt.imshow(np.reshape(W[0][:,i],(10,20))) # 20개 1차원 벡터 영상
plt.suptitle('Last Dense layer Weights (10 hidden units)',size=30)
plt.show()  

###########################################################3
# Example 3. Keras 분류를 위한 CNN
############################################################    
from tensorflow.keras.layers import Conv2D, MaxPooling2D

inputs = Input(shape=(28, 28, 1), name='input_layer')
x = Conv2D(64, (5, 5), activation='relu', name='conv_layer')(inputs)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max_pool')(x)
x = Flatten(name='flatten')(x)
x = Dense(100, activation='relu', name='dense_1')(x)
outputs = Dense(num_classes, activation='softmax', name='output_layer')(x)

# Functional 모델 정의
model = Model(inputs=inputs, outputs=outputs, name='mnist_model')

# modelC = Sequential()
# #keras.Input( shape = (28,28,1))
# modelC.add(Conv2D(64,(5,5), activation='relu', input_shape=(28,28,1))) 
# modelC.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))   
# modelC.add(Flatten())
# modelC.add(Dense(100, activation='relu'))
# modelC.add(Dense(num_classes,activation='softmax'))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
   
model.summary()

model.fit(X_train, y_train, validation_data=(X_test,y_test),\
           epochs=10, batch_size=200, verbose=2)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: {}\nError: {} %".format(scores[1], 100-scores[1]*100))      

# 중간계층 시각화
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('conv_layer').output)

# 중간 계층 출력 예측
intermediate_output = intermediate_layer_model.predict(X_train[:32], verbose=0)

# intermediate_layer_model = Model(inputs=Input(shape=(28,28,1)), outputs=model.get_layer('conv_layer').output)
# intermediate_output = intermediate_layer_model.predict(X_train[:32])

print("\nInput shape:", intermediate_layer_model.input_shape)
print("Intermediate output shape:", intermediate_output.shape)
#print(intermediate_layer_model.inputs.shape, intermediate_output.shape)  

# 시각화를 위한 첫 번째 이미지의 모든 필터 출력
IND = 0
plt.figure(figsize=(16, 16))
num_filters = intermediate_output.shape[-1]
size = intermediate_output.shape[1]
cols = 8
rows = num_filters // cols + (num_filters % cols > 0)
for i in range(num_filters):
    ax = plt.subplot(rows, cols, i + 1)
    plt.imshow(intermediate_output[IND, :, :, i], cmap='gray')
    plt.axis('off')
plt.suptitle('Intermediate Conv2D Layer Outputs', fontsize=16)
plt.tight_layout()
plt.show()  

IND = 1
plt.figure(figsize=(16, 16)) 
num_filters = intermediate_output.shape[-1]
size = intermediate_output.shape[1]
cols = 8
rows = num_filters // cols + (num_filters % cols > 0)
for i in range(num_filters):
    ax = plt.subplot(rows, cols, i + 1)
    plt.imshow(intermediate_output[IND, :, :, i], cmap='gray')
    plt.axis('off')
plt.suptitle('Intermediate Conv2D Layer Outputs', fontsize=16)
plt.tight_layout()
plt.show()  

