# Ch 10. Deep learning : Image classification
import numpy as np
# keras, tensorflow 라이브러리는 아나콘다에 포함되지 않음 
#pip install tensorflow
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_v2_behavior()

from tensorflow.keras.datasets import mnist 

(X_train, y_train), (X_test, y_test) = mnist.load_data()
np.random.seed(42)
train_indices = np.random.choice(60000, 50000, replace=False)
valid_indices = [i for i in range(60000) if i not in train_indices]
X_valid, y_valid = X_train[valid_indices,:,:], y_train[valid_indices]
X_train, y_train = X_train[train_indices,:,:], y_train[train_indices]
print(X_train.shape, X_valid.shape, X_test.shape)

image_size = 28
num_labels = 10
dims = image_size * image_size

def reformat(dataset, labels):
    dataset = dataset.reshape((-1,dims)).astype(np.float32)
    # one hot encoding : 1->[0,1,0 ...], 2->[0,0,1 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset,labels

X_train, y_train = reformat(X_train, y_train) 
X_valid, y_valid = reformat(X_valid, y_valid)
X_test,  y_test  = reformat(X_test,  y_test)
print('Training set', X_train.shape)
print('Validation set',X_valid.shape)
print('Test_set',X_test.shape)  

batch_size = 256
num_hidden_units = 1024
lambda1, lambda2 = 0.1, 0.1

graph = tf.Graph()
with graph.as_default():
    # 훈련을 위한 데이터 - 실행 시간 값 제공 위해 placeholder 사용, tf 2.0부터는 사용불가
    tf_train_dataset = tf.placeholder(dtype=tf.float32, shape=(batch_size,dims))
    #tf_train_dataset = tf.Variable(tf.ones(shape=(batch_size,dims)), dtype=tf.float32)
    tf_train_labels  = tf.placeholder(dtype=tf.float32, shape=(batch_size,num_labels))
    #tf_train_labels  = tf.Variable(tf.ones(shape=(batch_size,num_labels)), dtype=tf.float32)
    tf_valid_dataset = tf.constant(X_valid)
    tf_test_dataset  = tf.constant(X_test)
    
    # 변수들 지정
    weights1 = tf.Variable(tf.truncated_normal([dims, num_hidden_units]))
    biases1  = tf.Variable(tf.zeros([num_hidden_units]))
    # 행렬 내적 연산과 bias 더 해 입력을 은닉층으로 연결, relu 함수로 출력
    layer_1_outputs = tf.nn.relu(tf.matmul(tf_train_dataset, weights1)+biases1)
    weights2 = tf.Variable(tf.truncated_normal([num_hidden_units, num_labels]))
    biases2  = tf.Variable(tf.zeros([num_labels]))
    
    # 학습 계산
    logits = tf.matmul(layer_1_outputs, weights2) + biases2
    loss = tf.nn.softmax_cross_entropy_with_logits_v2( \
            labels=tf_train_labels, logits=logits) #_v2 버전으로 변경
    loss = tf.reduce_mean(loss + lambda1*tf.nn.l2_loss(weights1) \
                          + lambda2*tf.nn.l2_loss(weights2)) 
    # 확률적 경사하강(SGD)법을 이용한 최적화 수행-학습률 0.008
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.008).minimize(loss)
    train_prediction = tf.nn.softmax(logits) # 학습 데이터에 대한 예측 수행
    # 검증 데이터에 대한 출력층 계산 및 예측 수행
    layer_1_outputs = tf.nn.relu(tf.matmul(tf_valid_dataset, weights1)+biases1)
    valid_prediction = tf.nn.softmax(tf.matmul(layer_1_outputs, weights2)+biases2)
    
    layer_1_outputs = tf.nn.relu(tf.matmul(tf_test_dataset, weights1)+biases1)
    test_prediction = tf.nn.softmax(tf.matmul(layer_1_outputs, weights2)+biases2)
    
import matplotlib.pyplot as plt
def accuracy(predictions, labels):
    softmax = np.sum(np.argmax(predictions,1)==np.argmax(labels,1))
    return (100.0*softmax/predictions.shape[0])

num_steps = 6001
ll, atr, av = [],[],[]

with tf.Session(graph=graph) as session:
    #tf.global_variables_initializer().run()
    session.run(tf.initialize_all_variables())
    print("Initialized")
    for step in range(num_steps):
        # offset으로 랜덤하게 학습 데이터 선택, epochs로 더 나은 랜덤 배정 가능
        offset = (step*batch_size) % (y_train.shape[0] - batch_size)
        # 미니배치 생성
        batch_data = X_train[offset:(offset+batch_size),:]
        batch_labels = y_train[offset:(offset+batch_size),:]
        # 세션 실행에서 미니배치에 전할 딕셔너리 준비
        # 딕셔너리 키- 그래프의 플레이스 홀더 노드
        # 딕셔너리 값- 플레이스 홀더에 제공하는 numpy array
        feed_dict ={tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], \
                                        feed_dict=feed_dict)
        if (step % 500 == 0):
            ll.append(l)
            a1 = accuracy(predictions, batch_labels)
            a2 = accuracy(valid_prediction.eval(), y_valid)
            atr.append(a1)
            av.append(a2)
            # 테스트 데이터 500개마다 결과 출력
            print("Minibatch loss at step %d: %f"%(step,l))
            print("Minibatch accuracy: %.1f%%"% a1)
            print("Validation accuracy: %.1f%%"% a2)
    print("Test accuracy: %.1f%%"% accuracy(test_prediction.eval(), y_test))
  
# 계층 1의 가중치를 시각화
    images = weights1.eval()
    plt.figure(figsize=(18,18))
    indices = np.random.choice(num_hidden_units,225)
    
    for j,idx in enumerate(indices):
        plt.subplot(15,15,j+1)
        plt.imshow(np.reshape(images[:,idx], (image_size,image_size)), cmap='gray')
        plt.xticks([],[]), plt.yticks([],[])
    
    plt.suptitle('SGD after Setp '+str(step)+ ' with lambda1=lambda2='+str(lambda1),size=30)
    plt.show()  
    
plt.figure(figsize=(8,12))
plt.subplot(211)
plt.plot(range(0,num_steps,500), atr, '.-', label="training accuracy")
plt.plot(range(0,num_steps,500), av, '.-', label="validation accuracy")
plt.xlabel('GD steps'), plt.ylabel('Accuracy')
plt.legend(loc="lower right")
plt.subplot(212)
plt.plot(range(0,num_steps,500), ll,'.-')
plt.xlabel('GD steps'), plt.ylabel("Softmax loss")
plt.show() 

