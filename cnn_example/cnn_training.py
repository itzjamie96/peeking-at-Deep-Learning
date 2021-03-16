import os
import sys
import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Dropout, Dense, BatchNormalization
from keras.layers import Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
import datetime
import matplotlib.pyplot as plt
import tensorflow as tf


# 경고 메세지 무시하게 해줌 (tf 경고 코드 오지게 많으니까)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# GPU 메모리 할당할 때 오류생길 수 있어서 해결하는 코드
# Code for Correcting Error
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# progress bar
# Define Print Progress Bar Function
def print_progress(iteration, total, prefix='>> Progress:', suffix='Complete', decimals=1, bar_length=55):
    str_format = "{0:." + str(decimals) + "f}"
    current_progress = iteration / float(total)
    percents = str_format.format(100 * current_progress)
    filled_length = int(round(bar_length * current_progress))
    bar = "■" * filled_length + '□' * (bar_length - filled_length)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


# Set Basic Parameters
# # Training Data Path (Directory)
train_dir = "../Dataset/train_data/"
validation_dir = "../Dataset/validation_data/"
# # Classes => Types of Fish
# class = category
cls = os.listdir(train_dir)
# # Number of Classes
num_cls = len(cls)
# # Size of Images
img_height = 128
img_width = 128


# Parsing Images
# # Parsing Training Image
# x = 이미지 사이즈/ RGB값
# y = labeling array
x_train = []
y_train = []
# ## Image Parsing Process
print("\nTraining Image Parsing Started")
for idx, cl in enumerate(cls):
    print("-" + cl + "- Image Parsing Progressing")
    label = [0 for i in range(num_cls)]
    label[idx] = 1
    img_dir = train_dir + cl + '/'
    for top, dir, f in os.walk(img_dir):
        for filename in f:
            print_progress(f.index(filename) + 1, len(f))
            img = Image.open(img_dir + filename)
            img = img.convert("RGB")
            img = img.resize((img_width, img_height))
            img_data = np.asarray(img)
            x_train.append(img_data/255)
            y_train.append(label)
x_train = np.array(x_train)
y_train = np.array(y_train)
print("Training Image Parsing Finished")
# # Parsing Validation Image
x_val = []
y_val = []
# ## Image Parsing Process
print("\nValidation Image Parsing Started")
for idx, cl in enumerate(cls):
    print("-" + cl + "- Image Parsing Progressing")
    label = [0 for i in range(num_cls)]
    label[idx] = 1
    img_dir = validation_dir + cl + '/'
    for top, dir, f in os.walk(img_dir):
        for filename in f:
            print_progress(f.index(filename) + 1, len(f))
            img = Image.open(img_dir + filename)
            img = img.convert("RGB")
            img = img.resize((img_width, img_height))
            img_data = np.asarray(img)
            x_val.append(img_data/255)
            y_val.append(label)
x_val = np.array(x_val)
y_val = np.array(y_val)
print("Validation Image Parsing Finished\n")


# Build CNN Model
Fish_Classifier = Sequential()
# # Feature Extraction Layer
# ## Convolution Layer
# input_shape(128,128,3(RGB))
# filters=32 (2의 제곱으로 설정하는게 일반적, 점차 커지게)
# strides = 열 어떻게 움직일 것인가
# activation=relu << 걍 function이름
Fish_Classifier.add(Conv2D(input_shape=(img_width, img_height, 3), filters=32, kernel_size=(3, 3), strides=(1, 1),
                           padding='same', activation='relu'))
Fish_Classifier.add(BatchNormalization())
# ## Pooling Layer (Max Pooling)
Fish_Classifier.add(MaxPool2D(pool_size=(2, 2)))
# ## Dropout Layer for Preventing Overfitting
# batchNormalization, Dropout = overfitting 방지
# pooling = max pooling 형태 제일 큰 것만 가져오기
# pooling 많이 할 수록 이미지 사이즈가 작아질 수 있으니 이런걸 고려해서 pooling size 설정해줘야한다
# layer가 많아질 수록 복잡해지니까 아웃오브메모리가 뜰 수도 있다
# 모델 건들기 = batch를 건들어보자
Fish_Classifier.add(Dropout(0.25))
Fish_Classifier.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
Fish_Classifier.add(BatchNormalization())
Fish_Classifier.add(MaxPool2D(pool_size=(2, 2)))
Fish_Classifier.add(Dropout(0.25))
Fish_Classifier.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
Fish_Classifier.add(BatchNormalization())
Fish_Classifier.add(MaxPool2D(pool_size=(2, 2)))
Fish_Classifier.add(Dropout(0.25))
Fish_Classifier.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
Fish_Classifier.add(BatchNormalization())
Fish_Classifier.add(MaxPool2D(pool_size=(2, 2)))
Fish_Classifier.add(Dropout(0.25))
Fish_Classifier.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
Fish_Classifier.add(BatchNormalization())
Fish_Classifier.add(MaxPool2D(pool_size=(2, 2)))
Fish_Classifier.add(Dropout(0.25))
Fish_Classifier.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
Fish_Classifier.add(BatchNormalization())
Fish_Classifier.add(MaxPool2D(pool_size=(2, 2)))
Fish_Classifier.add(Dropout(0.25))


# # Fully-Connected Layer

# 2차원의 그림을 1차원으로 바꿔줌
Fish_Classifier.add(Flatten())
# 입력계층 몇개로 할지 정하기
Fish_Classifier.add(Dense(512, activation='relu'))
Fish_Classifier.add(BatchNormalization())
Fish_Classifier.add(Dropout(0.5))


# ## Output Layer
# 우리가 가진 클래스 수를 정해줌 (우리가 가진 클래스의 갯수만큼 출력층이 나와야함)
Fish_Classifier.add(Dense(num_cls, activation="softmax"))


# # Set Optimizer (RMSprop) and Learning Rate
# #### The smaller the learning rate, the slower the training speed,
# #### but the higher the possibility that the performance of the model is good,
# #### and the larger the learning rate is, the faster the training speed,
# #### but the higher the possibility that the performance of the model is less good.
opt = RMSprop(lr=0.0001)

# 클래스가 2개인 경우 = binary
# 3개 이상부터 categorical
Fish_Classifier.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])


# Set Training Condition
Datetime = datetime.datetime.now().strftime('%m%d_%H%M')


# # Set Saving Path of Trained Model
# 가중치 계속 업데이트하는데, 1~100 다 돌기 전에 좋은 모델이 나와서 저장하려면?
# =체크포인트를 만들어준다
# = validation loss값을 기준으로 좋은지 안좋은지 판단하겠다 = 이게 제일 작을 때마다 저장하게 됨
Check_Pointer = ModelCheckpoint(filepath="fish_classification.h5", monitor='val_loss', verbose=1, save_best_only=True)

# 일정 수준이 지나도 모델의 성능이 더 바뀌지 않는 것 같을 때 학습 일찍 종료
# 50번동안 모델이 개선되지 않으면 종료시키겠다
Early_Stopping_Callback = EarlyStopping(monitor='val_loss', patience=50)

# Batch Learning
# batch_size = mini batch에 들어갈 데이터의 갯수 (요즘은 30언저리에서 설정)
history = Fish_Classifier.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1, validation_data=(x_val, y_val),
                              callbacks=[Early_Stopping_Callback, Check_Pointer])


# Visualize Training Process and Result
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()
loss_ax.plot(history.history['loss'], 'y', label='train loss')
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
acc_ax.plot(history.history['acc'], 'b', label='train acc')
acc_ax.plot(history.history['val_acc'], 'g', label='val acc')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')
loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
plt.show()