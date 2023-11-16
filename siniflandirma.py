import numpy as np
import pandas as pd
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import pickle
import tensorflow as tf

path = "myData"
myList = os.listdir(path)
noOfClasses = len(myList)


images = []
classNo = []


for i in range(noOfClasses):
    myimageList = os.listdir(path + "//" + str(i))
    for j in myimageList:
        img = cv2.imread(path + "//" + str(i) + "//"+ j)
        img = cv2.resize(img, (32,32))
        images.append(img)
        classNo.append(i)

# print(len(images))
# print(len(classNo))

images = np.array(images)
classNo = np.array(classNo)

#print(images.shape)
#print(classNo.shape)

x_train, x_test, y_train, y_test = train_test_split(images, classNo, test_size=0.5, random_state=42)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# print(images.shape)
# print(x_train.shape)
# print(x_test.shape)
# print(x_validation.shape)

y_train_series = pd.Series(y_train)
y_test_series = pd.Series(y_test)
y_validation_series = pd.Series(y_validation)

fig, axes = plt.subplots(3, 1, figsize=(7, 7))
fig.subplots_adjust(hspace=0.5)

# Her bir sınıfın sayısını çiz
# sns.countplot(x=y_train_series, ax=axes[0] )
# axes[0].set_title("y_train")

# sns.countplot(x=y_test_series, ax=axes[1])
# axes[1].set_title("y_test")

# sns.countplot(x=y_validation_series, ax=axes[2])
# axes[2].set_title("y_validation")

# plt.show()

def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255

    return img

# idx = 4063
# img = preProcess(x_train[idx])
# img = cv2.resize(img,(300,300))
# cv2.imshow("Preprocess", img)
# cv2.waitKey(0)

x_train = np.array(list(map(preProcess, x_train)))
x_test = np.array(list(map(preProcess, x_test)))
x_validation = np.array(list(map(preProcess, x_validation)))

x_train = x_train.reshape(-1,32,32,1)
x_test = x_test.reshape(-1,32,32,1)
x_validation = x_validation.reshape(-1,32,32,1)
#print(x_validation.shape)

dateGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             rotation_range=10)
dateGen.fit(x_train)
y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)
#print(y_validation.shape)

model = Sequential()
model.add(Conv2D(input_shape = (32,32,1), filters= 8 , kernel_size=(5,5), activation="relu", padding = "same"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=16, kernel_size=(5, 3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=256, activation= "relu"))
model.add(Dropout(0.2))
model.add(Dense(units=noOfClasses, activation="softmax"))

model.compile( loss = "categorical_crossentropy", optimizer=("Adam"), metrics=["accuracy"])
batch_size = 16

hist = model.fit_generator(dateGen.flow(x_train, y_train, batch_size= batch_size),
                                        validation_data = (x_validation, y_validation),
                                        epochs = 50, steps_per_epoch= x_train.shape[0]//batch_size, shuffle = 1)

model.save('model.h5')



# plt.figure()
# plt.plot(hist.history["loss"], label = "Egitim Loss")
# plt.plot(hist.history["val_loss"], label = "Val Loss")
# plt.show()

# plt.figure()
# plt.plot(hist.history["accuracy"], label = "Egitim Accuracy")
# plt.plot(hist.history["val_accuracy"], label = "Val Accuracy")
# plt.show()

score = model.evaluate(x_test, y_test, verbose= 1)
# print("Test Lost: ", score[0])
# print("Test Accuracy: ", score[1])

y_pred = model.predict(x_validation)
y_pred_class = np.argmax(y_pred, axis = 1)
y_true = np.argmax(y_validation, axis = 1)

cm = confusion_matrix(y_true, y_pred_class)

# f, ax = plt.subplots(figsize=(8,8))
# sns.heatmap(cm, annot = True, linewidths=0.01, cmap = "magma", linecolor="gray", fmt=".1f", ax=ax)
# plt.xlabel("prediction")
# plt.ylabel("true")
# plt.title("cm")
# plt.show()



