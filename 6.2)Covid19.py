import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from IPython.display import Image, display
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
import sys
from imutils import paths
from os import path

args={}
args["dataset"]='covid/dataset'
args["plot"]='plot.png'
args["model"]='covid19.model'
args["testset"]='covid/testset'


directory = input("Select the path to save the model to ")
try:
    os.path.isfile('model.h5')
    print("Path is valid")
except IOError:
    print("Error: Path is invalid")

#learning rate
LR = 1e-3
#epochs for training
EPOCHS = 25
#batch size
BS = 8


def Imagepreprocess(image):
 image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 # negative transformation to make everything looks clearer
 invert_neg = (255 - image)
 # Power Law transformation 
 powerlaw = np.array(255*(invert_neg/255)**0.5, dtype='uint8')
 resultimage = cv2.cvtColor(powerlaw, cv2.COLOR_GRAY2RGB)
 return resultimage


print("Images loading")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []
 
    
for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (224, 224))
    image = Imagepreprocess(image)
    data.append(image)
    labels.append(label)
data = np.array(data) / 255.0
labels = np.array(labels)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=0.20, stratify=labels, random_state=7)
trainAug = ImageDataGenerator(rotation_range=15,fill_mode="nearest")
    
print("Images are loaded\n")

baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

print("Models are being compiled\n")
opt = Adam(lr=LR, decay=LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])


print("Model head Training\n")
H = model.fit_generator(
trainAug.flow(trainX, trainY, batch_size=BS),
steps_per_epoch=len(trainX) // BS,
validation_data=(testX, testY),
validation_steps=len(testX) // BS,
epochs=EPOCHS)

print("Model evalution\n")
predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testY.argmax(axis=1), predIdxs,
target_names=lb.classes_))

confusionmatrix = confusion_matrix(testY.argmax(axis=1), predIdxs)
total = sum(sum(confusionmatrix))
acc = (confusionmatrix[0, 0] + confusionmatrix[1, 1]) / total
sensitivity = confusionmatrix[0, 0] / (confusionmatrix[0, 0] + confusionmatrix[0, 1])
spec = confusionmatrix[1, 1] / (confusionmatrix[1, 0] + confusionmatrix[1, 1])

print(confusionmatrix)
print("accuracy: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(spec))

N = EPOCHS
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on COVID-19 Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

model.save(directory) 

print("Loading images")
imagePaths = list(paths.list_images(args["testset"]))
testX = []
testY = []
for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (224, 224))
    image = transform(image)
    testX.append(image)
    testY.append(label)
testX = np.array(testX) / 255.0
testY = np.array(testY)

lb = LabelBinarizer()
testY = lb.fit_transform(testY)
testY = to_categorical(testY)
print("Done")

print("Evaluating model")
predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testY.argmax(axis=1), predIdxs,
target_names=lb.classes_))

confusionmatrix = confusion_matrix(testY.argmax(axis=1), predIdxs)
total = sum(sum(confusionmatrix))
acc = (confusionmatrix[0, 0] + confusionmatrix[1, 1]) / total
sensitivity = confusionmatrix[0, 0] / (confusionmatrix[0, 0] + confusionmatrix[0, 1])
spec = confusionmatrix[1, 1] / (confusionmatrix[1, 0] + confusionmatrix[1, 1])

print(confusionmatrix)
print("accuracy: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(spec))