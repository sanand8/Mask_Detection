import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.preprocessing import LabelBinarizer
from keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input,AveragePooling2D,Flatten,Dense,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt



#initial learning rate, number of epochs and batch size
#lower learning rate will let you get the better accuracy soon
init_lr = 1e-4
epochs = 20
batch_size = 32

DIRECTORY = r"C:\Users\devil\Desktop\Mask Detection\dataset"
Categories = ["with_mask","without_mask"]

print("Loading the images...")
#data append all images as array 
data = []
#labels will contain the image labels
labels = []

for category in Categories:
    #it will join the directory
    path = os.path.join(DIRECTORY, category)
    #os.listdir will list the images in the directory
    for img in os.listdir(path):
        img_path = os.path.join(path,img)
        image = load_img(img_path, target_size=(244, 244))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(category)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

#Img data generator will generate many image from a single image
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

baseModel = MobileNetV2(weights="imagenet",include_top=False,
input_tensor=Input(shape=(224, 244, 3)))

#construct the head model that has to placed at the top
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7,7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

#now place this on top of model
model = Model(inputs=baseModel.input, outputs=headModel)

#freeze all the layers in the base model 
for layer in baseModel.layers:
    layer.trainable = False

#compile the model
print("Now compiling the model")
opt = Adam(lr=init_lr,decay=init_lr / epochs)
model.compile(Loss="binary_crossentropy", optimizer=opt,
metrics=["accuracy"])

#train the head of the network
print("training the head of the network")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=batch_size),
    steps_per_epoch=len(trainX) // batch_size,
    validation_data=(testX, testY),
    validation_steps=len(testX) // batch_size,
    epochs=epochs)

#make prediction on the testing set
print("evaluating the network")
pred = model.predict(testX, batch_size=batch_size)

pred = np.argmax(pred, axis=1)

#show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), pred, target_names=lb.classes_))


#serialize the model to disk
print("saving the mask detector in the model")
model.save("mask_detector.model", save_format="h5")

#plot the training loss and accuracy
n = epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arrange(0,n), H.history["loss"], Label="train_loss")
plt.plot(np.arrange(0,n), H.history["val_loss"], Label="val_loss")
plt.plot(np.arrange(0,n), H.history["accuracy"], Label="train_acc")
plt.plot(np.arrange(0,n), H.history["val_accuracy"], Label="val_acc")
plt.title("training the loss and accuracy")
plt.xlabel("epoch #")
plt.ylabel("loss/accuracy")
plt.legend(Loc="lower left")
plt.savefig("plot.jpg")
