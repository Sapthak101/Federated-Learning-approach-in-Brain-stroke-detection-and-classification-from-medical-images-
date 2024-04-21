import flwr as fl
import numpy as np
import sys
import seaborn as sns
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D,Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, TimeDistributed, Bidirectional, LSTM, GRU, Dense, Dropout, Input, concatenate
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import layers
from tensorflow.keras import Model
from keras.applications.vgg16 import VGG16
from keras.layers import Reshape
import os
import cv2
from keras.regularizers import l2
import argparse
import os
from catboost import CatBoostClassifier, Pool
import warnings 
warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import log_loss
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.metrics import cohen_kappa_score
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import os
import cv2
import time
import shutil
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
sns.set_style('darkgrid')
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, Adamax
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.models import Model, load_model, Sequential
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization

def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (224, 224))  # Resize the image to 224x224
            images.append(img)
            labels.append(folder.split('/')[-1])  # Extracting the label from folder path
    return images, labels

# Define the paths to the folders containing the images
folder_paths = ['Dataset/Normal', 'Dataset/Stroke_Any']

all_images = []
all_labels = []

# Iterate through each folder
for folder_path in folder_paths:
    images, labels = load_images_from_folder(folder_path)
    all_images.extend(images)
    all_labels.extend(labels)

# Convert lists to numpy arrays
all_images = np.array(all_images)
all_labels = np.array(all_labels)

# Preprocess images
#normalized_images = all_images / 255.0  # Normalize pixel values

# Preprocess labels
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(all_labels)
integer_encoded = integer_encoded.reshape(-1, 1)  # Reshape to a 2D array

onehot_encoder = OneHotEncoder(categories='auto', sparse_output=False)  # Specify sparse=False
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

X_train, X_test, y_train, y_test=train_test_split(all_images, onehot_encoded, test_size=0.1, random_state=101)
#X_train, X_val, y_train, y_val=train_test_split(X_train, y_train, test_size=0.1, random_state=101)

IMG_WIDTH = 224
IMG_HEIGHT = 224
from tensorflow.keras.applications import EfficientNetB0
pre_trained_model = EfficientNetB0(input_shape = (IMG_WIDTH, IMG_HEIGHT, 3), 
                                include_top = False, 
                                weights = "imagenet", pooling= 'max')

for layer in pre_trained_model.layers:
    layer.trainable = True
from tensorflow.keras import layers, Model, callbacks
#model = Sequential([
#    pre_trained_model,
#    BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001), Flatten(),
#    Dense(256, kernel_regularizer= regularizers.l2(l= 0.016), activity_regularizer= regularizers.l1(0.006),
#                bias_regularizer= regularizers.l1(0.006), activation= 'relu'),
#    Dropout(rate= 0.45, seed= 123),
#    Dense(2, activation= 'softmax')
#])
model=Sequential()
model.add(Input(shape = (IMG_WIDTH, IMG_HEIGHT, 3)))
#x = layers.GlobalAveragePooling2D()(pre_trained_model.output)
model.add(pre_trained_model)
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(1024, activation = "relu", kernel_regularizer=l2(0.01)))
model.add(Dense(512, activation = "relu", kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3))
model.add(Dense(2, activation = "softmax"))

print(model.summary())
from tensorflow.keras.optimizers import Adam, Adamax
model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])
class CifarClient(fl.client.NumPyClient):
    def __init__(self):
        self.bst = None
        self.config = None
        self.num=20
        self.index=0
        self.index1=0
    def get_parameters(self, config):
        return model.get_weights()
    
    def fit(self, parameters, config):
        global hist
        hist=[]

        epochs = 30
        history = model.fit(X_train,y_train,
                   batch_size = 32,
                   validation_split = 0.2,
                   epochs = epochs)
        
        model_loss=pd.DataFrame(model.history.history)
        model_loss.plot()

        plt.xlabel('Number of Epochs')
        str1="Loss_accu_image_fed_round_client1"+str(self.index1)+".png"
        plt.savefig(str1)

        self.index1=self.index1+1
        #print("Fit history : " ,hist)
        return model.get_weights(), len(X_train), {}
    
    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy=model.evaluate(X_test)
        print("Eval accuracy : ", accuracy)
        return loss, len(X_test), {"accuracy": accuracy}
    
fl.client.start_client(
    server_address="127.0.0.1:16080", 
    client=CifarClient().to_client())

#Perform the required analysis
y_pred=model.predict(X_test)
probabilities = y_pred

indices = np.argmax(probabilities, axis=1)

one_hot_probabilities = np.zeros((probabilities.shape[0], probabilities.shape[1]))
one_hot_probabilities[np.arange(probabilities.shape[0]), indices] = 1

labels=['Normal', 'Stroke_Any']
print(labels)
y_pred=one_hot_probabilities

kappa = cohen_kappa_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print("Cohen's kappa score:", kappa)

cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
plt.figure(figsize=(20,10))
ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax); 

labels=['0: Normal', '1: Stroke_Any']
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels)

plt.savefig("confusion_matrix_client1.png")

print(classification_report(y_test, y_pred, labels=[0,1]))