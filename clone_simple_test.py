import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Open csv file in data folder
lines=[]
with open('./simple_test.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Read all images and steering angles and save them in lists
images = []
measurements = []
for line in lines:
    data = []
    data.append(int(line[0]))
    data.append(int(line[1]))
    measurement = int(line[2])
    
    images.append(data)
    measurements.append(measurement)
    
# Convert lists to numpy arrays
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential, Model
from keras.layers import Lambda, Flatten, Dense
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout

print (X_train.shape)

# Model based on NVIDIA's DAVE-2
model = Sequential()
#model.add(Flatten()))
model.add(Dense(500, input_shape=(2,)))
model.add(Dense(1))

# Compile model with squared error function          
model.compile(loss='mse', optimizer='adam')
          
# Shuffle data, take 20% as validation data, fit the model
model.fit(X_train, y_train, verbose=1, validation_split=0.2, shuffle=True, nb_epoch=5)

# Save model
model.save('simple_model.h5')

