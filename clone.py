import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import image_preprocessing as ip

print ("Loading data...")
# Open csv file in data folder
lines=[]
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for i, line in enumerate(reader):
        lines.append(line)

images = []
measurements = []

print ("Preprocessing data...")
for line in lines:
    for column in range(0,3):
        # Read image data
        source_path = line[column]
        filename = source_path.split('/')[-1]
        current_path = './data/IMG/' + filename
        image = cv2.imread(current_path)
        
        ### Preprocess steering angle data ###
        measurement = float(line[3])

        # Round steering angle data
        measurement = round(measurement, 4)
            
        # Change steering angle data depending on the camera the image is from  
        if column == 0:
            offset = 0
        elif column == 1:
            offset = 0.1
        elif column == 2:
            offset = -0.1

        measurement = measurement + offset  

        ### Preprocess Image Data FOr Different Cameras ###
        image, measurement = ip.pre_process(image, measurement, True)
        
        # Only allow a maximum amount of the same steering angle values to avoid bias (going straight)
        maximum_angles = 60
        if measurements.count(measurement) < maximum_angles:  
            images.append(image)
            measurements.append(measurement)

#cv2.imshow("Test Image", images[500])
#cv2.imwrite("Writeup/Image Processing/new_pic.png", images[500])

X_train = np.array(images)
y_train = np.array(measurements)

print ("Image data processed.")

"""
# Normalize Steering angles
max_angle = y_train.max()
min_angle = y_train.min()
new_max = 1
new_min = -1
y_train = (new_max - new_min) / (max_angle - new_min) * (y_train - max_angle) + new_max
"""
"""
# Visualize data - Histogram of the steering data
print ("Mean: {}".format(y_train.mean()))
print ("Max: {}".format(y_train.max()))
print ("Min: {}".format(y_train.min()))
plt.hist(y_train, bins=500, facecolor='green')
plt.xlabel('Steering Angle')
plt.ylabel('Frequency')
plt.title(r'Frequency of steering angles')
plt.grid(True)
plt.show()
"""

from keras.models import Sequential, Model
from keras.layers import Activation, Lambda, Flatten, Dense, Reshape, Dropout
from keras.layers.convolutional import Conv2D, Cropping2D
from pathlib import Path

# Look for file of already trained model
if Path("model.h5").is_file():
    print ("Existing model found...Loading Model.")
    model = load_model("model.h5")
else: 
    # New Model - model based on NVIDIA's DAVE-2
    model = Sequential()
    
    model.add(Conv2D(24,5,2, input_shape=(80,80,3), activation ='elu'))
    model.add(Conv2D(36,5,2, activation ='elu'))
    """
    model.add(Conv2D(48,5,2, activation ='elu'))
    model.add(Conv2D(64,3,1, activation ='elu'))
    model.add(Conv2D(64,3,1, activation ='elu'))
    """
    model.add(Flatten())
    """
    model.add(Dense(1000, activation ='elu')) #1164
    model.add(Dense(100, activation ='elu'))
    model.add(Dense(50, activation ='elu'))
    """
    model.add(Dense(10, activation ='elu'))
    model.add(Dense(1))

# Compile model with squared error function          
model.compile(loss='mse', optimizer='adam')
          
# Shuffle data, take 20% as validation data, fit the model
model.fit(X_train, y_train, verbose=1, validation_split=0.2, shuffle=True, nb_epoch=1)

# Save model
model.save('model.h5')
print ("Model saved")

