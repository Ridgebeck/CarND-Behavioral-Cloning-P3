import csv
import cv2
import numpy as np

# Open csv file in data folder
lines=[]
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Read all images and steering angles and save them in numpy arrays
images = []
measurements = []
augment = True
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './data/IMG/' + filename
    image = cv2.imread(current_path)
    measurement = float(line[3])
    
    # Augment image data
    # Flip every second image and steering angle
    if augment == True:
        image = np.fliplr(image)
        measurement = -measurement

    # RGB to YUV
        
    augment = not augment    
    images.append(image)
    measurements.append(measurement)
    
X_train = np.array(images)
y_train = np.array(measurements)

# Visualize data
import matplotlib.pyplot as plt

# Histogram of the steering data
plt.hist(y_train, bins=20, facecolor='green')

plt.xlabel('Steering Angle')
plt.ylabel('Frequency')
plt.title(r'Frequency of steering angles')
plt.grid(True)
plt.show()




from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

# Model based on NVIDIA's DAVE-2
model = Sequential()
Lambda(lambda x: (x / 255.0) - 0.5)

# Convolution 1 - 3
model.add(Conv2D(6,5,2, activation='relu',input_shape=(160,320,3)))
model.add(Conv2D(6,5,2, activation='relu'))
model.add(Conv2D(6,5,2, activation='relu'))
# Convolution 4-5
model.add(Conv2D(6,3,0, activation='relu'))
model.add(Conv2D(6,3,0, activation='relu'))

# Flatten Layer
model.add(Flatten())

# Fully connected layer        
model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))

# Output Layer
model.add(Dense(1, activation='relu'))

# Compile model with squared error function          
model.compile(loss='mse', optimizer='adam')
          
# Shuffle data, take 20% as validation data, fir the model
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

# Save model
model.save('model.h5')

