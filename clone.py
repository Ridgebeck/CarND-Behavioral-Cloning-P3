import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import image_preprocessing as ip
from pathlib import Path
import os

# Specify data and image path
data_path = 'data/'
data_image_path = 'data/IMG/'

# Create empty lists for images, angles, and lines
images = []
measurements = []
lines=[]

while True:
    image_choice = input("What do you want to read?\n 1 -center images only\n 2 -left and right images only\n 3 -all images\n")
    if image_choice == '1':
        lower_range = 0
        upper_range = 1
        break
    if image_choice == '2':
        lower_range = 1
        upper_range = 3
        break
    if image_choice == '3'  :
        lower_range = 0
        upper_range = 3
        break
"""
def generate_arrays_from_file(path):
    # Preprocess data
    print ("Preprocessing data...")
    percentage = 0.1
    while True:
        with open(path) as csvfile:
            reader = csv.reader(csvfile)
            for count, line in enumerate(reader):
                for column in range(lower_range,upper_range):
                    # Show progress
                    if count == round(len(lines) * percentage):
                        print ("{}% of data processed".format(int(100 * percentage)))
                        percentage += 0.1
            
                    # Read image data
                    source_path = line[column]
                    filename = source_path.split('/')[-1]
                    image = cv2.imread(data_image_path + filename)
                    
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

                    ### Preprocess Image Data For Different Cameras ###
                    image, measurement = ip.pre_process(image, measurement, True)

                    images.append(image)
                    measurements.append(measurement)
                    
                    yield (np.array(images), np.array(measurements))
    
"""

# Open csv file in data folder
print ("Loading new driving data...")
with open(data_path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
# Preprocess data
print ("Preprocessing data...")
percentage = 0.1
for count, line in enumerate(lines):
    for column in range(lower_range,upper_range):
        # Show progress
        if count == round(len(lines) * percentage):
            print ("{}% of data processed".format(int(100 * percentage)))
            percentage += 0.1
        
        # Read image data
        source_path = line[column]
        filename = source_path.split('/')[-1]
        image = cv2.imread(data_image_path + filename)
        
        ### Preprocess steering angle data ###
        measurement = float(line[3])

        # Round steering angle data
        measurement = round(measurement, 4)
            
        # Change steering angle data depending on the camera the image is from  
        if column == 0:
            offset = 0
        elif column == 1:
            offset = 0.2
        elif column == 2:
            offset = -0.2

        measurement = measurement + offset  

        ### Preprocess Image Data For Different Cameras ###
        image, measurement = ip.pre_process(image, measurement, True)
        
        # Only allow a maximum amount of the same steering angle values to avoid bias (going straight)
        maximum_angles = 200
        if measurements.count(measurement) < maximum_angles:  
            images.append(image)
            measurements.append(measurement)

            """
            # Save processed images to new folder
            cv2.imwrite(processed_image_path + filename, image)
            # Save processed angles to new csv file
            with open(processed_path + 'processed_driving_log.csv', 'a', newline='') as new_csvfile:
                writer = csv.writer(new_csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([processed_image_path + filename, measurement])
            """

print ("Image data has been loaded. Modifying data set.")
"""
# Balance data set by copying the underepresented values
data_processed = False
while data_processed == False:
    data_processed = True
    print ("Array Length: {}".format(len(measurements)))
    
    for x_image, y_angle in zip(images, measurements):
        if measurements.count(y_angle) < maximum_angles:
            # append angle and image to the arrays
            measurements.append(y_angle)
            images.append(x_image)
            data_processed = False
"""
# Save data in numpy arrays
X_train = np.array(images)
y_train = np.array(measurements)

print ("Image data has been modified.")

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

from keras.models import Sequential, Model, load_model
from keras.layers import Activation, Lambda, Flatten, Dense, Reshape, Dropout
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

# Look for file of already trained model
new_model = False

if Path("model.h5").is_file():
    print ("Existing model found.")
    while True:
        variable = input("Use existing model? (y/n)")
        if variable == "y":
            print("Training existing model.")
            break
        if variable == "n":
            new_model = True
            os.remove("model.h5")
            print("Old model removed!")
            break 
else:
    print ("No model found.")
    new_model = True


if new_model == False:
    model = load_model("model.h5")
else:
    # New Model - model based on NVIDIA's DAVE-2
    print ("Training new model.")
    model = Sequential()
    """
    # Convolutional layers
    model.add(Conv2D(24,5,2, input_shape=(80,80,3), activation ='elu'))
    model.add(Conv2D(36,5,2, activation ='elu'))
    model.add(Conv2D(48,5,2, activation ='elu'))
    model.add(Conv2D(64,3,1, activation ='elu'))
    model.add(Conv2D(64,3,1, activation ='elu'))
    
    # Flatten layer
    model.add(Flatten())
    
    # Fully connected layers
    model.add(Dense(200, activation ='elu')) #1164
    model.add(Dense(100, activation ='elu'))
    model.add(Dense(50, activation ='elu'))
    model.add(Dense(10, activation ='elu'))
    model.add(Dense(1))
    """

    # Convolutional layers
    model.add(Lambda(lambda x: x / 255.0, input_shape=(80,80,3)))
    model.add(Conv2D(24,5,2, activation ='elu'))
    model.add(Conv2D(36,5,2, activation ='elu'))
    model.add(Conv2D(48,5,2, activation ='elu')) #48,5,2
    model.add(Conv2D(64,3,1, activation ='elu')) #64,3,1
    model.add(Conv2D(64,3,1, activation ='elu')) #64,3,1
    #model.add(Dropout(0.5))

    # Flatten layer
    model.add(Flatten())
    
    # Fully connected layers
    #model.add(Dense(500, activation ='elu')) #1164
    model.add(Dense(100, activation ='elu'))
    model.add(Dense(50, activation ='elu'))
    model.add(Dense(10, activation ='elu'))
    model.add(Dense(1))
    
    # Compile model with squared error function          
    model.compile(loss='mse', optimizer='adam')
          
# Shuffle data, take 20% as validation data, fit the model
model.fit(X_train, y_train, verbose=1, validation_split=0.2, shuffle=True, nb_epoch=5)

#model.fit_generator(generate_arrays_from_file(data_path + 'driving_log.csv'), samples_per_epoch=100, nb_epoch=10)

# Save model
model.save('model.h5')
print ("Model saved")

