import csv
import cv2
import argparse
import numpy as np
from keras.models import load_model

#model = None

model = load_model("simple_model.h5")

# Open csv file in data folder
lines=[]
with open('./simple_test/simple_test.csv') as csvfile:
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
image_test = np.array(images)
angle_test = np.array(measurements)

# Let the model predict the steering angle
print (model.predict(image_test))



