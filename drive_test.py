import csv
import cv2
import argparse
import numpy as np
from keras.models import load_model
import image_preprocessing as ip

model = load_model("model.h5")

# Open csv file in data folder
lines=[]
with open('./test_data/test_data.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Read all images and steering angles and save them in lists
images = []
measurements = []
  
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './test_data/' + filename
    image = cv2.imread(current_path)
    measurement = float(line[3])
    image, measurement = ip.pre_process(image, measurement, False)
    cv2.imshow(filename, image)
    images.append(image)
    measurements.append(measurement)

# Convert lists to numpy arrays
image_test = np.array(images)
angle_test = np.array(measurements)

# Let the model predict the steering angle
print ((model.predict(image_test)))
print (angle_test)
