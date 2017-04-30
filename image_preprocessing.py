import random
import cv2
import numpy as np

def pre_process(image, measurement, training):

    if training == True:
        """
        # Add random brightness / darkness
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        brightness = random.uniform(0.5, 1.0)
        hsv[:,:,2] = brightness * hsv[:,:,2]
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        """
        
        # Flip 50% of the images and steering angles to avoid bias in one direction
        if np.random.rand() > 0.5: 
            image = cv2.flip(image,1)
            measurement = -measurement
        
        # Rotate slightly (random)
        num_rows, num_cols = image.shape[:2]
        angle = random.uniform(-5, 5)
        rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), angle, 1)
        image = cv2.warpAffine(image, rotation_matrix, (num_cols, num_rows))

    # Apply mask
    mask = cv2.imread('mask.png',0)
    image = cv2.bitwise_and(image,image,mask = mask)
    # Crop
    image = image[60:140, 6:314]
    # Resize
    image = cv2.resize(image, (80, 80))

    # Convert to YUV
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    # Enhance Contrast
    #image[:,:,0] = cv2.equalizeHist(image[:,:,0])

    return image, measurement
