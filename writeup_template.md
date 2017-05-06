# **Behavioral Cloning** 

## Writeup for Project 3 - Behavioral Cloning

---

[//]: # (Image References)

[image1]: https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-768x1095.png "Network Structure"
[image2]: ./Writeup/data_distribution/steering_angle_1_track_original.png "Data Distribution"
[image3]: ./Writeup/data_distribution/steering_angle_1_track_flipped.png "Data Flipped Images"
[image4]: ./Writeup/data_distribution/steering_angle_1_track_maximum.png "Maximum Number of Angles"
[image5]: ./Writeup/data_distribution/steering_angle_1_track_3_cameras.png "Three Cameras"
[image6]: ./Writeup/image_processing/original_pic.png "Original Picture"
[image7]: ./Writeup/image_processing/cropped_pic.png "Cropped Picture"
[image8]: ./Writeup/image_processing/brightness_pic.png "Random Brightness"
[image9]: ./Writeup/image_processing/mask_pic.png "Masked Picture"




## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows image preprocessing steps, the pipeline for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the Nvidia Dave-2 Network and consists of 5 convolutional layers and 4 fully connected layers. For a more detailed description see _Model Architecture and Training Strategy_.

#### 2. Attempts to reduce overfitting in the model

The model was first trained on a small dataset to see if overfitting would happen as expected. If the algorithm is well designed to learn from the small dataset and overfitting happens at a high degree, it is proven that the algorithm has looked at the correct features to make its prediictions correctly. The problem is that it looked too closely at some details and more or less memorized the different images instead of making generalized rules. The overfitting can than be reduced in order to allow generalization for different situations to make the use of the algorithm practical for driving a car under different conditions.

Important for reducing overfitting was collecting and preprocessing the data. The image preprocessing steps can be found in the file _image_preprocessing.py_. During the first tests on training data for only one track the model was steering towards the left. A look at the data shows that due to the fact the course was going in a circle the model was trained to be biased to steer more towards the left side.

![alt text][image2]

This was later reduced by recording more data from multiple laps driving in different directions. On top of that roughly every second image (random) was horizontally flipped and the steering angle was negated. On the example data of only one lap the data looks like this:

![alt text][image3]

Because the course has a lot of straight sections the data had to be augmented in order to avoid a bias to steer straight. Therefore additional data that was recorded on the curved sections of the track was added. While reading in the data it was also made sure that no steering value could have more than a certain amount of samples to get a more balanced data set.

![alt text][image4]

Furthermore, the images of the left and right camera were used and an offset to the recorded steering angle was applied.

![alt text][image5]

Here is an example of a original picture from a center camera:

![alt text][image6]

The image was cropped to remove the details on the horizon and the hood of the car which are not of particular interest to predict the steering angle.

![alt text][image7]

In order to allow faster training the image was also resized to 80 x 80 pixels. The image data was also randomly adjusted in brightness and randomly rotated in order to create "new data" every time the model was fed with the same original images.

![alt text][image8]

I also applied a mask over the center section of the road in order to let the algorithm focus more on the sides of the picture and the curvature of the street. 

![alt text][image9]

The model was trained and validated on different data sets to ensure that the model was not overfitting.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

See *2.* above.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:



I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
