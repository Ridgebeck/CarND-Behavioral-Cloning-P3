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
[image8]: ./Writeup/image_processing/flipped_pic.png "Flipped Picture"
[image9]: ./Writeup/final_model_architecture.JPG "Final Model Architecture"
[image10]: ./Writeup/image_processing/modified_pic.png "Modified Picture"




## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* clone.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* image_preprocessing.py contains the augmentation of the testing set data
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* recording_track_1.mp4 shows a video of the model on the first track

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The clone.py file contains the code for training and saving the convolution neural network. It shows the pipeline for training and validating the model and it contains comments to explain how the code works. The image_preprocessing.py file shows image preprocessing steps.

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

In order to avoid bias into one direction 50% of the images of the traning set (chosen randomly) were flipped horizontally.

![alt text][image8]

I also added some random rotation and shifts as well as a random adjustment of brightness in order to help the model to generalize better. 

![alt text][image10]

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

See *2.* above.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a simple model and test the learning process with preprocessed data. After the image preprocessing was completed I moved towards more complex architecture that took a lot longer to train. I tried LeNet and achieved decent results and moved to the Nvidia Dave-2 architecture in the end.

#### 2. Final Model Architecture

The final model is based on the Nvidia Dave-2 network and consists of 5 convolutional layers and 4 fully connected layers. I added a dropout layer after the convolutional layers to prevent overfitting and to make the model more robust. Here is a visualization of the architecture of the Nvidia network:

![alt text][image1]

...and here you can see my implementation in keras:

![alt text][image9]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps counter-clockwise on track one using center lane driving. In order to get a higher resolution on the steering angle I used the mouse to control the car. Then I recorded two laps with the same technique going the other way (clockwise) and saved the data in a separate file. Because the course is mostly straight and in order to avoid a bias towards steering straight I recorded three rounds in each direction only driving through the curvy sections of the course.

The steps for data augmentation are described in _Attempts to reduce overfitting in the model_.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.
