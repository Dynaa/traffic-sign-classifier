# **Traffic Sign Recognition** 

## Writeup

### The goal of this project was to design and train a model for traffic sign recognition. 


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README


You're reading it! and here is a link to my [project code](https://github.com/Dynaa/traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Dataset summary
The dataset is composed of images coming from the German Traffic Signs Dataset.

The pickled data is a dictionary with 4 key/value pairs:

-   `'features'`  is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
-   `'labels'`  is a 1D array containing the label/class id of the traffic sign. The file  `signnames.csv`  contains id -> name mappings for each id.
-   `'sizes'`  is a list containing tuples, (width, height) representing the original width and height the image.
-   `'coords'`  is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image.

I used numpy and pandas libraries to calculate summary statistics of the traffic
signs data set :
* Number of training examples = 34799
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

I display randomly some images of the dataset. 

![enter image description here](https://github.com/Dynaa/traffic-sign-classifier/blob/master/readme_images/sample_images.png)


#### 2. Include an exploratory visualization of the dataset.

By using pandas and numpy, I extract some informations about the constitution of the dataset. We can see the distribution chart for the train, validation and test dataset. 

* No. of training images: 34799
* No. of classes:    43
* Class with the maximum data :  2
* Number of training of max class : 2010
* Mean distribution : 809

![enter image description here](https://github.com/Dynaa/traffic-sign-classifier/blob/master/readme_images/distribution.png)

### Design and Test a Model Architecture

#### 1.  Dataset enrichment 

First of all, I transform dataset to grayscale and apply normalization on the dataset. 
Secondly, as describe in [Sermanet, 2011], I choose to augment my dataset by using some operation on the images. Samples are randomly perturbed in position ([-2,2] pixels), in scale ([.9,1.1] ratio) and rotation ([-15,+15] degrees). Here is an example of an original image and an augmented image :

![enter image description here](https://github.com/Dynaa/traffic-sign-classifier/blob/master/readme_images/augmentation.png)

In order to determine the number of images to create, I decided to use the max value of each class, and create random sample based on this number mulplied by 4 in order to port the number of samples to 310921.


#### 2. Final model architecture : 

I started by using LeNet architecture, containing the following layers :
![Lenet architecture](https://github.com/Dynaa/traffic-sign-classifier/blob/master/readme_images/lenet.png)


#### 3. Model parameters

I used the SciKit Learn train_test_split function to create a validation set out of the training set. I used 20% of the testing set to create the validation set.

#### Hyperparameters

I’ve chosen an Epoch length of 30, a batch size of 128 and a learning rate of 0.0005, but feel free to play around with these values. I also use an Adam optimizer instead of say, Stochastic Gradient Descent.

#### Additional modifications

I’ve also added dropout layers to my CNN with a keep-drop probability of 0.8 to reduce overfitting. A quick explanation of the way dropout works is activations are randomly chosen to be ignored or dropped out. The NN can no longer depend on any one activation in the net, so it must learn redundant representations of everything: It reduces overfitting since every representation requires more 'votes'.

#### 4. Final results

My final model results were:
* Validation Accuracy = 99.280% 
Test Accuracy = 94.9%

### Test a Model on New Images

#### 1. New signs

In order to test my model on new images, I used Google Street View in Munich area to pick some traffic sign. 
Here are five German traffic signs : 
![enter image description here](https://github.com/Dynaa/traffic-sign-classifier/blob/master/readme_images/new_images1.png)
![enter image description here](https://github.com/Dynaa/traffic-sign-classifier/blob/master/readme_images/new_images2.png)
![enter image description here](https://github.com/Dynaa/traffic-sign-classifier/blob/master/readme_images/new_images3.png)
For every images, I apply some transformations as resizing, and obviously normalization and grayscale convertion. 

#### 2. Prediction

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield      			| Yield   										| 
| Turn right ahead     	| No entry 										|
| Priority road			| Priority road									|
| Keep right	      	| Keep right					 				|
| Speed limit (30km/h)	| Speed limit (30km/h)d      					|
| No entry				| No entry										|

The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83%. This is result is less than the accuracy obtained with the previous test data set. This could be due to the small size of this set. 

#### 3. Softmax score

Correct label for image 0  is  13 
 Scores [1.0000000e+00 1.8908402e-09 4.8804509e-11 2.7348561e-11 1.3017260e-11] over labels [13  3 15 35 12]

Correct label for image 1  is  33 
 Scores [9.9982065e-01 1.1936639e-04 4.3107677e-05 1.2213248e-05 1.7992135e-06] over labels [17 38 40  1 37]

Correct label for image 2  is  12 
 Scores [9.9999988e-01 8.4182439e-08 3.4915644e-13 2.8882170e-13 1.8907495e-13] over labels [12 40  3 42 38]

Correct label for image 3  is  38 
 Scores [1.0000000e+00 4.1368877e-17 1.8340459e-17 1.7449915e-18 5.3984366e-19] over labels [38 37 23  1 11]

Correct label for image 4  is  1 
 Scores [0.65047467 0.165693   0.14257255 0.02756177 0.00720835] over labels [ 1  2 14  3  0]

Correct label for image 5  is  17 
 Scores [1.0000000e+00 4.2816973e-15 9.2483938e-21 1.0141998e-21 9.7901814e-22] over labels [17  9 14 32 33]
 
### Result

We have 83% accuracy, but the softmax scores let us examine  **_confidence_** : 
- If we look at the first image, score for the first have a large margin, reflecting high confidence. 
- On the other hand, if we look at the forth image, the confidence is not so high. 
