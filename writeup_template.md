#**Traffic Sign Recognition** 

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
[image2]: ./examples/image_org.png "Original image"
[image9]: ./examples/image_norm.png "Grayscaled and Nomralized image"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/achtung_vorfahrt.thumbnail.jpg "Yield"
[image5]: ./examples/baustelle.thumbnail.jpg "Road work"
[image6]: ./examples/images.thumbnail.jpg "Speed limit (30km/h)"
[image7]: ./examples/Vorfahrt.thumbnail.jpg "Priority road"
[image8]: ./examples/vorfahrt_an_naechster_kreuzung.thumbnail.jpg "Right-of-way at the next intersection"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####README that includes all the rubric points and how you addressed each one. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/christiankay/SDC_projects_P2_deep_traffic_signs/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Summary of the data set. 

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. First, an image example of each class/label of the data set is plotted with the numeric label according to the sign name provided by a excel sheet. Additionally, a bar graph shows the distribution of the data with respect to the classes. Finally the minimum and maximum data/images per class are given.


![alt text][image1]

###Design and Test a Model Architecture


As a first step, I decided to convert the images to grayscale because in a direct comparision with three color images, the prediction performance was higher.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2] ![alt text][image9]

As a last step, I normalized the image data to avoid bad conditioning of the training data and to to improve the numeric stability / performance of the model.




####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x128 				|
| Convolution 5x5	    |1x1 stride, valid padding, 10x10x196    									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x196 				|
| Fully connected		| 1100       									|
| RELU					|												|
| Fully connected		| 43      									|

 


####3. Training of the model

To train the model, I used an AdamOptimizer to reduce cross entropy. I used 20 epochs with a learning rate of 0.001 and batch size of 125. For initial parameter set up the normal distrition function were set to mu = 0 and sigma = 0.01.

####4. Approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. 

My final model results were:
* training set accuracy of 0.987
* validation set accuracy of 0.940 
* test set accuracy of 0.901

The basis of my architectur is provided by LeNet (LeCun) which has been used during the carnd-term1 course teaching.
After first trys with the original model structure I was not able to reach 0.93 validation accuracy. Due to that I decided to modify the model. I removed one fully connected and relu function layer to reduce the models complexity and to avoid overfitting. I decided to increase the filter depth to 128 respectivly 196 in the convolutional layers to gather more different features from the convolutional layers.


 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield	      		| Yield   									| 
| Road work     			| Road work 										|
| Speed limit (30km/h)				| Pedestrians											|
| Priority road      		| Priority road					 				|
| Right-of-way at the next intersection			| Right-of-way at the next intersection     							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of german traffic sign database.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a yield sign (probability of 0.99), and the image does contain a yield sign. The top five soft max probabilities were 35,9,28,10 (Ahead only, No passing, Children crossing, No passing for vehicles over 3.5 metric tons). 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Yield   									| 
| .53     				| Road work 										|
| .69					| Pedestrians										|
| .99	      			| Priority road					 				|
| .98				    | Right-of-way at the next intersection      							|




### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


