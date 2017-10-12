# Car Tracking

This project copes with the identification of cars in a video. To do so, several functions are used to extract features from an image. Those functions can e.g. be:

* Color Histogram
* Gradient Information
* Spatial Binning

and many more. Also, using different color spaces of images can generate a feature space. With that feature space, a classifier can be identified, that tries to separate car from non car images.
In the end this classifier can be used to identify cars on images. Using filtering, false positives can be filtered out, and true positives can be brought down to a bounding box calculation, which can be used to predict the future position of the car in the image.

## Feature Extraction

To start out on this project, different approaches to feature extraction are tested. This way, we can identify useful features, which can be given to a classifier to identify car images, and to differ them from non car images. First off, we want to run some sanity and balancing checks. Is a class overrepresented?

[balance]: ./examples/class_balance.PNG "Class Balance"
![alt text][balance]

As we can see from the image, both classes are equally represented.

[sanity]: ./examples/sanity.PNG "Dataset Test"
![alt text][sanity]

We can see that the dataset is ok and the right classes are provided for the images, based on a random subset of data.

### Feature Extraction

To extract features from the images, I started off with the principles learned in the course.

The first approach taken was the HOG extraction, which is coming with the scikit library.

[hog]: ./examples/hog.PNG "HOG Test"
![alt text][hog]

We can see the gradients for the car and noncar images below. Now we can try to train a classifier on the informations provided by this feature. Applying the HOG to all three color channels and training with a linear classifier yields an accuracy
of 92.7%, however the feature vetor has 5200 features and is quite slow.

If we want to speed up performance, we can apply the HOG to just one color channel, this reduces the feature count to 1400, but the accuracy is reduced to 89.9% only. 

### Adding more Features

In the next step I want to add more features to the pipeline. To start off, the color histogram can be a good indicator.

Of course, colors itself are rather uninformative, however, using the Hue / Light / Saturation colorspace, we can focus on the Saturation channel. Cars tend to be in saturated colors, which should actually differentiate them from their surroundings

[hist]: ./examples/hist.PNG "Color Histogram Test"
![alt text][hist]

The color histogram shows much higher values for car images, thus a clear difference can be made out even by eye. An algorithm can leverage that information even better to classify the car images.

In a next step I also want to introduce spatial binning of the image. In this way we are going to reduce the image size and input ALL the pixels to the training algorithm. This way we can put all the color information in the algorithm too.

To combine the features, all the features are combined in an 1D array. In the first run I could achieve an Accuracy of 98.4% using the follwing parameters: 

* HOG orient 9
* HOG pixel per cell 16
* HOG cells per block 2
* Saturation Histogram of LUV space, 64 bins
* 8, 8 binned image

Let's see if we can optimize the parameters for the Support Vector Machine. Using a smaller C value to teach the Support Vector Machine can reduce overfitting. Also, using a nonlinear Kernel could also improve the accuracy. Here are the accuracies that could be reached throughout the optimization process.

* Linear SVC (Standard Parameters) : 98.4 %
* Linear SVC (C = 0.1) : 98.5%
* Linear SVC (C = 10.0) : 98.4%
* Nonlinear SVC (Standard Parameters) : 99.3 %
* Nonlinear SVC (C = 0.1) : 98.0%
* Nonlinear SVC (C = 10.0) : 99.4%
* Keras Neural Network : 98.86 %

Another important benchmark, instead of just accuracy is also the execution time of the algorithm. Using the 1164 features described above, an execution time from image -> features -> algorithm -> result was calculated:

* Linear SVC: 0.0034 seconds
* Nonlinear SVC: 0.0060 seconds
* Keras: 0.0110 seconds

As we can see the linear SVC is the fastest algorithm, but has some shortcomings in accuracy.  That's why for the final implementation I sticked with the Noninear SVC, because it shows a good performance and a high accuracy.

## Sliding Window on Images

In the next step after I trained my classifier I had a look at the training images to run it on. As a next step, we define a region of interest, which will define the search are where we want to find cars

[aoi]: ./examples/aoi.PNG "Area of Interest"
![alt text][aoi]

Let's test the classifier on a sample image:

[classifier_test]: ./examples/classifier_test.PNG "Classifier Test"
![alt text][classifier_test]

As we can see, it is able to classify the image correctly.

To reduce the computational road, small sliding windows are only calculated further up in the image, where small cards are to be expected, the closer we get to the car, the larger the sliding windows are.

Using such a separation in distance and size of the windows allows us to run less windows over each image. In the end the computational load increases with each window that we have in the image.

For the sliding window, I specified multiple areas and multiple sized for the window. This way smaller cars are detected at the far end of the road, whereas bigger cars are detected closer to the car itself.

[slidingwindow]: ./examples/slidingwindow.PNG "Sliding Windows"
![alt text][slidingwindow]

## Run the Pipeline

A first run of the algorithm with the sliding windows shows a relatively good performance

[firstRun]: ./examples/firstRun.PNG "First run of classifier"
![alt text][firstRun]

To improve the false positive rate, the heatmap was averaged over the last 8 images. because of the continuous driving and appearance of cars, this should still return a good estimate for the car positions. Also, car positions are estimated, and any change above
70 pixels is discarded as false positive.
The final cropped boxes are then run through the classification algorithm again, to make sure a car image was captured in that exact area.

# Summary

To summarize this pipeline, I think that the svc classifier is a bit too slow. Also the HOG sampling needs a lot of calculation, even with the global method provided in the lessons, with the 4 different sampling window sizes, it still requires 4 HOG over the whole image, which I deem rather inefficient.

In this case, I would probably use a fast convolutional network trained to identify pedestrians and cars in an image. Systems such as darknet can run really fast on embedded devices and are perfectly applicable for such an application.

All in all it's still a nice insight into the identification of cars in images, and how a systematic approach looks like.

I think the biggest weakness of this algorithm is still it's hyperparameter space, which are dozens of parameters to tune in the end. Also the calculations based on certain colorspaces may be uncertain for rainy or foggy conditions, which are much easier to train in a neural network or similar architecture.

Unfortunately, I didn't get the hog-algorithm to run more efficiently when compiled on the whole image, although I calculated it only on the area of interest each time, the code was up to 3 times slower than when compiled on every single frame.