# Tyre-Anomaly-Detection
Little summer project to study deep learning using Tensorflow and Keras. The aim is to recognise when a tyre has problems or defects by its picture(s)

## What is it?

After a little bit of vacation I could't just wait for classes to start again and so I decided to begin studying deep learning and approaching myself to this field. 
As everything tho, without a project to actually test the theory it would have all resulted to nothing, so I imagined having a device for detecting worn tyres and in general big defects on the surface.

The idea is to build a simple binary classificator that tells me if a tyre is GOOD or DEFECTIVE: let's start simple.

TLDR:


## Table of Content

- [BOM: Bill Of Material](https://github.com/baylonp/Tyre-Anomaly-Detection?tab=readme-ov-file#bom-bill-of-material)
- [Where To Get The Data?](https://github.com/baylonp/Tyre-Anomaly-Detection?tab=readme-ov-file#where-to-get-the-data)
- [What Is A Deep Learning Model And Where To Start Building One](https://github.com/baylonp/Tyre-Anomaly-Detection?tab=readme-ov-file#what-is-a-deep-learning-model-and-where-to-start-building-one)
- [Validation Split = 0.2](https://github.com/baylonp/Tyre-Anomaly-Detection?tab=readme-ov-file#validation_split--02)
- [Data Standardization](https://github.com/baylonp/Tyre-Anomaly-Detection?tab=readme-ov-file#data-standardization)
- [Fine Tuning](https://github.com/baylonp/Tyre-Anomaly-Detection?tab=readme-ov-file#fine-tuning)
- [The Heart](https://github.com/baylonp/Tyre-Anomaly-Detection?tab=readme-ov-file#the-heart)
- [Let's Run It](https://github.com/baylonp/Tyre-Anomaly-Detection?tab=readme-ov-file#lets-run-it)
- [The Most Importan Part: How To Validate A Model?](https://github.com/baylonp/Tyre-Anomaly-Detection?tab=readme-ov-file#the-most-importan-part-how-to-validate-a-model)
- [Image Processing](https://github.com/baylonp/Tyre-Anomaly-Detection?tab=readme-ov-file#image-processing)
- [A Cute DIscovery: Texture Recognition](https://github.com/baylonp/Tyre-Anomaly-Detection?tab=readme-ov-file#a-cute-discovery-texture-recognition)
- [Final Run: Let's Study The Results](https://github.com/baylonp/Tyre-Anomaly-Detection?tab=readme-ov-file#final-run-lets-study-the-results)
  
    - [Pictures Dataset Downloaded From the Internet](https://github.com/baylonp/Tyre-Anomaly-Detection?tab=readme-ov-file#dataset-n1--pictures-dataset-downloaded-from-the-internet)
    - [Pictures Dataset Downloaded From The Internet With Contour On](https://github.com/baylonp/Tyre-Anomaly-Detection/blob/main/README.md#dataset-n2--pictures-dataset-downloaded-from-the-internet-with-contour-on)
  
    - [Pictures Dataset Downloaded From The Internet  + Added Picture Taken By Us](https://github.com/baylonp/Tyre-Anomaly-Detection/blob/main/README.md#dataset-n3--pictures-dataset-downloaded-from-the-internet--added-picture-taken-by-us)
  
    - [Pictures Dataset Downloaded From The Internet With Contour On + Added Picture Taken By Us With Contour On](https://github.com/baylonp/Tyre-Anomaly-Detection/blob/main/README.md#dataset-n4--pictures-dataset-downloaded-from-the-internet-with-contour-on--added-picture-taken-by-us-with-contour-on)
  
 - [Room For Improvement](https://github.com/baylonp/Tyre-Anomaly-Detection/blob/main/README.md#room-for-improvement)

 - [Final notes](https://github.com/baylonp/Tyre-Anomaly-Detection/blob/main/README.md#final-notes)


## BOM: Bill Of material

What I used is:

- Jupyter Notebook in Anaconda environment for Linux
- NVIDIA RTX 2070Super (Offered by [ArkWow](https://github.com/Arkw0w))
- Among all the libraries, I worked alot with Keras and Tensorflow an OpenCV



## Where to get the data?

Let's start by quoting some words I have been found again and again in the endless paper and tutorial that I've followed online:

`A deep learning model is as agood as its dataset`

As I see, this it ultimately true. I started using a dataset of about 2000 pictures of tyres, in which 65% were GOOD tyres, adn 45% DEFECTIVE tyres, but after some trials and errors, I decided to get some help by a friend of mine (Lorenzo Cozzani) and go ask car dealers to take pictures of tyres. What we were looking for was GOOD tyres pictures, especially tread pictures and also close up of defects. We managed to increase the dataset by 500 pictures, evenly distributed among the 2 categories we were lacking.


## What is a deep learning model and where to start building one?
A deep learning model is basically a group of layers that apply to each input some trasformation. These transformations get repetead again and again, usually with a feedback model that re-input the previosu results in the system to better evaluate the so caled weights, they represent connections between nodes in different layers.

In my case I used the `tf.keras.Sequential` model, as I understood it is a good place to start for a beginner. This model is a linear ( sequential ) stack of layers where each layer has exactly one input tensor and one output tensor.

## Validation_split = 0.2
The model doesn't need just a dataset of GOOD and DEFECTIVE pictures, but a subset of these as `training_dataset` and another one as a `validation_dataset`, on which to test the results obtained from the training on the `training_dataset` and on which metrics will be computed to evalutate the performance of the model.

Code-wise, this is expressed in python using the function `tf.keras.utils.image_dataset_from_directory()` specifying the parameter `validation_split = 0.2`, meaning that 80% of the images will be used for training and 20% for validation.

## Data standardization

Before building te actual model, it is important to standardize the values we use to describe a picture. Generally, each RGB channel value is in the range of [0,255], but neural networks  wants small input values. The main reason is because the network has to deal with very big number during the trainig and if we don't scale them, these big number can actually stall the learning or making it extremely inefficient. Usually, activation function used by neural networks, can saturate easily when dealing with big numbers, so we resize.

how do we do that? We use the `tf.keras.layers.Rescaling()` function that will end up looking like this:

`normalization_layer = layers.Rescaling(1./255)`


You may ask yoursel how do we actually use it. Keep reading.

## Fine tuning
Usually, this step is performed later after noticing how the actual model has performed after the first run. The keep away from what is referred to as "overfitting" we added a `dropout` layer and a `data_augmentation` layer.
As the offical Tensoflow documentation states: 

#### Dropout regularization removes a random selection of a fixed number of the units in a network layer for a single gradient step. The more units dropped out, the stronger the regularization.


The actual code added to the final model is: `layers.Dropout(0.2)` where 0.2 meaning that it drops out 20% of the ouput units randomly from the applied layer


Data augmentation on the other hand is useful when there is a small number of training examples. Wha it does is generating additional pictures by taking the original ones and flipping, zooming and rotating them. This helps expose the model to more aspects of the data and generalize better.

It will be implemented by using `tf.keras.layers.RandomFlip()`, `tf.keras.layers.RandomRotation()` and `tf.keras.layers.RandomZoom()`.
Code will look like this:

```
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)
```


## The heart

If you read this far, now it is time to show you the heart of the machine. The actual layers of the model.

```
model = Sequential([

  tf.keras.Input(shape=(img_height, img_width, 3)),
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes, name="outputs")
])

```

Here you can see the different layers which the model is composed of. First there is the `input layer` that set the shape  decided for the trianing, then the `data_augmentation` layer ( see above ). After the rescaling layers there are layers that we didn't meet before. What are they?

- `layers.Conv2D()` this layer applies 16 different filters (kernels) to the images to study all of its features. We see hoe later it is applied 2 ore times with 32 and 64 filters to better extract all the features from the pictures. Attention shoudl be paid to the type of the activation function, in this cased I used the ReLU (Rectified Linear Unit). Activation funtions brings non linearity into the models, since otherwise big models wouuld  be unable to "learn" and aply the rules to never seen complex task, resulting in just linear regression models. See more [here](https://www.v7labs.com/blog/neural-networks-activation-functions)

- `layers.MaxPooling2D()` this layer divides the input feature map into a pool of smaller regions (usually squares) and takes the maximum value from each region. This operation helps to keep the strongest features while discarding irrelevant information.

- `layers.Flatten()` It reshapes data, going from a multi-dimensional feature maps into a single dimension

## Let's run it

Finally, it is time to run it and wait for it finish. 

From the screenshot below, we can see that the model will be trained on a total of 22MLN parameters

![Trainable Parameters](https://github.com/baylonp/Tyre-Anomaly-Detection/blob/main/images/Dataset_1856/contouring1.png)

We compile the code and we choose how many epochs ( the number of time the model will see the data)

```
epochs = #epochs
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
```


## The most importan part: How to validate a model?

To validate a model, we used different metrics. First of all the graphs created after the run ( code provided ) need to show a trend in which Training_Accuracy and Validation_Accuracy are very close to each other in a way such that the infamous Overfitting is not present.

Here are some examples of Accuracy and Loss graphs.

![Less Overfitting](https://github.com/baylonp/Tyre-Anomaly-Detection/blob/main/images/graph2.png)
![Overfitting present](https://github.com/baylonp/Tyre-Anomaly-Detection/blob/main/images/graph_1.jpg)

As you can see, in the second picture, the curves are very distant from one another. A clear sign of overfitting.

Other important metrics we used to validate our model were:
- **Precision**: **TP/(TP +FP)**
  The precision is calculated as the ratio between the number of Positive samples correctly classified(TP)
to the total number of samples classified as Positive (either correctly(TP) or incorrectly(FP)).  **The precision
measures the model's accuracy in classifying a sample as positive**. When the precision is high, you
can trust the model when it predicts a sample as Positive. Thus, the precision helps to know how
the model is accurate when it says that a sample is Positive.

- **Recall**: **TP/(TP +FN)**
The recall is calculated as the ratio between the number of Positive samples correctly classified as
Positive(TP) to the total number of Positive samples(TP + FN). The recall measures the model's ability to detect
Positive samples. **The higher the recall, the more positive samples detected.**
The recall cares only about how the positive samples are classified. This is independent of how the
negative samples are classified, e.g. for the precision. When the model classifies all the positive
samples as Positive, then the recall will be 100% even if all the negative samples were
incorrectly classified as Positive.

- **F1-Score:**
The F1 score is the harmonic mean of precision and recall. It provides a single metric that balances
the trade-off between precision and recall. The F1 score is useful when you need a balance between
precision and recall, and there is an uneven class distribution. It ranges from 0 to 1, where 1
indicates perfect precision and recall.

- **Confusion Matrix:**
It visually represent how the model scored in classifying never-seen-before images.

[Know more about the metrics -1 ](https://blog.paperspace.com/deep-learning-metrics-precision-recall-accuracy/)

[Know more about the metrics -2 ](https://medium.com/enjoy-algorithm/methods-to-check-the-performance-of-the-classificationmodels-55ec50e0a914)

Here is an example of a confusion matrix:
![Confusion Matrix](https://github.com/baylonp/Tyre-Anomaly-Detection/blob/main/images/confusion_matrix.png)

Ideally, you would want to have 1 in PredictedGood-TrueGood and 1 in PredictedDefective-TrueDefective

The code for the confusion matrix is provided, but it all resort to this:


```
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_pred = model.predict(test_ds)
y_pred_labels = np.argmax(y_pred, axis=1)

y_true_labels = np.concatenate([y for x, y in test_ds], axis=0)
cm = confusion_matrix(y_true_labels, y_pred_labels)

#normalize, so that the data is in percentages
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.show()

```


## Image processing

As of now, the model works but it is not that good. We need to optimize it a bit. And we can optimize it by feeding it with better data. How do we improve our data?

Via the python image processing library I thought of enhancing a bit more the data quality, showing to the model what were the actual feature it had to be trained on.

My journey through image processing took me to 8 different filters applied to the pictures, to end up with a filter that add the contour of the actual tyre defects to the picture. So I had the idea to try to feed this type of picture to the model, a sort of picture on steroid in which the defects are highlighted.

In my head this would improve the model since a non defective tyre has contours that are mostly symmetrical and right, not just scattered around like  defective tyre.

As you can see, this picture of a GOOD tyre has a more symmetrical contour line

![Good](https://github.com/baylonp/Tyre-Anomaly-Detection/blob/main/images/good_contour.png)


This one instead has a more scattered plot, due to the defect.

![Defect](https://github.com/baylonp/Tyre-Anomaly-Detection/blob/main/images/defect_contour.png)


In general, as I needed to have a general picture of what the filters were doing to the images, I wrote a little code to show them, and this is the cute output result.

This is the result.

![The whole](https://github.com/baylonp/Tyre-Anomaly-Detection/blob/main/images/whole_picture.png)

Later I will show you the difference it made regarding the metrics, but I can ancticipate you that the results were not that better, I would say even worse.


## A cute discovery: Texture Recognition

Since I wanted to push the work a little bit forward and try something new, I decided to sudy a little bit Texture Recognition. Basically i computed the Gray-Level Co-occurence Matrix (GLCM) to etract the following feature: Contrast, Correlation, Energy and Homogeneity. But none showed some interesting results.

So I kept studying and decided to apply the Gabor Filter to pictures. This filter has been shown to possess optimal localization properties in both spatial and frequency domains and thus is well-suited for texture segmentation problems. 

It is a linear, bandpass filter useful for feature extraction. 

This filter tells us when a tyre is very very old, can detect the wrinkles in it and all the not-common lines


**The original tyre:**

![Original](https://github.com/baylonp/Tyre-Anomaly-Detection/blob/main/images/Defective%20(878).jpg)


**The gabor Filtered tyre:**

![Gabor Filtered](https://github.com/baylonp/Tyre-Anomaly-Detection/blob/main/images/gabor_defective.png)


As you can see, all the wrinkles get accentuated. This is very cool.


## Final run: let's study the results

I will show you now the different results we had with the various dataset. We tested the model resulted from the training on these 4 dataset with a test batch of pictures: 10 GOOD and 10 DEFECTIVE downloaded from the internet (of course, not part of the datasets)

- Dataset N.1 = Pictures dataset downloaded from internet
- Dataset N.2 = Pictures dataset downloaded from internet **with contour on** 
- Dataset N.3 = Pictures Dataset downloaded from internet + added picture taken by us
- Dataset N.4 = Pictures Dataset downloaded from internet **with contour on** + added picture taken by us **with contour on**


  ### Dataset N.1 : Pictures dataset downloaded from internet
  With this dataset we obtained:
  - Precision = 0.76
  - Recall = 0.71
  - F1_score = 0.71

  ![Datset1](https://github.com/baylonp/Tyre-Anomaly-Detection/blob/main/images/Dataset_1856/no_contouring2.png)

  ![Dataset_2_matrix](https://github.com/baylonp/Tyre-Anomaly-Detection/blob/main/images/Dataset_1856/no_contouring4.png)


  ### Dataset N.2 : Pictures dataset downloaded from internet with contour on 

  With this dataset we obtained:
  - Precision = 0.66
  - Recall = 0.65
  - F1_score = 0.64

  ![Datset2](https://github.com/baylonp/Tyre-Anomaly-Detection/blob/main/images/Dataset_1856/contouring2.png)
  ![Dataset_2_matrix](https://github.com/baylonp/Tyre-Anomaly-Detection/blob/main/images/Dataset_1856/contouring4.png)


  ### Dataset N.3 : Pictures Dataset downloaded from internet + added picture taken by us

  With this dataset we obtained:
  - Precision = 0.86
  - Recall = 0.85
  - F1_score = 0.86 

 ![Datset3](https://github.com/baylonp/Tyre-Anomaly-Detection/blob/main/images/Dataset_2226/no_contouring_2.png)
  ![Dataset_3_matrix](https://github.com/baylonp/Tyre-Anomaly-Detection/blob/main/images/Dataset_2226/no_contouring_4.png)


  ### Dataset N.4 : Pictures Dataset downloaded from internet **with contour on** + added picture taken by us with contour on

  With this dataset we obtained:
  - Precision = 0.55
  - Recall = 0.55
  - F1_score = 0.55

 ![Datset3](https://github.com/baylonp/Tyre-Anomaly-Detection/blob/main/images/Dataset_2226/contouring_2.png)
 ![Dataset_3_matrix](https://github.com/baylonp/Tyre-Anomaly-Detection/blob/main/images/Dataset_2226/contouring_4.png)


## Room for improvement

There is much more to do, starting from fine tuning the model better to  implement a precise defect detection on the tyre using [Mask_RNCC](https://github.com/matterport/Mask_RCNN)

## Final Notes

This project could have not been complted succesfully without the help of [Lorenzo Cozzani](https://github.com/Arkw0w) who spent time with me taking pictures of tyres and helped running the model on his juicy NVIDIA RTX 2070Super

