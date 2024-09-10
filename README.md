# Tyre-Anomaly-Detection
Little summer project to study deep learning using Tensorflow and Keras. The aim is to recognise when a tyre has problems or defects by its picture(s)


## What is it?

After a little bit of vacation I could't just wait for classes to start again and so I decided to begin studying deep learning and approaching myself to this field. 
As everything tho, without a project to actually test the theory it would have all resulted to nothing, so I imagined having a device for detecting worn tyres and in general big defects on the surface.

The idea is to build a simple binary classificator that tells me if a tyre is GOOD or DEFECTIVE: let's start simple.

## BOM: Bill Of material

What I used is:

- Jupyter Notebook in Anaconda environment for Linux
- NVIDIA RTX 2070Super
- Among all the libraries, I worked alot with Keras and Tensorflow



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

