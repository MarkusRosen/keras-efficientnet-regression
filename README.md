# Transfer Learning with EfficientNet for Image Regression in Keras - Using Custom Data in Keras

![teaser](./teaser.png)

This is the Repo for my recent blog post: [Transfer Learning with EfficientNet for Image Regression in Keras - Using Custom Data in Keras](https://rosenfelder.ai/keras-regression-efficient-net/)

There are hundreds of tutorials online available on how to use Keras for deep learning. But at least to my impression, 99% of them just use the MNIST dataset and some form of a small custom convolutional neural network or ResNet for classification. Personally, I dislike the general idea of always using the easiest dataset for machine learning and deep learning tutorials since this leaves many important questions unanswered. Adapting these tutorials to a custom dataset for a regression problem can be a daunting and time-consuming task with hours of Googling and reading old StackOverflow questions or the official Keras documentation. Through this tutorial, I want to show you how to use a custom dataset and use transfer learning to get great results with very little training time. The following topics will be part of this tutorial:

- use ImageDataGenerators and Pandas DataFrames to load your custom dataset
- augment your image to improve prediction results
- plot augmentations
- adapt the state-of-the-art EfficientNet to a regression
- use the new Ranger optimizer from `tensorflow_addons`
- compare the EfficientNet results to a simpler custom convolutional neural network

