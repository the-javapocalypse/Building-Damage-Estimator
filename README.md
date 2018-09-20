# FYP


## Brief Overview

### Version 1 (Trannsfer Learning)
This folder contain scripts which use pre trained VGG 16, Inception, Resnet, Mobilenet for classification of relevant or irrelevant images. We will lookout for features that were common in images of earthquake and based on collective probabilities of top 5 features predicted by pre trained models, we will decide if the image is useful for us, i.e either it is relevant or irrelevant (Eg. Meme) and will move the relevant images to filtered directory

### Version 2 (Retrained Final Layer of VGG)
This folder contains script that will train the model using Transfer Learning (VGG 16) to do binary classification for classes:
* Damaged
* Preserved
This should provide us a good starting point and set standards for base accuracy.

### Version 3 (Damage Estimator Low/Moderate/High)
This folder contains script in which we will train a Convolutional Neural Network and substitute it with the final layer of VGG 16 to classify a given image on the basis of model confidence (probability) as:

* Not Damaged/Irrelevant
* Moderately Damaged
* Highly Damaged

We will also compare different architectures to find the best hyper paramters for the model.

### Version 4 (NLP)
In this folder, there are scripts which uses different techniques to extract information, like number of people killed, from textual data (tweets and captions). The results are not satisfactory, but however provided a bases for NLP part.


## Contribution

If you are contributing to the image/text part:
* Create a new folder. Name it as
```
Version X (Feature Description)
```
* Create a new branch and name it using the following scheme
```
code-yourName-featureDescription
```
* Submit a merge request.


If you are contributing to the documentation part:
* Pull from origin
* Place/Update the files in documentation folder.
* Create a new branch and name it using the following scheme
```
documentation-yourName-featureDescription
```
* Submit a merge request.



## Dependencies
Following are the dependecies.
* Python 3.6 or greater
* Keras
* Tensorflow
* NLTK
* Textblob
* Numpy
* Pandas
* Scikit 
* Matplotlib


## Authors
* Muhammad Ali Zia
* Saad Kamran
* Sarim Balkhi

