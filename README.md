# Building Damage Estimator


## Brief Overview

### Dataset
Initially we had a dataset of Nepal earthquake which had 6,529 images and 6,867 tweets. There was a lot of irrelevant images in the dataset.

Later, we separated images manually to train the model on data with minimum noise. This dataset had 420 training images and 80 testing images. But this data was subjective to our own thinking.

We then created a website to label this dataset which we manually created. The aim was to eliminate subjectivity.
![enter image description here](https://ucarecdn.com/e0967bef-b95a-49d8-96c1-48f6084e0b54/datalabelwebsite.PNG)

We later find two crisis datasets which were already labeled by humans via [Figure Eight](https://www.figure-eight.com/) (previously known as Crowd Flower). One dataset was used in paper ['Practical Extraction of Disaster-Relevant Information from Social Media'](https://arxiv.org/pdf/1805.00713.pdf) and second in ['Damage Assessment from Social Media Imagery Data During Disasters'](https://mimran.me/papers/Damage_assessment_from_Social_Media_Images.pdf)

### Version 1 (Transfer Learning)

**Motivation**
We needed a baseline to evaluate our future models. The best way to begin the project was setting a baseline through transfer learning.

**Description**
We used following pre-trained models for classification of relevant and irrelevant images. 

 - VGG 16 
 - Inception 
 - Resnet 
 - Mobilenet

We gathered features that were common in images of earthquake which are as follows. 
```
labels = ['knee_pad', 'stretcher', 'crash_helmet', 'cliff', 'ambulance', 'lakeside', 'half_track',
'garbage_truck', 'fire_engine', 'patio', 'plow', 'barrow', 'nail', 'hatchet', 'lumbermill',
'chain_saw', 'wood', 'military_uniform', 'rock', 'cobra', 'assault_rifle', 'syringe', 'mask',
'lifeboat', 'mountain', 'prison', 'swab', 'crutch', 'jinrikisha', 'hen-of-the-woods', 'tractor',
'snake', 'dwelling', 'church', 'monastery', 'band_aid', 'bath_towel', 'airliner', 'aircraft_carrier',
'shovel']
```

On the basis of collective probabilities of top 5 features predicted by pre-trained models, we decided if the image is useful for us, i.e either it is relevant or irrelevant (Eg. Meme) and moved the relevant images to a separate filtered directory

**Results**
We used this model to filter out irrelevant images from the data. Initially the dataset had a little over 6000 images and this reduced it to  a little over 4000 images.



### Version 2 (Fine Tuning)
**Motivation**
Now we had a reduced dataset, but still it had a lot of irrelevant images. We wanted to refine our dataset more.

**Description**
We separated few images manually and built a very small dataset containing 164 training samples and 20 validation samples. We then used transfer learning approach using VGG 16 to train the model to classify between classes:
* Damaged
* Preserved

Following is the architecture diagram of the model

![enter image description here](https://ucarecdn.com/54598ba0-4c39-451c-bb20-8424ea214100/transfer_learning.png)


**Results**
This provided us a good starting point and set standards for base accuracy. The dataset was very small but provided us decent results. We also used this model to further refine our original dataset and move images to relevant and irrelevant folders.

### Version 3 (Damage Estimator Low/Moderate/High)
**Motivation**
Previous model gave us decent results. This time we wanted to improve our previous model and get a better dataset.

**Description**
We now manually built a dataset consisting of 420 training samples and 80 validation samples. 
First We evaluated 6 different models with slightly different architectures and hyper parameters on same dataset and same number of epochs. In all models, we used "bottleneck features" from the VGG16 model: the last activation maps before the fully-connected layers. There was only one neuron in the final layer and three classes to classify in, thus turning this into regression. So the model classified given images on the basis of model confidence (probability) as:

* Not Damaged/Irrelevant
* Moderately Damaged
* Highly Damaged

Following is the architecture of the model

![enter image description here](https://ucarecdn.com/d0ac312f-5e91-4363-9604-080dd4870d33/Bottleneck_features.png)

Following were the best results out of the 6 models
```
Validation Results (Loss, Acc):

[1.9399483680725098, 0.8375]
```
When evaluating this model on new data which was not used for training or validation, it showed 91% accuracy.

Then we created a new model by Fine Tuning VGG 16. We trained the model on same dataset. We used early stopping to ensure generalization in the model. Following is the architecture of Fine Tuned Model.

![enter image description here](https://ucarecdn.com/ee7b1ccb-043f-414a-907b-340841170ba2/Fine_Tuned.png)

This model showed following accuracy during training
```
Validation Results (Loss, Acc):

[0.3440, 0.88125]
```

When we evaluated model on the same unseen data as that for previous model, it showed 100% accuracy. But it was doing Binary Classification.

### Version 4 (NLP)
**Motivation**
We had a good progress in images but we were lacking in text classification. We decided to get a baseline approach and accuracy for text classification.

**Description**
We Tried three different approaches. All of them failed though, but provided us a good track to work on.

#### Extracting Quantity and Object-Subject Relation Using Grammar
We used the following grammar to extract information from the tweets.
```
Noun(Singlular|Plural)?Adverb?adjective|comparative,Preposition,quantity|Noun(Singlular|Plural)
```
Following is a example of output we got on sentence
```
>>> 'In Canada, earthquake effected more than 300 houses'
[[('more', 'JJR'), ('than', 'IN'), ('300', 'CD'), ('houses', 'NNS')]]
```
This approach was subjective to the grammar. We needed a more generic approach.

#### Looking for Frequent Nouns & Verbs
The next thing we tried was to find the top most frequent nouns and verbs in sentences which are as follows.
```
top_features_verbs = ['doctors','rescue','aid','killed','injured','toll','supplies','collapsed','affected','missing','damaged', 'hospital', 'rescue', 'emergency', 'team', 'government', 'death', 'army', 'military','family', 'families', 'quake','nepal', 'earthquake']
top_features_noun = ['hospital', 'rescue', 'emergency', 'team', 'government', 'death', 'army', 'military','family', 'families', 'quake','nepal']

```
On the basis of these nouns & verbs, classify the tweets as relevant or irrelevant.  This also did not worked well in classification.

#### Clustering using K-means
Next we thought clustering might work and separate relevant and irrelevant tweets. But following were the most frequent terms in two clusters formed and both clusters had many relevant and irrelevant terms.
```
Top terms per cluster:

Cluster 0: rescue aid india china everest team killed chinese injured toll search government helicopter capital teams nepalese buildings disaster tuesday death million camp international indian army avalanche april areas helicopters saturday collapsed including base aftershocks sunday missing survivors foreign military medical supplies rubble reported told ministry officials dead district according police

Cluster 1: team aid family nepalese rescue government disaster medical india victims home support april 2015 children efforts says just saturday israel minister safe world international food affected new red time emergency students need water supplies israeli cross community group assistance members days like local million response missing 25 humanitarian foreign families
```

### Version 5 (NN for Text)
**Motivation**
The previous approaches failed, so we went with ANN. At this moment we had humanly labelled data. Thus we had a good dataset which was not subjective.

**Description**
We trained a simple ANN for Text Classification. Following is the architecture of ANN used.

![enter image description here](https://ucarecdn.com/42cd6fb3-1844-4b37-82ed-4638804d4c8a/model_1.png)

It showed following results during training.
```
Validation Results (Loss, Acc):

[7.2613, 0.5445]
```

On a new data, it showed 53.97% Accuracy.


### Version 6 (CNN for Text)

**Motivation**
ANN clearly performed well. We had done much literature review till this point and the most of the people were using LSTM and CNN for text Classification. We went with CNN, as it is the silver bullet. 

**Description**
We developed and evaluated three different models. A simple CNN, CNN with Embedding, CNN with Pre-Trained Embedding (GLOVE). Following are the architectures. 

#### CNN with Embedding

![enter image description here](https://ucarecdn.com/8a5879a9-0686-401a-9572-2a33376ab2ff/model_4.png)

#### CNN with Pre-Trained Embedding (GLOVE)

![enter image description here](https://ucarecdn.com/06304a32-036c-4123-ba24-67c24846bae7/model_5.png)

#### Results

The results were very good as compared to simple ANN. All of these models were trained on same data. Following are the evaluation results of these models.  
```
3011/3011 [==============================] - 0s 58us/step
53.97% Accurate ANN
--------------------------------------------------
3011/3011 [==============================] - 0s 53us/step
93.92% Accurate CNN with Embedding
--------------------------------------------------
3011/3011 [==============================] - 0s 67us/step
94.75% Accurate CNN with Embedding and more Filters
--------------------------------------------------
3011/3011 [==============================] - 0s 64us/step
95.78% Accurate CNN with PRe Trained GLOVE Embedding
```



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

