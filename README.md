# Medical Text Classification

### Description
This program is a basic predictive model developed to determine given a medical abstract, which of 5 classes it falls in.
Medical abstracts describe the current conditions of a patient. Doctors routinely scan dozens or hundreds of abstracts each day as they do their rounds in a hospital and must quickly pick up on the salient information pointing to the patient’s malady. Here, we are trying to design assistive technology that can identify, with high precision, the class of problems
described in the abstract. In the given dataset, abstracts from 5 different conditions have been included: digestive system diseases, cardiovascular diseases, neoplasms, nervous system diseases, and general pathological conditions.

### My Implementation
This predictive model uses min-epsilon kNN classifier. The min-epsilon kNN classifier is defined similarly as the k-NN classifier with the exception that neighbors 2 to k are additionally restricted to have a minimum similarity of epsilon with
the query object. In other words, restrict neighbors by both number of neighbors and minimum similarity, but always retrieve at least one neighbor.

### Data Description
The training dataset consists of 14438 records and the test dataset consists of 14442 records. The train data has classes whereas, the test data classes are needed to be predicted.
The data are provided as text in train.dat and test.dat, which should be processed appropriately.

### Requirements
You need to have pandas, numpy and nltk already installed to run this program or else you can do by using pip.
```
pip install pandas
pip install numpy
pip install nltk
```

