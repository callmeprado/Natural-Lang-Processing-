# Natural Language Processing 

#importing Libraries 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

#importing Data set
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3 )

#Cleaning the text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []

for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    """splitting the string so that non valuables can be stemmmed"""
    review = review.split() 
    
    ps = PorterStemmer()
    """ the for loop goes to every element of list and removes non essential words eg. this/that/the """
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english')) ] 
    """we join the words back into one string"""
    review = ' '.join(review)
    corpus.append(review)

"""From the list of essentials created, we will create a vector for each observation.
The vector will show 1 if it's an essential word, 0 if not. Next step would be to classify USING these 
essentials whether the result(review) of each obs was postive(1) or not(0). """ 

"""Sparsing: Observing the presence of these words to classify our dependent variabe vector."""
   
#Creating Bag of Words model 
from sklearn.feature_extraction.text import CountVectorizer 
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
Y = dataset.iloc[:,1].values

#Splitting into Train & Test 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size= 0.20, random_state = 0 )

#Fitting NB to training set 
from sklearn.naive_bayes import GaussianNB  
classifier = GaussianNB()
classifier.fit(x_train, y_train)

#Predicting Test set results 
y_pred = classifier.predict(x_test)

#Making Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Model Accuracy 
Accuracy = ( cm[0][0] + cm[1][1] )/len(y_test)
