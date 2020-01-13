# Common imports
import numpy as np
import os
import sklearn
import random
import nltk
import operator
import string
import requests
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
import re
from textblob import TextBlob as tb
import math


dev_dataset_positive = open('datasets_coursework1/IMDb/dev/imdb_dev_pos.txt').readlines()
dev_dataset_negative = open('datasets_coursework1/IMDb/dev/imdb_dev_neg.txt').readlines()

train_dataset_positive = open('datasets_coursework1/IMDb/train/imdb_train_pos.txt').readlines()
train_dataset_negative = open('datasets_coursework1/IMDb/train/imdb_train_neg.txt').readlines()


test_dataset_positive = open('datasets_coursework1/IMDb/test/imdb_test_pos.txt').readlines()
test_dataset_negative = open('datasets_coursework1/IMDb/test/imdb_test_neg.txt').readlines()


print('dev dataset positive : ' + str(len(dev_dataset_positive)))
print('dev dataset positive : ' + str(len(dev_dataset_negative)))
print('train dataset positive : ' + str(len(train_dataset_positive)))
print('train dataset negative : ' + str(len(train_dataset_negative)))
print('test dataset positive : ' + str(len(test_dataset_positive)))
print('test dataset negative : ' + str(len(test_dataset_negative)))

############################################################################################
#creating tuples for different data sets

dev_set = []
train_set = []
test_set = []

for review in dev_dataset_positive:
	dev_set.append((review.replace('\n','').translate(str.maketrans('','',string.punctuation)),1))
for review in dev_dataset_negative:
	dev_set.append((review.replace('\n','').translate(str.maketrans('','',string.punctuation)),0))

for review in train_dataset_positive:
	train_set.append((review.replace('\n','').translate(str.maketrans('','',string.punctuation)),1))
for review in train_dataset_negative:
	train_set.append((review.replace('\n','').translate(str.maketrans('','',string.punctuation)),0))

for review in test_dataset_positive:
	test_set.append((review.replace('\n','').translate(str.maketrans('','',string.punctuation)),1))
for review in test_dataset_negative:
	test_set.append((review.replace('\n','').translate(str.maketrans('','',string.punctuation)),0))

random.shuffle(dev_set)
random.shuffle(train_set)
random.shuffle(test_set)

def train_svm_classifier(training_set, vocabulary): # Function for training our svm classifier
  X_train=[]
  Y_train=[]
  for instance in training_set:
    vector_instance=get_vector_text(vocabulary,instance[0])
    X_train.append(vector_instance)
    Y_train.append(instance[1])
  # Finally, we train the SVM classifier 
  svm_clf=sklearn.svm.SVC(kernel="linear",gamma='auto')
  svm_clf.fit(np.asarray(X_train),np.asarray(Y_train))
  return svm_clf

def tf(word, blob):
    termFrequency= blob.words.count(word) / len(blob.words)
    return termFrequency

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)

# Function taken from Session 1
def remove_punctuation(string):
	string = string.split("<br />")
	string = " ".join(string)
	string = re.sub('[^0-9a-zA-Z\s]+', '', string)
	return string

def get_list_tokens(string):
	string = remove_punctuation(string)
	sentence_split=nltk.tokenize.sent_tokenize(string)
	list_tokens=[]
	for sentence in sentence_split:
		list_tokens_sentence=nltk.tokenize.word_tokenize(sentence)
	for token in list_tokens_sentence:
		list_tokens.append(lemmatizer.lemmatize(token).lower())
	return list_tokens

# def get_vocabulary(training_set, num_features): # Function to retrieve vocabulary
	# dict_word_frequency={}
	# tb_training_set = []
	# for training_review in training_set:
	# 	tb_training_set.append(tb(training_review[0]))

	# sorted_words = []

	# for i, blob in enumerate(tb_training_set):
	# 	scores = {word: tfidf(word, blob, tb_training_set) for word in blob.words if word not in stopwords}
	# 	sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:num_features]
	# vocabulary = []
	# for word,frequency in sorted_list:
	# 	vocabulary.append(word)
	# return vocabulary

def get_vocabulary(training_set, num_features): # Function to retrieve vocabulary
  dict_word_frequency={}
  for instance in training_set:
    sentence_tokens=get_list_tokens(instance[0])
    for word in sentence_tokens:
      if word in stopwords: continue
      if word not in dict_word_frequency: dict_word_frequency[word]=1
      else: dict_word_frequency[word]+=1
  sorted_list = sorted(dict_word_frequency.items(), key=operator.itemgetter(1), reverse=True)[:num_features]
  vocabulary=[]
  for word,frequency in sorted_list:
    vocabulary.append(word)
  return vocabulary


def get_vector_text(list_vocab,string):
  vector_text=np.zeros(len(list_vocab))
  list_tokens_string=get_list_tokens(string)
  for i, word in enumerate(list_vocab):
    if word in list_tokens_string:
      vector_text[i]=list_tokens_string.count(word)
  return vector_text



lemmatizer = nltk.stem.WordNetLemmatizer()
stopwords=set(nltk.corpus.stopwords.words('english'))

list_num_features=[500,750,1000,1250]
best_accuracy_dev=0.0

vocabulary = get_vocabulary(train_set,1250)
X_train_vector_set = []
Y_train = []
Y_test = []
Y_dev = []
for reviews in train_set:
	X_train_vector_set.append(get_vector_text(vocabulary,reviews[0]))
	Y_train.append(reviews[1])
vocabulary = get_vocabulary(train_set,1250)


for reviews in dev_set:
	Y_dev.append(reviews[1])

# Y_dev_gold=np.asarray(Y_dev)
Y_dev_gold=Y_dev
best_vocabulary = []

for num_features in list_num_features:
	vocabulary = get_vocabulary(dev_set,num_features)
	print('num_features')
	print(num_features)
	X_dev_vector_set = []
	svm_clf=train_svm_classifier(dev_set, vocabulary)
	for reviews in dev_set:
		X_dev_vector_set.append(get_vector_text(vocabulary,reviews[0]))
	X_dev_vector_set=np.asarray(X_dev_vector_set)
	Y_dev_predictions=svm_clf.predict(X_dev_vector_set)
	accuracy_dev=accuracy_score(Y_dev_gold, Y_dev_predictions)
	print ("Using "+str(num_features)+" features : "+str(round(accuracy_dev,3)))
	if accuracy_dev>=best_accuracy_dev:
		best_accuracy_dev=accuracy_dev
		best_num_features=num_features
		best_vocabulary=vocabulary
		best_svm_clf=svm_clf
	print ("\n Best accuracy is with "+str(best_num_features)+" features.")

X_test_vector_set = []
for reviews in test_set:
	X_test_vector_set.append(get_vector_text(best_vocabulary,reviews[0]))
	Y_test.append(reviews[1])

print(len(X_train_vector_set))

X_train_vector_set = np.asarray(X_train_vector_set)
svm_clf_sentanalysis=sklearn.svm.SVC(kernel='linear')
svm_clf_sentanalysis.fit(X_train_vector_set,Y_train) # Train the SVM model. This may also take a while.

Y_prediction = svm_clf_sentanalysis.predict(X_test_vector_set)

from sklearn.metrics import classification_report, confusion_matrix
# print(classification_report(Y_test,Y_prediction))

precision=precision_score(Y_test, Y_prediction, average='macro')
recall=recall_score(Y_test, Y_prediction, average='macro')
f1=f1_score(Y_test, Y_prediction, average='macro')
accuracy=accuracy_score(Y_prediction,Y_test)

print ("Precision: "+str(round(precision,3)))
print ("Recall: "+str(round(recall,3)))
print ("F1-Score: "+str(round(f1,3)))
print ("Accuracy: "+str(round(accuracy,3)))

print(np.mean(Y_prediction == Y_test))