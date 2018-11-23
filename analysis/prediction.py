
# coding: utf-8

SAVE_MODEL_TO_DISK = 0
LOAD_MODEL = 1

# GENERAL LIBS
import numpy as np 
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# SKLEARN
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.pipeline import FeatureUnion
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB

#NLTK
from nltk.stem.snowball import SnowballStemmer

# GENSIM
from gensim.sklearn_api import W2VTransformer
from gensim.models import Word2Vec

import warnings
warnings.filterwarnings('ignore')


# In[2]:


tweetsDF = pd.read_csv("preprocessed2.csv")

tweetsDF.drop(labels=["Unnamed: 0",
                      "airline", 
                      "negativereason", 
                      "airline_sentiment_confidence", 
                      "negativereason",
                      "negativereason_confidence",
                      "airline_sentiment",
                      "text"], axis=1, inplace=True)

def stemming(tokens):
    '''
    Apply stemming to each token
    
    @return:
        Return a list of stemmed tokens
    '''
    
    stemmer = SnowballStemmer("english")  
    stemmed = [stemmer.stem(w) for w in tokens.split()]
    return stemmed

tweetsDF["tweet2words"] = tweetsDF["tweet2words"].values.astype("U")
tweetsDF["correctedText"] = tweetsDF["correctedText"].values.astype("U")
tweetsDF.drop(labels=['tweet2words'], axis=1, inplace=True)

target = "sentiment"
features = [c for c in tweetsDF.columns.values if c not in [target]]
numeric_features =  [c for c in tweetsDF.columns.values if c not in ['tweet2words', 'correctedText', target]]

X_train = tweetsDF[features]
Y_train = tweetsDF[target]


# ---
# A criação dos pipelines foi feita utilizando os seguintes kernels no kaggle como auxiliadores;
# 1. [Building A Scikit Learn Classification Pipeline](https://www.kaggle.com/gautham11/building-a-scikit-learn-classification-pipeline)
# 2. [A Deep Dive Into Sklearn Pipelines](https://www.kaggle.com/baghern/a-deep-dive-into-sklearn-pipelines)
# ---

class TextSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on text columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def splitString(self, s):
        try:
            return s.split()
        except AttributeError:
            return ""
            
    
    def transform(self, X):
        # Apply the word2vec transformation
        a = X[self.key]
#         return wordvecs.fit_transform(a)
        return a
 
class Senteces(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()

class NumberSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]

text = Pipeline([
                ('selector', TextSelector(key='correctedText')),
                ('countVec', CountVectorizer(analyzer = "word"))
#                 ('w2v', Word2VecTransformation)
#                 ('w2v', Word2Vec())
            ])

numCapitalized = Pipeline([
                    ('selector', NumberSelector(key='num_capitalized')),
#                     ('standard', StandardScaler())
                    ('standard', MinMaxScaler())
                ])

tweetLength = Pipeline([
                    ('selector', NumberSelector(key='tweet_length')),
#                     ('standard', StandardScaler())
                    ('standard', MinMaxScaler())
                ])

numNegativeWords = Pipeline([
                    ('selector', NumberSelector(key='num_negative_words')),
                    #('standard', StandardScaler())
                    ('standard', MinMaxScaler())
                ])

numPositiveWords = Pipeline([
                    ('selector', NumberSelector(key='num_positive_words')),
#                     ('standard', StandardScaler())
                    ('standard', MinMaxScaler())
                ])

numNeutralWords = Pipeline([
                    ('selector', NumberSelector(key='num_neutral_words')),
#                     ('standard', StandardScaler())
                    ('standard', MinMaxScaler())
                ])


numCapitalizedPositiveWords = Pipeline([
                    ('selector', NumberSelector(key='num_capitalised_positive_words')),
#                     ('standard', StandardScaler())
                    ('standard', MinMaxScaler())
                ])

numCapitalizedNegativeWords = Pipeline([
                    ('selector', NumberSelector(key='num_capitalised_negative_words')),
#                     ('standard', StandardScaler())
                    ('standard', MinMaxScaler())
                ])

numHashtags = Pipeline([
                    ('selector', NumberSelector(key='num_hashtags')),
#                     ('standard', StandardScaler())
                    ('standard', MinMaxScaler())
                ])


numSpecialCharacter = Pipeline([
                    ('selector', NumberSelector(key='num_special_character')),
#                     ('standard', StandardScaler())
                    ('standard', MinMaxScaler())
                ])

feats = FeatureUnion([('text', text),
                      ('numCapitalized', numCapitalized),
                      ("tweetLength", tweetLength),
                      ("numNegativeWords", numNegativeWords),
                      ("numPositiveWords", numPositiveWords),
                      ("numNeutralWords", numNeutralWords),
                      ("numCapitalizedPositiveWords", numCapitalizedPositiveWords),
                      ("numCapitalizedNegativeWords", numCapitalizedNegativeWords),
                      ("numHashtags", numHashtags),
                      ("numSpecialCharacter", numSpecialCharacter)
                     ])

feature_processing = Pipeline([('feats', feats)])
feature_processing.fit_transform(X_train)

from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('features',feats),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state = 42)),
])

pipeline.fit(X_train, Y_train)

if LOAD_MODEL == 0:
    clfs = list()
    # clfs.append(LogisticRegression())
    # clfs.append(SVC())
    # clfs.append(DecisionTreeClassifier())
    # clfs.append(RandomForestClassifier(n_estimators=200, random_state = 42))
    # clfs.append(GradientBoostingClassifier())
    # clfs.append(MLPClassifier())
    clfs.append(MultinomialNB())

    scores = list()
    pipelineList = list()
    scores2 = list()

    for c in clfs:
        pipeline.set_params(classifier = c)
        pipeline.fit(X_train, Y_train)
        s = cross_validate(pipeline, X_train, Y_train, 
                           scoring=["accuracy", "recall", "precision", "f1"], 
                           cv=3, return_estimator = True)
        scores.append(s)
        pipelineList.append(pipeline)
        # scores2.append(pipeline.score(X_train, Y_train))
        print('---------------------------------')
        print(str(c))
        print('-----------------------------------')
        # for key, values in s.items():
        # print(key,' mean ', values.mean())
        # print(key,' std ', values.std())


    bestModels = list()
    for model in scores:
        maxAcc = max(model['test_accuracy'])
        bestModelX = np.where(model['test_accuracy'] == maxAcc)
        bestModels.append(model['estimator'][0])

import pickle
from os import listdir
from os.path import isfile, join

modelsPath = 'models/'
onlyfiles = [f for f in listdir(modelsPath) if isfile(join(modelsPath, f))]

from sklearn.externals import joblib

if SAVE_MODEL_TO_DISK == 1:
    for i in range(len(bestModels)):
        model = bestModels[i].get_params('classifier')
        file = str(model['classifier']).split('(')[0]
        joblib.dump(bestModels[i], file + ".joblib") 

elif LOAD_MODEL == 1:
    print("Loading models...")
    bestModels = list()
    for file in onlyfiles:
        bestModels.append(joblib.load(modelsPath + file))


from os import listdir
from os.path import isfile, join

tweetsPath = '03_processed/'
onlyfiles = [f for f in listdir(tweetsPath) if isfile(join(tweetsPath, f))]

for model in bestModels: 
    classifierName = str(model.get_params('classifier')['classifier']).split('(')[0]
    print("({}) Predicting...".format(classifierName))
    for file in onlyfiles:
        newTweets = pd.read_csv(tweetsPath + file)
        newTweets.drop(labels=['text', 'tweet2words'], axis = 1, inplace=True)
        newTweets['correctedText'] = newTweets["correctedText"].values.astype("U")
        newTweets['sentiment'] = model.predict(newTweets)
        
        newTweets.to_csv('teste/' + classifierName + '/' + file.split('.')[0] + '_prediction.csv')

from sklearn.metrics import confusion_matrix

allMeasures = dict()
print("Calculating measures...")
for i in range(len(bestModels)):
    model = bestModels[i]
    tn, fp, fn, tp = confusion_matrix(model.predict(X_train), Y_train).ravel()

    measures = dict()
    measures["acc"] = (tp + tn)/(tn + fp + fn + tp) * 100
    prec = tp/(tp + fp) * 100
    recall = tp/(tp + fn) * 100
    measures["prec"] = prec
    measures["recall_sens"] = recall
    measures["f1_score"] = (2 * prec * recall/(prec + recall))
    measures['miss_rate'] = (fp + fn) / float(tp + tn + fp + fn) * 100
    measures['spec'] = tn/float(tn + fp) * 100
    measures['fp_rate'] = fp/float(tn + fp) * 100
    allMeasures[i] = measures

# allMeasures
performance = pd.DataFrame.from_dict(data=allMeasures, orient='index')
performance.set_axis(labels=['SVC', 'DecisionTreeClassifier', 'MLPClassifier', 'MultinomialNB'], inplace=True)
# performance.set_axis(labels=['NB'], inplace=True)
print(performance)

