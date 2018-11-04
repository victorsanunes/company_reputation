
# coding: utf-8

# In[1]:


import sklearn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
import numpy

get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
plt.style.use('ggplot')


# ## IMPORT FILES

# In[2]:

tweetsDF = pd.read_csv("Tweets.csv")

with open("lexicons/positive-words.txt") as file:
    positiveList = set(file.read().splitlines())
    
with open("lexicons/negative-words.txt") as file:
    negativeList = set(file.read().splitlines())

tweetsDF.drop(labels=['tweet_id',
                      'name',
                      'retweet_count',
                      'tweet_coord',
                      'tweet_created',
                      'tweet_location', 
                      'user_timezone'], axis=1, inplace=True)


tweetsDF.drop(labels=['airline_sentiment_gold',
                      'negativereason_gold'], axis=1,inplace=True)


# Missing values
from sklearn.preprocessing import Imputer
imputer = Imputer()

# transformed = imputer.fit_transform(tweetsDF["negativereason_confidence"])
negativeMean = tweetsDF["negativereason_confidence"].mean()
tweetsDF["negativereason_confidence"].fillna(negativeMean, inplace=True)
tweetsDF["negativereason"].fillna("Can't Tell", inplace=True)


# In[9]:


tweetsDF.info()


# In[10]:


## Distribuição de sentimentos
sentimentCounter = tweetsDF['airline_sentiment'].value_counts()
sentimentCounter


# In[11]:


Index = [1,2,3]
plt.bar(Index,sentimentCounter)
plt.xticks(Index,['negative','neutral','positive'],rotation=45)
plt.ylabel('Mood Count')
plt.xlabel('Mood')
plt.title('Count of Moods')


# ## Distribuição de tweets por companhia

# In[12]:


tweetsDF['airline'].value_counts().plot(kind='bar')


# In[13]:


def plot_sub_sentiment(airline, tweet_df):
    df = tweet_df[tweet_df['airline']==airline]
    count = df['airline_sentiment'].value_counts()
    index = [1,2,3]
    plt.bar(index,count)
    plt.xticks(index,['negative','neutral','positive'])
    plt.ylabel('Frequencia')
    plt.xlabel('Sentimento')
    plt.title('Dist de sentimentos'+airline)
#     plt.ylim(0, len(tweet_df))
    plt.ylim(0, 3000)


# In[14]:


plt.figure(1,figsize=(12, 12))
plt.subplot(231)
plot_sub_sentiment('US Airways', tweetsDF)

plt.subplot(232)
plot_sub_sentiment('United', tweetsDF)

plt.subplot(233)
plot_sub_sentiment('American', tweetsDF)

plt.subplot(234)
plot_sub_sentiment('Southwest', tweetsDF)

plt.subplot(235)
plot_sub_sentiment('Delta', tweetsDF)

plt.subplot(236)
plot_sub_sentiment('Virgin America', tweetsDF)


# ---
# ## Motivos de reclamação
# Vamos analisar a distribuição dos motivos de reclamacao

# In[15]:


nrCounter = dict(tweetsDF['negativereason'].value_counts(sort=False))


# In[16]:


def NR_Count(Airline, tweet_df):
    if Airline == 'All':
        df = tweet_df
    else:
        df = tweet_df[tweet_df['airline'] == Airline]
    
    count = dict(df['negativereason'].value_counts())
    Unique_reason = list(tweet_df['negativereason'].unique())
    Unique_reason = [x for x in Unique_reason if str(x) != 'nan']
    Reason_frame = pd.DataFrame({'Reasons':Unique_reason})
    Reason_frame['count'] = Reason_frame['Reasons'].apply(lambda x: count[x])
    return Reason_frame

def plot_reason(Airline, tweet):
    df = NR_Count(Airline, tweet)
    count = df['count']
    Index = range(1,(len(df)+1))
    plt.bar(Index,count)
    plt.xticks(Index,df['Reasons'],rotation=90)
    plt.ylabel('Count')
    plt.xlabel('Reason')
    plt.title('Count of Reasons for '+Airline)


# In[17]:


plot_reason('All', tweetsDF)


# Reclamacoes sobre o servico ao cliente sao as mais frequentes, seguidas de atraso de voo. É interessante notar que mais de 1000 reclamacoes estao em categorias diversas.

# ## Motivos de reclamacoes por companhia
# Vamos analisar as reclamacoes por cada companhia

# In[18]:


companies = list(tweetsDF["airline"].unique())
subPlot = 231
plt.figure(1,figsize=(12, 12))
for c in companies:
    plt.subplot(subPlot)
    plot_reason(c, tweetsDF)
    subPlot += 1


# Em todas as companhias, a maior reclamacao é a do serviço ao cliente. Agora vamos plotar um wordcloud para ver quais as palavras mais frequentemente usadas. 

# In[19]:


from wordcloud import WordCloud, STOPWORDS


# In[20]:


df = tweetsDF[tweetsDF["airline_sentiment"]=='negative']
words = ' '.join(df["text"])
cleanedWord = " ".join([w for w in words.split()
                            if "http" not in w
                            and not w.startswith("@")
                            and w != "RT"
                       ])


# In[21]:


wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color='black',
                      width=3000,
                      height=2500
                     ).generate(cleanedWord)


# In[22]:


plt.figure(1,figsize=(12, 12))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# Como as reclamacoes mais frequentes estao relacionadas ao servico ao consumidor e as palavras mais frequentes são "plane", "bag", "flight", podemos propor que a qualidade das aeronaves nao satisfaz os consumidores. Podem haver problemas com as bagagens de mao (espaço insuficiente, ou incovenientes). Além disso, a oferta de voos tambem pode ser um motivo de reclamacao

# ## Extração de características
# Características basedas [neste](https://www.microsoft.com/en-us/research/wp-content/uploads/2015/03/main.pdf) trabalho serão utilizadas. Vale ressaltar que o autor extraiu 34 características. Nem todas elas serão extraídas aqui.

# In[23]:


# feature 3
def countPositiveCapitalized(tokens):
    """
    Calculates the number of positive words that are capitalized
    
    @params:
        tokens: The non stopwords list
    """
    counter = 0
    tokensSplit = tokens.split()
    for t in tokensSplit:
        if t[0].isupper() == True and t in positiveList:
            counter += 1
    return counter

# feature 4
def countNegativeCapitalized(tokens):
    """
    Calculates the number of negative words that are capitalized
    
    @params:
        tokens: The non stopwords list
    """
    counter = 0
    tokensSplit = tokens.split()
    for t in tokensSplit:
        if t[0].isupper() == True and t in negativeList:
            counter += 1
    return counter

# feature 5
def hasCapitalized(tokens):
    """
    Check if the tweet has capitalized words
    
    @params:
        tokens: The non stopwords list
    """
    counter = 0
    tokensSplit = tokens.split()
    for t in tokensSplit:
        if t[0].isupper() == True:
            return 1
    return 0

# feature 6
def countHashtags(tokens):
    """
    Count the number of words that starts with # (hashtags)
    
    @params:
        tokens: The non stopwords list
    """
    counter = 0
    for t in tokens:
        if t.startswith("#"):
            counter += 1
    return counter

# feature 7
def countPositive(tokens):
    """
    Calculates the number of words that are in the positive words list
    
    @params:
        tokens: The non stopwords list
    """
    counter = 0
    tokensSplit = tokens.split()
    for t in tokensSplit:
        if t.lower() in positiveList:
            counter += 1
    return counter

# feature 8
def countNegative(tokens):
    """
    Calculates the number of words that are in the negative words list
    
    @params:
        tokens: The non stopwords list
    """
    counter = 0
    tokensSplit = tokens.split()
    for t in tokensSplit:
        if t.lower() in negativeList:
            counter += 1
    return counter

# feature 9
def countNeutral(tokens):
    """
    Calculates the number of words that are in the neutral words list
    
    @params:
        tokens: The non stopwords list
    """
    counter = 0
    tokensSplit = tokens.split()
    for t in tokensSplit:
        if t.lower() not in negativeList and t.lower() not in positiveList:
            counter += 1
    return counter

# feature 10
def countCapitalizedWords(tokens):
    """
    Calculates the number of words that are capitalized
    
    @params:
        tokens: The non stopwords list
    """
    counter = 0
    tokensSplit = tokens.split()
    for t in tokensSplit:
        if t.isupper() and len(t) > 1:
            counter += 1
    return counter

# feature 11
def countSpecialCharacters(tokens):
    """
    Calculates the number of occurrencies of all special character
    
    @params:
        tokens: The non stopwords list
    """
    counter = 0
    tokensSplit = tokens.split()
    for t in tokensSplit:
        if not re.match("^[a-zA-Z0-9_]*$", t):
            counter += 1
    return counter

def countSpecificSpecialCharacter(specialCharacter, tokens):
    """
    Calculates the number of occurrencies of a specific special character
    
    @params:
        tokens: The non stopwords list
    """
    counter = 0
    tokensSplit = tokens.split()
    for t in tokensSplit:
        if t == specialCharacter:
            counter += 1
    return counter


# ## Preprocessamento
# 
# Agora iremos preparar os tweets para servirem de input aos classificadores

# In[24]:


import re
import nltk
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score


# In[25]:


tweetsDF.head()


# In[26]:


def tweet2words(raw_tweet):
    """
    Split the tweet string into words list and remove stopwords
        
    @params:
        raw_tweet: the tweet string collectd
    """
    callout_regex = "@[A-Za-z0-9_]+"
    #Remove mencoes a perfis
    letters_only = re.sub(callout_regex, " ", raw_tweet)
    letters_only = re.sub("[^a-zA-Z]", " ", letters_only)
    
    words = letters_only.lower().split()                             
    words = letters_only.split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops] 
    return( " ".join( meaningful_words )) 


def clean_tweet_length(raw_tweet):
    """
    Calculates the number of non stopwords
    
    @params:
        raw_tweet: the tweet string collectd
    """
    letters_only = re.sub("[^a-zA-Z]", " ",raw_tweet) 
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops] 
    return(len(meaningful_words)) 


# In[27]:


tweetsDF['sentiment'] = tweetsDF['airline_sentiment'].apply(lambda x: 0 if x=="negative" else 1)

tweetsDF['tweet2words'] = tweetsDF['text'].apply(tweet2words)

tweetsDF["num_capitalized"] = tweetsDF["tweet2words"].apply(countCapitalizedWords)

tweetsDF['tweet_length'] = tweetsDF['text'].apply(clean_tweet_length)

tweetsDF["num_negative_words"] = tweetsDF["tweet2words"].apply(countNegative)


# In[28]:


# Number of occurrencies
tweetsDF["num_positive_words"] = tweetsDF['text'].apply(countPositive)
tweetsDF["num_negative_words"] = tweetsDF['text'].apply(countNegative)
tweetsDF["num_neutral_words"] = tweetsDF['text'].apply(countNeutral)

# Capitalized words
tweetsDF["has_capitalized"] = tweetsDF['text'].apply(hasCapitalized)
tweetsDF["num_capitalised_positive_words"] = tweetsDF['text'].apply(countPositiveCapitalized)
tweetsDF["num_capitalised_negative_words"] = tweetsDF['text'].apply(countNegativeCapitalized)


tweetsDF["num_hashtags"] = tweetsDF['text'].apply(countHashtags)
tweetsDF["num_special_character"] = tweetsDF['text'].apply(countSpecialCharacters)


# In[29]:


tweetsDF.info()


# In[38]:


tweetsDF.to_csv("preprocessed.csv",sep=',',header=True)


# In[31]:


train,test = train_test_split(tweetsDF,test_size=0.2,random_state=42)


# In[32]:


train_clean_tweet=[]
for tweet in train['tweet2words']:
    train_clean_tweet.append(tweet)
    
test_clean_tweet=[]
for tweet in test['tweet2words']:
    test_clean_tweet.append(tweet)


# In[33]:


from sklearn.feature_extraction.text import CountVectorizer

v = CountVectorizer(analyzer = "word")
train_features = v.fit_transform(train_clean_tweet)
test_features = v.transform(test_clean_tweet)


# In[34]:


len(train_clean_tweet)


# In[35]:


Classifiers = [
    LogisticRegression(C=0.000000001,solver='liblinear',max_iter=200),
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=200),
    AdaBoostClassifier(),
    GaussianNB(),
    MLPClassifier(),
]


# In[36]:


dense_features = train_features.toarray()
dense_test = test_features.toarray()

Accuracy=[]
Model=[]

for classifier in Classifiers:
    try:
        fit = classifier.fit(train_features,train['sentiment'])
        pred = fit.predict(test_features)
    except Exception:
        fit = classifier.fit(dense_features,train['sentiment'])
        pred = fit.predict(dense_test)
    accuracy = accuracy_score(pred,test['sentiment'])
    Accuracy.append(accuracy)
    Model.append(classifier.__class__.__name__)
    print('Accuracy of '+classifier.__class__.__name__+'is '+str(accuracy))


# In[37]:


Index = [i for i in range(1, len(Model)+1)]
plt.bar(Index,Accuracy)
plt.xticks(Index, Model,rotation=45)
plt.ylabel('Acuracia')
plt.xlabel('Modelo')
plt.title('Acuracia dos modelos')

