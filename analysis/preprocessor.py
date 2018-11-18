import numpy as np 
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import hunspell
spellchecker = hunspell.HunSpell('/usr/share/hunspell/en_US.dic',
                            '/usr/share/hunspell/en_US.aff')

from os import listdir
from os.path import isfile, join

with open("lexicons/positive-words.txt") as file:
    positiveList = set(file.read().splitlines())
    
with open("lexicons/negative-words.txt") as file:
    negativeList = set(file.read().splitlines())

class PreProcessing:
    def __init__(self, df, filepath):
        self.filepath = filepath
        self.df = df
        self.df.drop(labels=['username',
                      'user_handle',
                      'date',
                      'retweets',
                      'favorites',
                      'geological_location',
                      'mentions',
                      'hashtags',
                      'tweet_id',
                      'permalink',
                      'col1', 'col2', 'col3'], axis=1, inplace=True)

    # feature 3
    def countPositiveCapitalized(self, tokens):
        """
        Calculates the number of positive words that are capitalized
        
        @params:
            tokens: The non stopwords list
        """
        ######print('countPositiveCapitalized()')
        counter = 0
        tokensSplit = tokens.split()
        for t in tokensSplit:
            if t[0].isupper() == True and t in positiveList:
                counter += 1
        return counter

    # feature 4
    def countNegativeCapitalized(self, tokens):
        """
        Calculates the number of negative words that are capitalized
        
        @params:
            tokens: The non stopwords list
        """
        #####print('countNegativeCapitalized()')
        counter = 0
        tokensSplit = tokens.split()
        for t in tokensSplit:
            if t[0].isupper() == True and t in negativeList:
                counter += 1
        return counter

    # feature 5
    def hasCapitalized(self, tokens):
        """
        Check if the tweet has capitalized words
        
        @params:
            tokens: The non stopwords list
        """
        #####print('hasCapitalized()')
        counter = 0
        tokensSplit = tokens.split()
        for t in tokensSplit:
            if t[0].isupper() == True:
                return 1
        return 0

    # feature 6
    def countHashtags(self, tokens):
        """
        Count the number of words that starts with # (hashtags)
        
        @params:
            tokens: The non stopwords list
        """

        #####print('countHashtags()')
        counter = 0
        for t in tokens:
            if t.startswith("#"):
                counter += 1
        return counter

    # feature 7
    def countPositive(self, tokens):
        """
        Calculates the number of words that are in the positive words list
        
        @params:
            tokens: The non stopwords list
        """

        #####print('countPositive()')
        counter = 0
        tokensSplit = tokens.split()
        for t in tokensSplit:
            if t.lower() in positiveList:
                counter += 1
        return counter

    # feature 8
    def countNegative(self, tokens):
        """
        Calculates the number of words that are in the negative words list
        
        @params:
            tokens: The non stopwords list
        """

        #####print('countNegative()')
        counter = 0
        tokensSplit = tokens.split()
        for t in tokensSplit:
            if t.lower() in negativeList:
                counter += 1
        return counter

    # feature 9
    def countNeutral(self, tokens):
        """
        Calculates the number of words that are in the neutral words list
        
        @params:
            tokens: The non stopwords list
        """

        #####print('countNeutral()')
        counter = 0
        tokensSplit = tokens.split()
        for t in tokensSplit:
            if t.lower() not in negativeList and t.lower() not in positiveList:
                counter += 1
        return counter

    # feature 10
    def countCapitalizedWords(self, tokens):
        """
        Calculates the number of words that are capitalized
        
        @params:
            tokens: The non stopwords list
        """

        #####print('countCapitalizedWords()')
        counter = 0
        tokensSplit = tokens.split()
        for t in tokensSplit:
            if t.isupper() and len(t) > 1:
                counter += 1
        return counter

    # feature 11
    def countSpecialCharacters(self, tokens):
        """
        Calculates the number of occurrencies of all special character
        
        @params:
            tokens: The non stopwords list
        """

        #####print('countSpecialCharacters()')
        counter = 0
        tokensSplit = tokens.split()
        for t in tokensSplit:
            if not re.match("^[a-zA-Z0-9_]*$", t):
                counter += 1
        return counter

    def countSpecificSpecialCharacter(self, specialCharacter, tokens):
        """
        Calculates the number of occurrencies of a specific special character
        
        @params:
            tokens: The non stopwords list
        """

        #####print('countSpecificSpecialCharacter()')
        counter = 0
        tokensSplit = tokens.split()
        for t in tokensSplit:
            if t == specialCharacter:
                counter += 1
        return counter

    def fixSpelling(self, tokens):
        #####print('fixSpelling()')

                            
        words = tokens.split()              
        newWords = ""#list()
        for w in words:
            if not spellchecker.spell(w):
    #             newWords.append(spellchecker.suggest(w)[0])
                try:
                    newWords += " " + spellchecker.suggest(w)[0]
                except(IndexError):
                    newWords += " " + ""
            else:
    #             newWords.append(w)
                newWords += " " + w
        return newWords
            
    def stemming(self, tokens):
        '''
        Apply stemming to each token
        '''
        
        stemmer = SnowballStemmer("english")  
        stemmed = [stemmer.stem(w) for w in tokens.split()]
        return stemmed

    def tweet2words(self, raw_tweet):
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

    def clean_tweet_length(self, raw_tweet):
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

    def preprocessing(self):
        self.df["text"] = self.df["text"].values.astype("U")
        
        self.df['tweet2words'] = self.df['text'].apply(self.tweet2words)

        self.df["num_capitalized"] = self.df["tweet2words"].apply(self.countCapitalizedWords)

        self.df['tweet_length'] = self.df['text'].apply(self.clean_tweet_length)

        self.df["num_negative_words"] = self.df["tweet2words"].apply(self.countNegative)
        # Number of occurrencies
        self.df["num_positive_words"] = self.df['text'].apply(self.countPositive)
        self.df["num_negative_words"] = self.df['text'].apply(self.countNegative)
        self.df["num_neutral_words"] = self.df['text'].apply(self.countNeutral)

        # Capitalized words
        self.df["has_capitalized"] = self.df['text'].apply(self.hasCapitalized)
        self.df["num_capitalised_positive_words"] = self.df['text'].apply(self.countPositiveCapitalized)
        self.df["num_capitalised_negative_words"] = self.df['text'].apply(self.countNegativeCapitalized)


        self.df["num_hashtags"] = self.df['text'].apply(self.countHashtags)
        self.df["num_special_character"] = self.df['text'].apply(self.countSpecialCharacters)
        self.df['correctedText'] =  self.df['tweet2words'].apply(self.fixSpelling)

    def exportDataframe(self):
        self.df.to_csv(self.filepath + '_preprocessed.csv', index=False)

if __name__== '__main__':
    tweetsPath = 'prediction_tweets'
    onlyfiles = [f for f in listdir(tweetsPath) if isfile(join(tweetsPath, f))]

    for path in onlyfiles:
        print("Processing " + path)
        # print('prediction_tweets/' + path)
        df = pd.read_csv('prediction_tweets/' + path, error_bad_lines=False)
        preproc = PreProcessing(df, 'output/' + path.split('.')[0])
        preproc.preprocessing()
        preproc.exportDataframe()
