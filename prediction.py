class TextFeatureExtractor():
    def __init__(self):
        pass
    
    # feature 7
    def countPositive(self, tokens):
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
    
    def transform(self, df, y=None):
        return df[""]


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC

# build the feature matrices
ngram_counter = CountVectorizer(ngram_range=(1, 4), analyzer='char')
X_train = ngram_counter.fit_transform(data_train)
X_test  = ngram_counter.transform(data_test)

# train the classifier
classifier = LinearSVC()
model = classifier.fit(X_train, y_train)

# test the classifier
y_test = model.predict(X_test)