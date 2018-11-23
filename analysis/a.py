def countPositiveCapitalized(tokens):
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
def countNegativeCapitalized(tokens):
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
def hasCapitalized(tokens):
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
def countHashtags(tokens):
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
def countPositive(tokens):
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
def countNegative(tokens):
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
def countNeutral(tokens):
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
def countCapitalizedWords(tokens):
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
def countSpecialCharacters(tokens):
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

def countSpecificSpecialCharacter(specialCharacter, tokens):
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

def fixSpelling(tokens):
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

def stemming(tokens):
    '''
    Apply stemming to each token
    '''

    stemmer = SnowballStemmer("english")  
    stemmed = [stemmer.stem(w) for w in tokens.split()]
    return stemmed

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