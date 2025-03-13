import math
import numpy as np

class Naive_Bayes:
    
    """
    NOTES: 
        ·Probabilities are returned on a logarithmic scale to avoid numerical instability due to rounding issues.
        ·'alpha' (AKA α) is used as the smoothing parameter. 
    """
    
    def __init__(self, alpha=1):
        self.vocabulary = []
        self.dict_positive = {}
        self.dict_negative = {}
        self.probs_table = []
        self.alpha = alpha

    def preprocess_text(self, text):
        if text is not None:
            text = str(text)
        else:
            text = ""
        return text.split()

    def build_dictionaries(self, X_train, y_train):
        dict_positive_temp = {}
        dict_negative_temp = {}
        
        for text, target in zip(X_train, y_train):
            key_words = self.preprocess_text(text)
            
            if target == 1:  # Positive tweet
                for key_word in key_words:
                    dict_positive_temp[key_word] = dict_positive_temp.get(key_word, 0) + 1
            else:  # Negative tweet
                for key_word in key_words:
                    dict_negative_temp[key_word] = dict_negative_temp.get(key_word, 0) + 1
        
        self.dict_positive = dict_positive_temp
        self.dict_negative = dict_negative_temp

    def build_vocabulary(self, X_train):
        vocabulary_temp = set()
        for text in X_train:
            key_words = self.preprocess_text(text)
            vocabulary_temp.update(key_words)
        self.vocabulary = vocabulary_temp

    def compute_probabilities(self, word, dictionary, total_class_words, vocab_size):
        
        """
        Applying:
        
        P(w_i | C) = (frequency(w_i, C) + α) / (|C| + α * |V|)

        Where:
            w_i is the current word being processed.
            C is the class of each tweet (either positive or negative).
            P(w_i | C) is the probability of the word w_i belonging to the class C.
            frequency(w_i, C) is the number of times w_i has appeared in tweets of class C.
            α is the smoothing parameter (1 for Laplace smoothing).
            V is the set of all unique words in the vocabulary.
            |X| is the number of elements in the set X.
        """

        freq = dictionary.get(word, 0)
        prob = (freq + self.alpha) / (total_class_words + self.alpha * vocab_size)
        return math.log(prob, 2)

    def build_probs_table(self):
        total_positive_words = sum(self.dict_positive.values())
        total_negative_words = sum(self.dict_negative.values())
        vocab_size = len(self.vocabulary)
        
        if len(self.dict_positive) == 0:
            self.default_pos_value = -math.inf
        else   :
            self.default_pos_value = math.log(self.alpha/(sum(self.dict_positive.values())+self.alpha*len(self.dict_positive)))
        
        if len(self.dict_negative) == 0:
            self.default_neg_value = -math.inf
        else:
            self.default_neg_value = math.log(self.alpha/(sum(self.dict_negative.values())+self.alpha*len(self.dict_negative)))
        
        probs_table_temp = {}
        
        for word in self.vocabulary:
            log_P_w_knowing_positive = self.compute_probabilities(
                word, self.dict_positive, total_positive_words, vocab_size
            )
            log_P_w_knowing_negative = self.compute_probabilities(
                word, self.dict_negative, total_negative_words, vocab_size
            )
            
            probs_table_temp[word] = {
                "positive": log_P_w_knowing_positive,
                "negative": log_P_w_knowing_negative
            }
        
        self.probs_table = probs_table_temp

    def classify_tweet(self, tweet):
        key_words = self.preprocess_text(tweet)
        
        log_prob_positive = math.log(self.prob_positive, 2)
        log_prob_negative = math.log(self.prob_negative, 2)
        
        for word in key_words:
            log_prob_positive_word = self.probs_table.get(word, {}).get("positive", self.default_pos_value)
            log_prob_negative_word = self.probs_table.get(word, {}).get("negative", self.default_neg_value)
            
            if log_prob_positive_word > -math.inf:
                log_prob_positive += log_prob_positive_word
            if log_prob_negative_word > -math.inf:
                log_prob_negative += log_prob_negative_word
        
        return 1 if log_prob_positive > log_prob_negative else 0

    def classify_tweets(self, X_test):
        return [self.classify_tweet(tweet) for tweet in X_test]
    
    def fit(self, X_train, y_train):
        
        if not isinstance(y_train, np.ndarray):
            try:
                y_train = np.array(y_train)
            except (ValueError, TypeError) as e:
                raise ValueError("y_train must be a numpy array. The conversion attempt failed.") from e
            
        """
        Next step calculates Dirichlet priors, taking into account the smoothing.
        
        P (C = Ck) = (#D{C = Ck} + α) / (|D| + α * 2)
        
        Where:
            P (C = Ck) is the prior of Ck class
            #D{C = Ck} is the number of tweets belonging to Ck class
            α is the smoothing parameter (1 for Laplace smoothing)
            |D| is the total number of tweets
        """
        
        self.prob_positive = (len(y_train[y_train == 1]) + self.alpha) / (len(y_train) + self.alpha * 2)
        self.prob_negative = (len(y_train[y_train == 0]) + self.alpha) / (len(y_train) + self.alpha * 2)
        
        self.build_vocabulary(X_train)
        self.build_dictionaries(X_train, y_train)
        self.build_probs_table()
        
    def predict(self, X_test):
        return self.classify_tweets(X_test)
