import io
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader

import transformers
from transformers import AutoTokenizer
from nltk.tokenize import TweetTokenizer
from emoji import demojize
import re

class Data(object):

    """
    The class takes in a dataframe and returns the train, validation and test data in the form of a
    dictionary. 
    The dictionary contains the following keys: 
    - input_ids
    - attention_mask
    - token_type_ids
    - labels
    The values of the dictionary are the corresponding values for the keys.  
    The class also contains a method called normalizeTweet which is used to normalize the tweets. 
    The normalizeTweet method is used to normalize the tweets.
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer()
        
    def get_full_text(self, dataframe):
        """
        If the tweet is an extended tweet, then the full text is the full text of the extended tweet.
        Otherwise, the full text is the text of the tweet
        
        :param dataframe: the dataframe you want to add the full text to
        :return: The dataframe with the full text of the tweet.
        """
        dataframe["fulltext"]=""
        for i in range(len(dataframe.index)):
            if type(dataframe['extended_tweet'][i]) != float: 
                dataframe['fulltext'][i]= dataframe['extended_tweet'][i]['full_text']
            else: 
                dataframe['fulltext'][i]= dataframe['text'][i]
        return dataframe
        
    def normalize_token(self, token):
        """
        It takes a token, lowercases it, and then checks if it starts with @, http, or www. If it does, it
        returns a string. If it doesn't, it checks if the token is a single character. If it is, it returns
        the token. If it isn't, it checks if the token is a special character. If it is, it returns a
        string. If it isn't, it returns the token
        
        :param token: The token to be normalized
        :return: The normalized token.
        """
        lowercased_token = token.lower()
        if token.startswith("@"):
            return "@USER"
        elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
            return "HTTPURL"
        elif len(token) == 1:
            return demojize(token)
        else:
            if token == "’":
                return "'"
            elif token == "…":
                return "..."
            else:
                return token
            
    def normalize_tweet(self, tweet):
        """
        It takes a tweet, tokenizes it, normalizes each token, and then puts it back together
        
        :param tweet: The tweet to be normalized
        :return: A string of the normalized tweet.
        """

        tokens = self.tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
        normTweet = " ".join([self.normalizeToken(token) for token in tokens])

        normTweet = normTweet.replace("cannot ", "can not ").replace("n't ", " n't ").replace("n 't ", " n't ").replace("ca n't", "can't").replace("ai n't", "ain't")
        normTweet = normTweet.replace("'m ", " 'm ").replace("'re ", " 're ").replace("'s ", " 's ").replace("'ll ", " 'll ").replace("'d ", " 'd ").replace("'ve ", " 've ")
        normTweet = normTweet.replace(" p . m .", "  p.m.") .replace(" p . m ", " p.m ").replace(" a . m .", " a.m.").replace(" a . m ", " a.m ")

        normTweet = re.sub(r",([0-9]{2,4}) , ([0-9]{2,4})", r",\1,\2", normTweet)
        normTweet = re.sub(r"([0-9]{1,3}) / ([0-9]{2,4})", r"\1/\2", normTweet)
        normTweet = re.sub(r"([0-9]{1,3})- ([0-9]{2,4})", r"\1-\2", normTweet)
    
        return " ".join(normTweet.split())
    
    def prepare_data(self, data):
        """
        We take the data, normalize it, split it into train, validation and test sets, tokenize it, and
        return the tokenized data
        
        :param data: The dataframe containing the text and labels
        :return: the train_encodings, train_labels, val_encodings, val_labels, test_encodings, test_labels
        """
        text = data["text"].map(self.normalizeTweet).values.tolist()
        labels = data["label"].values.tolist()
        train_texts, test_texts, train_labels, test_labels = train_test_split(text, labels, test_size=0.33)
        train_texts, val_texts, train_labels, val_labels = train_test_split(text, labels, test_size=0.2)
        print("Train: {}".format(len(train_texts)))
        print("Val: {}".format(len(val_texts)))
        print("Test: {}".format(len(test_texts)))
        tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
        train_encodings = tokenizer(train_texts, truncation=True, padding=True)
        val_encodings = tokenizer(val_texts, truncation=True, padding=True)
        test_encodings = tokenizer(test_texts, truncation=True, padding=True)
        print(val_encodings.keys())
        return train_encodings, train_labels, val_encodings, val_labels, test_encodings, test_labels

class TweetDataSet(torch.utils.data.Dataset):

    """
    It takes in a dictionary of numpy arrays and returns a dictionary of tensors
    """

    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        for key in self.encodings.keys():
            return len(self.encodings[key])
        #return len(self.labels)
        
    def proba_to_category(self, row):
        """
        If the probability of the first class is less than 0.5 and the probability of the second class is
        greater than 0.5, then return 1. Otherwise, return 0
        
        :param row: the row of the dataframe that we're currently looking at
        :return: the predicted category.
        """
      #print(row)
        score_0, score_1 = row.iloc[0], row.iloc[1]
        if score_0 < 0.5 and score_1 >= 0.5:
            return 1
        else: return 0