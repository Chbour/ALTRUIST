import re
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from emoji import demojize
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DistilBertTokenizerFast, BertForSequenceClassification, AutoTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, AutoModelForSequenceClassification

os.environ["WANDB_DISABLED"] = "true"

class DataToTrain(object):
    
    """
    It takes a dataframe with a column called "text" and a column called "label" and returns a
    dictionary of encodings for each of the three splits
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer()
        

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
    It's a class that takes in a dictionary of tensors and a list of labels, and returns a dictionary of
    tensors and a tensor of labels
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class ModelTrainer(object) : 
    """
    It's a class that contains a method to train a model and a method to save the trained model
    """
    def __init__(self):
        super().__init__()


    def compute_metrics(self, pred):
        """
        It takes in a prediction object and returns a dictionary of metrics
        
        :param pred: the prediction object
        :return: A dictionary with the keys 'accuracy', 'f1', 'precision', and 'recall', more can be added
        """
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall}
    

    def trainer(self, training_args, model, train_dataset, val_dataset):
        """
        > The trainer function takes in the training arguments, the model, and the training and validation
        datasets, and returns a Trainer object
        
        :param training_args: The training arguments, defined above
        :param model: the instantiated 🤗 Transformers model to be trained
        :param train_dataset: The training dataset
        :param val_dataset: The validation dataset
        :return: A Trainer object
        """
        return Trainer(
            model = model,                         # the instantiated 🤗 Transformers model to be trained
            args = training_args,                  # training arguments, defined above
            compute_metrics = self.compute_metrics,
            train_dataset = train_dataset,         # training dataset
            eval_dataset = val_dataset)             # evaluation dataset


    def save_model(self, trainer, path_to_save):
        """
        It takes in a trained model and saves it in the given path
        
        :param trainer: The trainer object that you created in the previous step
        :param path_to_save: The path where you want to save the model
        """
        trainer.save_model(path_to_save)
        print("Model was saved in the given path.")