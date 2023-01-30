#import libraries
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import pandas as pd
from tqdm import tqdm
import json
import os
import swifter
from sentence_transformers import SentenceTransformer, util

os.environ["CUDA_VISIBLE_DEVICES"]="3" #connexion to gpu

class TestModel(object):
    """
    The class takes in a path to a json file that contains a dictionary of keywords. The class then has
    a method that takes in a dataframe and a list of models. The method then loops through the models
    and the keywords and creates a new column for each keyword in the dataframe. The new column contains
    the cosine similarity between the sentence in the dataframe and the keyword. 
    The class is used like this:  
    test = TestModel("path/to/json/file")
    df = test.TryModels(df, ["model1", "model2"]), models can be found on https://huggingface.co/models?library=sentence-transformers
    """

    def __init__(self, path_dict_keywords="data/dict_concepts_keywords.txt"):
        self.keywords = json.load(open(path_dict_keywords))
    
    def TryModels(self, df, models): 
        for model in models: 
            model = SentenceTransformer(model)
            for concept in list(self.keywords.keys()):
                df[concept] = df["prep"].swifter.apply(lambda x: util.cos_sim(model.encode(x, convert_to_tensor=True), model.encode(self.keywords[concept], convert_to_tensor=True))[0][0].item())
        return df
