import pandas as pd
import json
from sentence_transformers import SentenceTransformer, util

class DefineThreshold(object):
    """
    1. It takes a dataframe, a model name, and a path to a dictionary of keywords as input. 
    2. It then applies the cosine similarity between the sentence embeddings of the dataframe and the
    keywords. 
    3. It then sorts the dataframe by the cosine similarity and prints the top 500 sentences.
    """

    def __init__(self, dataframe, model_name, path_dict_concept_keywords="config.json"):
        with open(path_dict_concept_keywords) as f:
            data = json.load(f)
        self.model = SentenceTransformer(model_name)
        self.df = dataframe
        self.dict_concept_keywords = data["concepts_keywords"]

    def apply_similarities(self):
        df = self.df
        model = self.model
        for concept in list(self.dict_concept_keywords.keys()): 
            df[concept] = df["prep"].swifter.apply(lambda x: util.cos_sim(model.encode(x, convert_to_tensor=True), model.encode(self.dict_concept_keywords[concept], convert_to_tensor=True))[0][0].item())
            df = df.sort_values(by=concept, ascending=False).reset_index(drop=True)
            for i in range(0,500):
                print(df["prep"][i],df[concept][i])