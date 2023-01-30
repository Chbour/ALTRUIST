import torch
import pandas as pd
from tqdm import tqdm
import datetime
import json
import os
import swifter
from sentence_transformers import SentenceTransformer, util
# os.environ["CUDA_VISIBLE_DEVICES"]="3"

class Similarities(object):
    """
    For each user timeline, we apply the similarities between the define concepts/keywords and the
    then prepare directly the data to prepare the "cohort-like" table
    """
    def __init__(self, db, chosen_model, list_ids, path_dict_threshold="dict_threshold.txt", path_dict_keywords="dict_concepts_keywords.txt"):
        self.mongo = db
        self.model = SentenceTransformer(chosen_model)
        self.ids = list_ids
        self.thresholds = json.load(open(path_dict_threshold))
        self.keywords = json.load(open(path_dict_keywords))
    
    def ApplySimilarities(self, exposure:str, outcome:str):
        """
        It takes user's timeline per user's timeline, and for each tweet, it computes the similarity between the tweet and
        each of the keywords related to a concept and then send it back in a mongodb collection
        """
        db = self.mongo
        model = self.model
        for userid in tqdm(self.ids):
            k=0
            user_df = pd.DataFrame(db.Preprocessed_timelines.find({"user.id" : userid}))
            user_df = user_df[["created_at", "id", "text", "user", "prep", "normalised_date"]] 
            if len(user_df.index) > 100:
                for concept in [exposure, outcome]:
                    user_df[concept] = user_df["prep"].swifter.apply(lambda x: util.cos_sim(model.encode(x, convert_to_tensor=True), model.encode(self.keywords[concept], convert_to_tensor=True))[0][0].item())
                user_df = user_df.drop_duplicates(subset=['id'])
                t0 = user_df["normalised_date"][0]
                tn = user_df["normalised_date"][user_df.index[-1]]
                df_quarter = pd.DataFrame()
                t = t0 + relativedelta(months = +3) - relativedelta(days = 1)
                krows = len(user_df.index)
                quarter = 1
                while krows > 0: 
                    df_user_quarter = user_df[(user_df['normalised_date'] >= t0) & (user_df['normalised_date'] <= t)]
                    df_user_quarter["quarter"] = "Q"+str(quarter)
                    df_quarter = pd.concat([df_quarter, df_user_quarter])
                    quarter +=1
                    t0 , t = t + relativedelta(days = +1), t + relativedelta(months = +3)
                    krows = krows - len(df_user_quarter.index)
                try: 
                    index = next(x[0] for x in enumerate(df_quarter[outcome]) if x[1] >= outcome_threshold)
                    date = df_quarter["normalised_date"][index]
                    index_lastnews = df_quarter.index[df_quarter['normalised_date'] == date].tolist()[-1] + 1
                    k = 1
                except:
                    index_lastnews = len(df_quarter.index)
                    k = 0
            df_quarter = df_quarter[:index_lastnews]
            list_quarters = list(dict.fromkeys(df_quarter["quarter"].tolist()))
            for Q in list_quarters:
                df_under_quart = df_quarter[df_quarter["quarter"] == Q]
                df_count = df_count.append({"user" : userid,
                                "d_t0" : df_under_quart.iloc[0]["normalised_date"],
                                "date_tn" : df_under_quart.iloc[-1]["normalised_date"],
                                "exposition" : 1 if any(df_under_quart[expo] >= expo_threshold) else 0,
                                "outcome" : k}, ignore_index = True)
            if list_ids.index(userid) % 1000 == 0:
                df_count.to_csv("df_"+exposure+"_"+outcome+".csv")
        return df_count