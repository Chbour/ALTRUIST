import pandas as pd
import contractions
import re
import swifter
import json
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer, util

class PreprocessTimeline(object):

    """
    The class PreprocessTimeline takes a timeline and returns a preprocessed timeline.
    These different steps will improve the similarities and remove tweets with no information.
    """

    def __init__(self, path_to_data="config.json"):
        with open(path_to_data) as f:
            data = json.load(f)
        self.keywords = data["twitter"]["keywords"]
        self.key_concept = data["concepts_keywords"]

    def get_full_text(self, tweet):
        """
        If the tweet has an extended tweet field, return the full text of the extended tweet. Otherwise, return
        the text of the tweet. Has to be applied to the whole timeline.
        
        :param tweet: the tweet object
        :return: The full text of the tweet.
        """
        try:
            return tweet["extended_tweet"]["fulltext"]
        except:
            return tweet["text"]


    def remove_url_rts_mentions(self, timeline):
        """
        This function takes a dataframe as an input and returns a dataframe with the same columns but with
        the rows that start with 'RT @' removed (in case we collected some), and a new column named prep
        with tweet where we removed urls and no user mentions. We also remove NaN values in the newly 
        created column.
        
        :param timeline: the dataframe of tweets
        :return: A dataframe with the tweets that are not retweets, no user mention and do not contain a url.
        """
        timeline = timeline.drop_duplicates(subset=['id'])
        timeline['prep'] = timeline['text'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
        timeline = timeline[~timeline['prep'].astype(str).str.startswith('RT @')]
        timeline["prep"] = timeline["prep"].apply(lambda x: re.sub(r'@\w+', '', x))
        timeline = timeline.dropna(subset=["prep"]).reset_index(drop=True)
        return timeline


    def translate(self, timeline): 
        """
        The function takes a dataframe as an input, and for each row in the dataframe, if the language is
        not English, it translates the text in that row to English. 
        
        The function returns the dataframe with the translated text. 
        
        The function uses the GoogleTranslator library to translate the text. 
        
        The function is called translate. 
        
        The function takes a dataframe as an input. 
        
        The function returns a dataframe.
        
        :param timeline: the dataframe that contains the tweets
        :return: A dataframe with the translated tweets.
        """
        for i in range(len(timeline.index)):
            if timeline["lang"][i] != "en":
                timeline["text"][i] = GoogleTranslator(source='auto', target='en').translate(timeline["text"][i])
        return timeline


    def format_date(self, timeline):
        """
        It takes a dataframe, converts the 'created_at' column to a datetime object, sorts the dataframe by
        date, creates a new column called 'normalised_date' which is the date without the time, and then
        resets the index
        
        :param timeline: the dataframe containing the tweets
        :return: A dataframe with the date column converted to datetime format, sorted by date, and a new
        column with the date normalised.
        """
        timeline["date"] = pd.to_datetime(timeline["created_at"])
        timeline = timeline.sort_values(by='date').reset_index(drop=True)
        timeline['normalised_date'] = timeline['date'].dt.normalize()
        timeline = timeline.reset_index(drop=True)
        return timeline


    def remove_contractions(self, tweet):
        """
        It takes a tweet, removes the contractions, and returns the tweet
        
        :param tweet: The tweet to be processed
        :return: The tweet["prep"] is being returned.
        """
        return contractions.fix(tweet["prep"])
    


    def remove_empty_tweets(self, timeline, min_len):
        """
        It removes tweets that are less than a certain length.
        
        :param timeline: the dataframe containing the tweets
        :param min_len: minimum length of a tweet to be considered
        :return: A dataframe with the tweets that have a length greater than the min_len parameter.
        """
        L=[]
        for i in list(timeline.index):
            if len(timeline["prep"][i].split())<=min_len:
                L.append(i)
        timeline = timeline.drop(L)
        timeline = timeline.reset_index(drop=True)
        return timeline
    

    def t0(self, timeline, concept, threshold): 
        """
        If the first value in the timeline is greater than or equal to the threshold, return the index of
        that value. If not, return the index of the last value in the timeline.
        
        :param timeline: the timeline of the concept
        :param concept: the concept you want to look at
        :param threshold: the threshold for the concept to be considered "active"
        :return: The index of the first value in the timeline that is greater than or equal to the
        threshold.
        """
        try:
            return next(x[0] for x in enumerate(timeline[concept]) if x[1] >= threshold)
        except:
            return timeline.tail(1).index.stop


    def define_t0(self, timeline, threshold, model): 
        """
        - The function takes in a timeline and a threshold. 
        - It then calculates the cosine similarity between the first key concept and the timeline. 
        - It then finds the first index where the cosine similarity is higher than the threshold. 
        - It then finds the first index where the keywords appear in the timeline. 
        - It then returns the minimum of the two. 
        - Finally the timeline is truncated to keep only tweets after this min.
        
        :param timeline: the dataframe of the timeline
        :param threshold: the threshold for the cosine similarity between the sentence and the key concept
        """
        model = SentenceTransformer(model)
        timeline[list(self.key_concept.keys())[0]] = timeline["prep"].swifter.apply(lambda x: util.cos_sim(model.encode(x, convert_to_tensor=True), model.encode(self.key_concept[list(self.key_concept.keys())[0]], convert_to_tensor=True))[0][0].item())
        first_index_higher_threshold = self.t0(timeline, list(self.key_concept.keys())[0], threshold)
        list_index_words = [ i for i in timeline.index if any(word in timeline["prep"][i] for word in self.keywords)]
        try:
            first_index_word_appear = list_index_words[0]
        except: 
            first_index_word_appear = timeline.tail(1).index.stop
        t0_timeline = min(first_index_higher_threshold, first_index_word_appear)
        timeline = timeline[t0_timeline:].reset_index(drop=True)
        return timeline