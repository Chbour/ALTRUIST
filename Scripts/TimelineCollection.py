import tweepy
import csv
from tqdm import tqdm
import time
import pandas as pd

class TimelineCollection(object):

    """
    It takes a list of user_ids and collects their timelines, then inserts them into a mongodb
    collection
    """

    def __init__(self, db, api):
        self.mongo = db
        self.api = api

    def exists(self, userid):
        """
        If the user exists, return True, else return False
        
        :param userid: The userid of the user you want to check
        :return: True or False
        """
        api = self.api
        try: 
            api.get_user(userid)
            return True
        except tweepy.TweepError:
            return False


    def get_all_tweets(self, user_id):
        """
        It takes a user_id and returns a list of all the tweets from that user
        
        :param user_id: The ID of the user for whom to return results for
        :return: A list of tweets
        """
        api = self.api
    #Twitter only allows access to a users most recent 3240 tweets with this method
    
        #initialize a list to hold all the tweepy Tweets
        alltweets = []  
    
        #make initial request for most recent tweets (200 is the maximum allowed count)
        new_tweets = api.user_timeline(user_id = user_id,count=200)
    
        #save most recent tweets
        alltweets.extend(new_tweets)
    
        #save the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1
    
        #keep grabbing tweets until there are no tweets left to grab
        while len(new_tweets) > 0:
            #print(f"getting tweets before {oldest}")
        
            #all subsiquent requests use the max_id param to prevent duplicates
            new_tweets = api.user_timeline(user_id = user_id,count=200,max_id=oldest, tweet_mode="extended")
        
            #save most recent tweets
            alltweets.extend(new_tweets)
        
            #update the id of the oldest tweet less one
            oldest = alltweets[-1].id - 1
            #print(f"...{len(alltweets)} tweets downloaded so far")
    
        return alltweets


    def collect_timelines(self, users_to_collect):
        """
        It takes a list of user_ids and collects all of their tweets (except retweets), then inserts them into a MongoDB
        collection.
        
        :param users_to_collect: list of user ids to collect timelines for
        """
        db = self.mongo
        for user_id in tqdm(users_to_collect):
            try:
                timeline = self.GetAllTweets(user_id=int(user_id))
                db.test_timeline.insert_many([el._json for el in timeline if not el._json[list(el._json.keys())[3]].startswith("RT @")])
            except tweepy.TweepError:
                pass
            except IndexError:
                pass
            except TypeError:
                pass
            except Exception as e:
                print(e)
                break