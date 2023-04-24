from typing_extensions import Self
import tweepy
import json
import time
from pymongo import MongoClient
import datetime

class InitialisationTwitter(object):
    """ 
    It takes a path to a file containing keywords and a path to a file containing connection data, and
    returns a connection to Twitter
    """

    def __init__(self, twitter_connec="config.json"):
        with open(twitter_connec) as f:
            data = json.load(f)
        self.keywords = data["twitter"]["keywords"]
        self.connec_data_twitter = data["twitter"]



    def connection_to_twitter(self):
        """
        It takes the data from the connection_data.json file and uses it to connect to Twitter
        :return: The auth object is being returned.
        """
        CONSUMER_KEY = self.connec_data_twitter["twitter_consumer_key"]
        CONSUMER_SECRET = self.connec_data_twitter["twitter_consumer_secret"]
        ACCESS_TOKEN = self.connec_data_twitter["twitter_access_token"]
        ACCESS_TOKEN_SECRET = self.connec_data_twitter["twitter_token_secret"]
        auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
        auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
        return auth

class StreamListener(tweepy.StreamListener): 
    """
    The class StreamListener inherits from the class tweepy.StreamListener. The class StreamListener has
    a constructor that takes three parameters: db, twitter and api. The constructor calls the
    constructor of the superclass. The class StreamListener has three methods: on_connect(), on_error()
    and on_data(). The method on_connect() prints out a message when the connection to the streaming API
    is established. The method on_error() prints out an error message when an error occurs. The method
    on_data() is called every time a new tweet is received. The method on_data() decodes the JSON from
    Twitter, grabs the 'created_at' data from the Tweet to use for display, changes the created_at
    format to datetime, and stores the tweet in a collection named "tweets_collection".
    """
    def __init__(self, db, twitter, api):
        super().__init__()
        self.mongo = db
        self.twitter = twitter
        self.api = api   

    def on_connect(self):
        """
        It prints out a message when the connection to the streaming API is established.
        """
        print("You are now connected to the streaming API.") #Connection to the streaming API
 
    def on_error(self, status_code): #if Output an error occurs, display the error / status code
        """
        The on_data function is called when a tweet is received. The tweet is then parsed and the text is
        extracted. The text is then cleaned and the cleaned text is then written to a file
        
        :param status_code: The error code
        :return: The status code
        """
        print('An Error has occured: ' + repr(status_code))
        return False
 
    def on_data(self, data): #connection to mongoDB and storing the tweet
        """
        The function on_data() is called every time a new tweet is received. 
        :param data: the data returned from Twitter
        """
        db = self.mongo

        try:
    
            datajson = json.loads(data)  # Decode the JSON from Twitter
            created_at = datajson['created_at'] #grab the 'created_at' data from the Tweet to use for display
            datajson["created_at"] = datetime.datetime.strptime(datajson["created_at"],"%a %b %d %H:%M:%S %z %Y") #Change created_at format to datetime
            tweetID = datajson["id"]

            #It is possible that the Stream will try to collect several time the same tweet. 
            #In order to avoid duplicates, we check if the tweet is not already in the collection.
            if len(list(db.tweets_collection.find({"id":tweetID})))==0:
                print("Tweet collected at " + str(created_at))  #print out a message 
                db.tweets_collection.insert_one(datajson)  # !!! Data will be stored in a collection named "tweets_collection". 
                                                           # If you want to have another name, change "tweets_collection".
                             
        except Exception as e:
            print(e)