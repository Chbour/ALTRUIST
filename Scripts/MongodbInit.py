from typing_extensions import Self
import json
from pymongo import MongoClient

import json
from pymongo import MongoClient


class MongoInitialisation(object):
    """
    Authenticate with MongoDB, data related to mongodb connexion is stored in config.json file
    """
    
    def __init__(self, path_to_data="config.json"):
        self.connec_data = self.connection_data(path_to_data)

    def connection_data(self, path_to_data): 
        """
        It opens the file, reads the data, loads the data into a json object, and returns the json object
        
        :param path_to_data: This is the path to the JSON file that contains the connection data
        :return: The data from the json file.
        """
        with open(path_to_data) as f:
            data = f.read()
        js = json.loads(data)
        return js

    def mongodb_connection(self):
        """
        It connects to the MongoDB database.
        :return: The database object.
        """
        MONGO_HOST= self.connec_data["mongo"]["mongo_string"]
        try:
            client = MongoClient(MONGO_HOST, 
                                tls = self.connec_data["mongo"]["tls"],
                                tlsCAFile = self.connec_data["mongo"]["mongo_tlsCAFile"] if self.connec_data["mongo"]["tls"] else False)
            db = client.database # Use database (If it doesn't exist, it will be created)
            return db
        except Exception as e:
            return e, 
