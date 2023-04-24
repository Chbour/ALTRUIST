import numpy as np
import pandas as pd
from tqdm import tqdm
import json

class PrepareCox(object):
    """
    It takes a list of user IDs, a MongoDB database, a CoxPHFitter object, and a path to a JSON file
    containing the thresholds for the concepts of interest. It then returns a dataframe containing the
    data needed to run a Cox regression
    """

    def __init__(self, db, list_ids, cph):
        self.mongo = db
        self.list_ids = list_ids
        self.cph = cph()

    # Returns a DataFrame containing the data for each user in the dataset.
    def prepare_data(self, df_cohort):
        # Convert the date columns to datetime format
        df_cohort[["d_t0", "date_tn"]] = df_cohort[["d_t0", "date_tn"]].apply(pd.to_datetime)
        
        # Add new columns to the DataFrame
        df_cohort["start"] = 0
        df_cohort["end"] = 0
        df_cohort["delta"] = df_cohort["date_tn"] - df_cohort["d_t0"]

        # Create an empty DataFrame
        df = pd.DataFrame()

        # Loop over each user ID
        for ids in tqdm(self.list_ids) : 
            # Get the rows for the current user and reset the index
            userdf = df_cohort[df_cohort["user"]==ids].reset_index(drop=True)
            
            # Loop over each row in the user's DataFrame
            for i in range(len(userdf)):
                # Set the "end" value for the first row to the "delta" value
                if i == 0:
                    userdf["end"][i] = userdf["delta"][i]
                else:
                    # Set the "start" value for each subsequent row to the previous row's "end" value
                    userdf["start"][i] = userdf["end"][i-1]
                    # Set the "end" value for each row to the sum of the "start" and "delta" values
                    userdf["end"][i] = userdf["start"][i] + userdf["delta"][i]
                
                # Concatenate the user's DataFrame with the main DataFrame
                df = pd.concat([df, userdf])

                # Drop unnecessary columns
                df = df.drop(["d_t0", "date_tn", "delta"], axis=1)
        
        # Create a new DataFrame
        df2 = pd.DataFrame()

        # Loop over each unique user ID in the main DataFrame
        for i in list(set(df.user.tolist())):
            # Get the rows for the current user and reset the index
            dfi = df[df["user"] == i].reset_index(drop=True)

             # Find the index of the row where the "outcome" value is 1
            try:
                dfi = dfi.loc[: dfi[(dfi['outcome'] == 1)].index[0], :]

                # Concatenate the current user's DataFrame with the new DataFrame
                df2 = pd.concat([df2, dfi])
            except:
                # Concatenate the current user's DataFrame with the new DataFrame if no row has an "outcome" value of 1
                df2 = pd.concat([df2, dfi])

            # Remove rows where "start" and "end" are both 0
            df2 = df2.loc[~((df2["start"] == df2["end"]) & (df2["start"] == 0))]
            
            # Reset the index of the new DataFrame
            df2 = df2.reset_index(drop=True)
        
        # Convert all columns to numeric format and return the new DataFrame
        df2 = df2.apply(pd.to_numeric)
        return df2
        
   
    def cox(self, dataset_cox, outcome) : 
        """
        The function takes a dataset as an input, fits the Cox Proportional Hazards model to the dataset,
        and returns the summary of the model. 
        
        The function returns the summary of the Cox Proportional Hazards model. 
        
        The Cox Proportional Hazards model is a survival analysis model. The model is used to predict the
        time to event. The model is used to predict the time to event for a given set of covariates. 
        
        The Cox Proportional Hazards model is a semi-parametric model. The model does not make any
        assumptions about the distribution of the time to event.
        
        :param dataset_cox: the dataframe that contains the data to be used for the cox regression
        :return: The summary of the cox model
        """

        cph = self.cph
        cph.fit(dataset_cox, id_col="user", event_col=outcome, start_col="start", stop_col="end", show_progress=True)
        return cph.print_summary()