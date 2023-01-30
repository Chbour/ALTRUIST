import numpy as np
import pandas as pd
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

    def prepare_data(self, df_cohort):
        df_cohort[["d_t0", "date_tn"]] = df_cohort[["d_t0", "date_tn"]].apply(pd.to_datetime)
        df_cohort["start"] = 0
        df_cohort["end"] = 0
        df_cohort["delta"] = df_cohort["date_tn"] - df_cohort["d_t0"]
        df = pd.DataFrame()
        for ids in tqdm(self.list_ids) : 
            userdf = df_cohort[df_cohort["user"]==ids].reset_index(drop=True)
            for i in range(len(userdf)):
                if i == 0:
                    userdf["end"][i] = userdf["delta"][i]
                else:
                    userdf["start"][i] = userdf["end"][i-1]
                    userdf["end"][i] = userdf["start"][i] + userdf["delta"][i]
                df = pd.concat([df, userdf])
                df = df.drop(["d_t0", "date_tn", "delta"], axis=1)
        df2 = pd.DataFrame()
        for i in list(set(df.user.tolist())):
            dfi = df[df["user"] == i].reset_index(drop=True)
            try:
                dfi = dfi.loc[: dfi[(dfi['outcome'] == 1)].index[0], :]
                df2 = pd.concat([df2, dfi])
            except:
                df2 = pd.concat([df2, dfi])
            df2 = df2.loc[~((df2["start"] == df2["end"]) & (df2["start"] == 0))]
            df2 = df2.reset_index(drop=True)
        df2 = df2.apply(pd.to_numeric)
        return df2
        
   
    def cox(self, dataset_cox) : 
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
        cph.fit(dataset_cox, id_col="user", event_col="outcome", start_col="start", stop_col="end", show_progress=True)
        return cph.print_summary()