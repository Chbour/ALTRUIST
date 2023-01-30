# Cohort-package

## py files
1. **Mongodb_init** : initialisation and connexion to mongodb using the connexion data in connection_data.txt file
2. **data_collection** : Collection of tweets using keywords defined in keywords.txt
3. **TrainPersonalClassifier** : Creation of a personal content classifier from a CSV file of tweets labeled according to the fact that they relate a personal experience with the user and their own disease.
4. **ApplyClassifier** : Classifies collected tweets and returns personal ones
5. **TimelineCollection** : checks if a user still exists using their id and returns list of existing ids. Then, allows to collect all their timelines.
6. **TimelinePreprocessing** : different preprocessing steps to prepare data to apply similarities, allows also to define users' inclusion time (definition of t0 should be done after the next 2 steps).
7. **ChooseTransformer** : apply transformer to a dataframe of preprocessed tweets to see if the model is suitable for the task. A list of different transformers can be used as an argument.
8. **DefThreshold** : once the model is chosen, use it on the different concepts to define the threshold from which the similarities between concepts and tweets are representative.
9. **TimelineSimilarities** : Takes all preprocessed timelines and applies similarities between each tweets and the different concepts.
10. **Data2Cox** : takes an outcome and an exposition as argument and, for each timeline, identifies the first mention to the outcome and then counts the number of times the exposition's threshold is exceeded. Then, this count is converted to binary (users who mention expo vs users who doesn't) and Cox model is applied.

## Notebook file
**Notebook_example.ipynb** : example of how the scripts can be used to perform the above steps.

## text files
- **connection_data.txt** : connexion strings to mongodb, path to tlsfile (if needed) and key tokens to connect to the Twitter API
- **dict_concept_keywords.txt** : dict which match each concept to the list of keywords. (eg "Food" : "food, nutrition, diet")
- **dict_threshold.txt** : dict matching each concept to the threshold defined in step 8. (eg "Food" : 0.34)
- **keywords.txt** : all the disease-oriented keywords to collect tweets related to the disease.
