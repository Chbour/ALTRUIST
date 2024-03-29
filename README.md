# ALTRUIST

![ALTRUIST](https://user-images.githubusercontent.com/70756697/215432914-748359b1-8d62-4a10-b0e0-488c626aa46c.png)
stUdy usIng
Social media daTa
## About this package
ALTRUIST stands for virtu**AL** digi**T**al coho**R**t st**U**dy us**I**ng **S**ocial media da**T**a. It is a Python package that aims to emulate a cohort on Twitter data. An article to present the package and a use case on diabetes have been submitted for publication.

## How to use? 
It is suggested to `git-clone` this repository and to modify the notebook file according to your use case.

## Files description
### Scripts / .py files
- 1. `MongodbInit` : initialisation and connexion to mongodb using the connexion data in connection_data.txt file
- 2. `DataCollection` : Collection of tweets using keywords defined in keywords.txt
- 3. `TrainPersonalClassifier` : Creation of a personal content classifier from a CSV file of tweets labeled according to the fact that they relate a personal experience with the user and their own disease. The use has to randomly extract tweets and manually label them to train the classifier. 
- 4. `ApplyClassifier` : Classifies collected tweets and returns personal ones
- 5. `TimelineCollection` : checks if a user still exists using their id and returns list of existing ids. Then, allows to collect all their timelines.
- 6. `TimelinePreprocessing` : different preprocessing steps to prepare data to apply similarities, allows also to define users' inclusion time (definition of t0 should be done after the next 2 steps).
- 7. `ChooseTransformer` : apply transformer to a dataframe of preprocessed tweets to see if the model is suitable for the task. A list of different transformers can be used as an argument.
- 8. `DefThreshold` : once the model is chosen, use it on the different concepts to define the threshold from which the similarities between concepts and tweets are representative.
- 9. `TimelineSimilarities` : Takes all preprocessed timelines and applies similarities between each tweets and the different concepts. The occurrences of exposures and outcomes are then followed and a "cohort type" dataset is built. 
- 10. `Data2Cox` : Applies Cox model using lifelines package. 

### Notebook file
`Notebook_example_diabetes.ipynb` : example of how the scripts can be used to perform the different steps of a cohort on diabetes.

### config.json
The config file is a nested json file with: 
- `mongo` : a dictionnary to be filled with the connexion data to mongodb,
- `twitter` : a dictionnary to be filled with the connextion details to the Twitter API,
- `concept_keywords` : a dictionnary to be filled with the keywords associated with the concepts,
- `thresholds` : a dictionnary to be filled with the thresholds associated with each concept.
