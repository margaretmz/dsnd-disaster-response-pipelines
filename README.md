# Disaster Response Pipeline Project

This project is part of the [Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025).
I analyze disaster data from [Figure Eight](https://www.figure-eight.com/) and built an ML pipeline to classify the messages,
so that they can be sent to the appropriate disaster relief agency.

### Repo structure

There are 3 main components in this repo:
1. ETL Pipeline
Python script `process_data.py` provides the data pipeline:
   * loads the `messages` and `categories` datasets from 2 csv files
   * Merges the 2 dataframes
   * Cleans the data
   * Stores the data in a SQLite database called `disaster_response.db`
   
Note: The code in `ETL Pipeline Preparation.ipynb` notebook is similar to the 'process_data.py' script.

2. ML Pipeline
Python script `train_classifier.py` provides the ML pipeline:
   * Loads data from the SQLite database
   * Splits the dataset into training and test sets
   * Builds a text processing and machine learning pipeline
   * Trains and tunes a model using GridSearchCV
   * Outputs results on the test set
   * Exports the final model as a pickle file: `classifier.pkl`
   
Note: The code in `ML Pipeline Preparation.ipynb` notebook is similar to the 'train_classifier.py' script.

3. Flask Web App
The web app provides the following functionality:
   * Data visualization of the messages
   * Allow user to enter a message
   * Outputs the categories of the message, classified by the ML model

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ to view the web app

### Dependencies
The following libraries were required for the project: 

* Python 3.5+
* Data processing & ML: pandas, numpy, scikit-learn & sqlalchemy
* Natural Language Processing: NLTK
* Model saving and loading: Pickle
* Data visualization: Matplotlib & seaborn
* Web app: Flask & Plotly

### How I completed the project
Step 1. First I worked in 2 notebooks: ETL Pipeline Preparation.ipynb & ML Pipeline Preparation.ipynb.
Step 2. Then I completed the 2 scripts `process_data.py` and `train_classifier.py` with the code from the notebooks
Step 3. For the web app, I used PyCharm as the IDE and used Virtualenv to create a 
virtual environment (will all the required dependencies) for the project. The main work for the web app was data visualisation, which was written in Python with the help of Plotly in `run.py`.  
  