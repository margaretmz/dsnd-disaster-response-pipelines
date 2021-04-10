import sys

import nltk

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier

# for 6. Improve your model
from sklearn.model_selection import GridSearchCV

# 8. Further improve model
from sklearn.linear_model import SGDClassifier

# Export your model as a pickle file
import pickle


def load_data(database_filepath):
    """
    Load data from a sqlite db, create dataframe for training data
    :param database_filepath: database file path
    :return: training data, labels (categories), category names
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    # engine.table_names() # get a list of table names
    df = pd.read_sql_table("message", engine)

    X = df.message
    Y = df.iloc[:, 4:]

    return X, Y, Y.columns.values


def tokenize(text):
    """
    Tokenize and clean the text
    :param text: input text
    :return: tokenized data
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    detected_urls = re.findall(url_regex, text)

    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # tokenize text
    tokens = word_tokenize(text)

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(SGDClassifier(max_iter=5, tol=None)))])

    parameters = {
        'clf__estimator__max_depth': [2, 5, None],
        'clf__estimator__n_estimators': [10, 20]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=4)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    # Save the model to a pickle file
    :param model: trained classifier
    :param model_filepath: model file path and name
    :return: none
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/disaster_response.db classifier.pkl')


if __name__ == '__main__':
    main()
