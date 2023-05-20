import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
nltk.download('wordnet')

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import pickle



def load_data(database_filepath):
    """
    Load data from a SQLite database.

    Parameters:
        database_filepath (str): Filepath of the SQLite database.

    Returns:
        tuple: A tuple containing the following elements:
            X (pandas.Series): The input messages.
            Y (pandas.DataFrame): The output labels.
            category_names (list): The list of category names.

    """
    engine = create_engine("sqlite:///"+database_filepath)
    df = pd.read_sql_table('DisasterDF', engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    Tokenize and preprocess text data.

    Parameters:
        text (str): Input text to be tokenized.

    Returns:
        list: A list of preprocessed tokens.

    """

    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Extract the word tokens from the provided text
    tokens = nltk.word_tokenize(text)
    #Lemmanitizer to remove inflectional and derivationally related forms of a word
    lemmatizer = nltk.WordNetLemmatizer()

    # List of clean tokens
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]
    return clean_tokens


def build_model():
    """
    Build a machine learning model using a pipeline and perform grid search.

    Returns:
        GridSearchCV: A GridSearchCV object configured with the pipeline and parameter grid.

    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ]))
        ])),

        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    # Define the parameter grid for grid search
    param_grid = {
        'features__text_pipeline__count_vectorizer__max_features': [1000, 5000],
        'classifier__estimator__n_estimators': [50, 100],
        'classifier__estimator__learning_rate': [0.1, 0.5]
    }

    # Perform grid search
    cv = GridSearchCV(pipeline, param_grid)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model performance measured by precision, recall, f1-score, support.

    Parameters:
        model: The trained machine learning model.
        X_test (pandas.Series): The input features of the test dataset.
        Y_test (pandas.DataFrame): The true labels of the test dataset.
        category_names (list): The list of category names.

    """
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred, columns = Y_test.columns)

    for column in category_names:
        print("Report for category:", column)
        print(classification_report(Y_test[column], Y_pred[column]))


def save_model(model, model_filepath):
    """
    Save model to a file using pickle.

    Parameters:
        model: The trained machine learning model to be saved.
        model_filepath (str): Filepath to save the model.

    """
    pickle.dump(model, open(model_filepath, 'wb'))


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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()