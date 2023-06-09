from email import message_from_binary_file
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load and merge data from message and categories CSV files.

    Parameters:
        messages_filepath (str): Filepath of the messages CSV file.
        categories_filepath (str): Filepath of the categories CSV file.

    Returns:
        pandas.DataFrame: Merged DataFrame containing message and categories data.

    """

    msg_df = pd.read_csv(messages_filepath)
    cat_df = pd.read_csv(categories_filepath)
    df = msg_df.merge(cat_df, on=('id'))
    return df


def clean_data(df):
    """
    Clean and preprocess the input DataFrame.

    Parameters:
        df (pandas.DataFrame): The input DataFrame to be cleaned.

    Returns:
        pandas.DataFrame: The cleaned DataFrame.

    """

    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(pat = ";", expand=True)
    row = categories.loc[1,:]
    # use this row retzuto extract a list of new column names for categories.
    category_colnames = list(map(lambda s: s[:-2], row))
    # rename the columns of `categories`
    categories.columns = category_colnames
    # convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1].astype(int)
    # drop the original categories column from `df`
    df = df.drop("categories", axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df = df.drop_duplicates()
    # recode "related" column to binary values
    df["related"] = df["related"].map(lambda x: 0 if x == 0 else 1)
    return df


def save_data(df, database_filename):
    """
    Save a DataFrame to a SQLite database.

    Parameters:
        df (pandas.DataFrame): The DataFrame to be saved.
        database_filename (str): Filepath of the SQLite database.

    Returns:
        bool: True if the data is successfully saved, False otherwise.

    """
    
    #engine = create_engine("sqlite:///"+database_filename)
    #df.to_sql('DisasterDF', engine, index=False, if_exists="replace") 
    try:
        engine = create_engine("sqlite:///" + database_filename)
        df.to_sql('DisasterDF', engine, index=False, if_exists="replace")
        return True
    except Exception as e:
        print(f"Error occurred while saving data to the database: {str(e)}")
        return False


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()