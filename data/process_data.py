import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load disaster messages and disaster categories csv files
    and merge them into a single dataframe
    :param messages_filepath: filepath to messages csv file
    :param categories_filepath: filepath to categories csv file
    :return dataframe: merged dataframe with disaster response messages and their categories
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath, dtype=str)
    # load categories dataset
    categories = pd.read_csv(categories_filepath, dtype=str)
    # merge datasets
    df = messages.merge(categories, on=['id'])

    return df

def clean_data(df):
    """
    Clean the dataframe by creating the category columns, remove dups and invalid data
    :param df: dataframe with disaster response messages and their categories
    :return: cleaned dataframe
    """

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # Convert category values to just numbers 0 or 1
        categories[column] = categories[column].astype(str).str[-1:]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # 5. Replace categories columns
    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, sort=False)

    # 6. Remove dups
    df.drop_duplicates(inplace=True)

    # 7. Remove invalid category column/rows
    # Drop the 'child_alone' column since it has only 1 value therefore it's useless
    df = df.drop('child_alone', axis=1)
    # Drop all rows where 'related'== 2
    index = df[df['related'] == 2].index
    df.drop(index, inplace=True)

    print("There are total {} of rows after dropping rows where 'related'== 2".format(len(df)))
    return df

def save_data(df, database_filename):
    """
    Save the dataframe to a sqlite database
    :param df: disaster response dataframe
    :param database_filename: disaster response database filename
    :return: none
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('message', engine, index=False)

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
              'disaster_response.db')


if __name__ == '__main__':
    main()