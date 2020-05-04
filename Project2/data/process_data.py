import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories from csv file
    
    Args:
        messages_filepath: path to messages.csv
        categories_filepath: path to categories.csv
    
    Returns:
        df_merged: merged messages and categories as DataFrame
    """
    # Load messages and categories
    messages=pd.read_csv(messages_filepath)
    categories=pd.read_csv(categories_filepath)
    
    # Merge messages and categories
    df_merged = pd.merge(messages, categories, left_on='id', right_on='id', how='outer')
    return df_merged


def clean_data(df):
    """
    Structure merged dataframe in different categories and remove duplicates
    
    Args:
        df: Merged dataframe of messages and categories
    
    Returns:
        df: Structured and categorised dataframe
    
    """
    # Split column categories by semicolon to multiple columns
    categories = df.categories.str.split(';',expand=True)
    
    # Rename columns
    row = categories.loc[0]
    category_colnames = row.apply(lambda x: x[:-2]).values.tolist()
    categories.columns = category_colnames
    
     # set each value to be the last character of the string
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df=pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(subset='id', inplace=True)
    
    return df

def save_data(df, database_filename):
    """
    Save preprocessed datafram into a sqlite database
    
    Args:
        df: Preprocessed dataframe
        database_filename: name of sqlite database
        
    Returns:
        none
    """
    # Save the clean dataset into an sqlite database
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('DisasterResponse', engine, index=False)
    
    
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