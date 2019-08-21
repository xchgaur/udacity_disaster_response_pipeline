import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    This will load the data from two csv files
    and will merge them into a dataframe
    args:
        messages_filepath: path of disaster messages csv file
        categories_filepath: path of disaster categpries csv file
    return:
        dataframe with merged values
    """
    messages = pd.read_csv(messages_filepath)
    messages = messages.drop_duplicates()
    print(messages.head())

    categories = pd.read_csv(categories_filepath)
    categories = categories.drop_duplicates()
    print(categories.head())

    df = messages.merge(categories, on ='id', how='left')
    print(df.tail())
    print(df.shape)
    return df


def clean_data(df):
    """
    This will clean and transform the data so that teh catgeories
    are well separated into ditinct columns with corresponding values
    args:
        df: input dataframe
    return:
        df: transformed dataframe
    """
    labels = df['categories'].str.split(';', expand=True)
    labels.columns = labels.iloc[1].apply(lambda x: x.split('-')[0])
    for col in labels.columns:
        labels[col] = labels[col].apply(lambda x: int(x.split('-')[1]))
    print(labels.columns)
    print(labels.head())
    df = df.drop('categories', axis=1)
    df = pd.concat([df,labels], axis = 1)
    df = df.drop_duplicates(subset = ['message'])
    print(df.head())
    return df


def save_data(df, database_filename):
    """
    This will push the dataframe data into database
    args:
        df: input dataframe
        database_filename: sqlite database file
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('disaster_pipeline', engine, index=False)


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
