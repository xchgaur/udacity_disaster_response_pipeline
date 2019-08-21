import sys
import pandas as pd
import pickle
import re
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report



def load_data(database_filepath):
    """
    This will load the data from databse into dataframe and split them into features and target.
    args:
        database_filepath: path of the database file
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disaster_pipeline', engine)
    print(df.head())
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    This will case normalize, lemmatize, and tokenize text
    args:
        text: input message text
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_regex, text)
    for url in urls:
        text = text.replace(url, "urlplaceholder")
    tokens = word_tokenize(text)
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
        
    tokens = [lemmatizer.lemmatize(word).lower().strip() for word in tokens if word not in stop_words]

    return tokens


def build_model():
    """
    Build a model pipeline
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])
    
    params = pipeline.get_params()
    print(params)

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2), (2,2)),
        #'vect__max_df': (0.5, 1.0),
        #'clf__estimator__max_depth': (25,  100, None),
        #'clf__estimator__min_samples_split': (6, 10, 25, 50), 
        'clf__estimator__n_estimators': [10, 50],
    }

    cv = GridSearchCV (pipeline, param_grid= parameters)
    return cv
    
    


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    for pred, orig, column in zip(y_pred.transpose(), Y_test.values.transpose(), category_names):
        print(column)
        print(classification_report(orig, pred))
    overall_accuracy = (y_pred == Y_test).mean().mean()
    print("Overall accuracy is {}".format(overall_accuracy))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()

        print(model)
        
        
        print('Training model...')
        model.fit(X_train, Y_train)

        print(model.best_params_)
        
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