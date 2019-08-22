# udacity_disaster_response_pipeline

## Overview
This project will develop a web app that can handle and classify the messages that come from various sources during a disaster situation.The data has been provided by FigureEight and it contains real messages that were sent during disaster events.
This project includes two pipelines to process data and build ML model.It also includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

## Files:

process_data.py : ETL script to clean data into proper format by splitting up categories and making new columns for each as target variables.

train_classifier.py : Script to tokenize messages from clean data and create new columns through feature engineering. The data with new features are trained with a ML pipeline and pickled.

run.py : Main file to run Flask app that classifies messages based on the model and shows data visualizations.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


