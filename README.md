# Disaster Response Pipeline Project
https://github.com/msplittgerber/disaster_response_pipeline_project
### Description:

This repository showcases a project completed as part of the Udacity Data Scientist Nanodegree program. The project focuses on building an end-to-end data pipeline for analyzing and categorizing disaster-related messages. The project provides code, datasets, and documentation to support the development of an end-to-end data pipeline. By leveraging natural language processing techniques and machine learning models, the pipeline enables efficient processing, classification, and response to disaster-related messages.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
