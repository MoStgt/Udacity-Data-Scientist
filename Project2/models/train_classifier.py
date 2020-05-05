import sys
import pandas as pd
from sqlalchemy import create_engine
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

def load_data(database_filepath):
    """
    Load data
    
    Args:
        database_filepath: path to SQLite database
        
    Return:
        X: Messages as DataFrame
        y: One-hot encoded categories as DataFrame
        category_names: Names of categories as list
    
    """
    
    # load data from database
    engine = create_engine('sqlite:///./'+database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = y.columns
    
    return X, y, category_names


def tokenize(text):
    """
    Tokenize text
    
    Args:
        text: text messages
        
    Return:
        clean_tokens: tokenized text and clean for ML
    """
    # Define url pattern
    url_re = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Detect and replace urls
    detected_urls = re.findall(url_re, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # tokenize sentences
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    # save cleaned tokens
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    
    # remove stopwords
    STOPWORDS = list(set(stopwords.words('english')))
    clean_tokens = [token for token in clean_tokens if token not in STOPWORDS]
    
    return clean_tokens

def build_model():
    """
    Build NLP pipeline - count words, tf-idf, multiple output classifier
    GridSearch the best parameters
    
    Args:
        None
        
    Returns:
        cv: cross validated classifier as object
    """
    
    pipeline = Pipeline([
        ('vec', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())) 
    ])
    
    # grid search
    parameters = {
                #'tfidf__smooth_idf':[True, False],
                #'clf__estimator__estimator__C': [2, 5]
                #'clf__estimator__max_depth': [2, 5,],
                'clf__estimator__min_samples_split': [2, 3],
                #'clf__estimator__n_estimators':[100, 200],
                #'clf__estimator__criterion': ['entropy']
    }

    cv = GridSearchCV(estimator=pipeline, param_grid = parameters, cv = 5)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model (f1 score, precision and recall)
    
    Args:
        model: model to evaluate
        X_test: X_test dataframe
        Y_test: y_test dataframe
        category_names: category list
     
    Returns:
        performances as dataframe
    """
    # predict on the X_test
    y_pred = model.predict(X_test)
    
    # build classification report on every column
    performances = []
    for i in range(len(category_names)):
        performances.append([f1_score(Y_test.iloc[:, i].values, y_pred[:, i], average='micro'),
                             precision_score(Y_test.iloc[:, i].values, y_pred[:, i], average='micro'),
                             recall_score(Y_test.iloc[:, i].values, y_pred[:, i], average='micro')])
    # build dataframe
    performances = pd.DataFrame(performances, columns=['f1 score', 'precision', 'recall'],
                                index = category_names)   
    return performances
    


def save_model(model, model_filepath):
    """
    Save model as pickle
    
    Args:
        model: model as object
        model_filepath: path to save pickle file
    """
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))
    pass

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