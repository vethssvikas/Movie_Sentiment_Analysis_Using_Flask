import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import preprocess_kgptalkie as ps
df = pd.read_csv('imdb_reviews.txt', sep = '\t', header = None)
df.columns = ['reviews', 'sentiment']

df['reviews'] = df['reviews'].apply(lambda x: ps.cont_exp(x))
df['reviews'] = df['reviews'].apply(lambda x: ps.remove_special_chars(x))

df['reviews'] = df['reviews'].apply(lambda x: ps.remove_accented_chars(x))
df['reviews'] = df['reviews'].apply(lambda x: ps.remove_emails(x))

df['reviews'] = df['reviews'].apply(lambda x: ps.remove_urls(x))
df['reviews'] = df['reviews'].apply(lambda x: ps.make_base(x))

df['reviews'] = df['reviews'].apply(lambda x: str(x).lower())

X = df['reviews']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)

from sklearn.svm import LinearSVC

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC())
])

hyperparameters = {
    'tfidf__max_df': (0.5, 1.0),
    'tfidf__ngram_range': ((1,1), (1,2)),
    'tfidf__use_idf': (True, False),
    'tfidf__analyzer': ('word', 'char', 'char_wb'),
    'clf__C': (1,2,2.5,3)
}

clf = GridSearchCV(pipeline, hyperparameters, n_jobs=-1, cv = 5)

clf.fit(X_train, y_train)

text = ['good movie']

print(clf.predict(text))

import pickle as pkl

pkl.dump(clf, open('model.pkl', 'wb'))