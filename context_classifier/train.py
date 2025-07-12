import numpy as np
import pandas as pd
import joblib
import os, json, gc, re, random
import torch, transformers, tokenizers
from itertools import chain
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.calibration import CalibratedClassifierCV


import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

data_file = 'context_classifier/input/arxiv-metadata-oai-snapshot.json'

def get_metadata():
    with open(data_file, 'r') as f:
        for line in f:
            yield line

def feature_importance(pipeline):
    '''
    Extract feature importances from pipeline.
    Since I am using CalibratedClassifierCV I will average the coefficients over calibrated classifiers.
    '''
    # average coefficients over all calibrated classifiers
    coef_avg = 0
    classifiers = pipeline[1].estimators_[0].calibrated_classifiers_
    for i in classifiers:
        coef_avg = coef_avg + i.base_estimator.coef_
    coef_avg  = (coef_avg/len(classifiers)).tolist()[0]
    # get feature names from tf-idf vectorizer
    features = pipeline[0].get_feature_names()
    print(len(features))
    # get 10 most important features
    top_f = pd.DataFrame(list(zip(features,coef_avg)), columns = ['token','coef']) \
        .nlargest(10,'coef').to_dict(orient = 'records')
    return top_f


metadata = get_metadata()

titles = []
abstracts = []
categories = []

for paper in tqdm(metadata):
    paper_dict = json.loads(paper)
    category = paper_dict.get('categories')
    try:
        context = None
        if 'cs.' in category:
            context = 'cs'
        elif 'q-bio.' in category:
            context = 'bio'
        elif 'q-fin.' in category:
            context = 'fin'
        if context:
            titles.append(paper_dict.get('title'))
            abstracts.append(paper_dict.get('abstract'))
            categories.append([context])
    except:
        pass

papers = pd.DataFrame({
    'title': titles,
    'abstract': abstracts,
    'categories': categories,
})

size = len(titles)

papers['abstract'] = papers['abstract'].apply(lambda x: x.replace("\n",""))
papers['abstract'] = papers['abstract'].apply(lambda x: x.strip())
papers['text'] = papers['title'] + '. ' + papers['abstract']

sample_size = papers['categories'].value_counts()[-1]
papers['categories'] = list(chain.from_iterable(papers['categories']))
sample = pd.DataFrame(papers.groupby('categories').apply(lambda x: x.sample(sample_size)))
sample['categories'] = sample['categories'].apply(lambda x: [x])
sample.reset_index(inplace=True, drop=True)

# convert general category into label columns
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(sample.categories)

# concatenate with the abstracts
df = pd.concat([sample[['abstract','title']], pd.DataFrame(labels)], axis=1)
df.columns = ['abstract','title'] + list(mlb.classes_)
categories = df.columns[2:]

# split into train and test
train, test = train_test_split(df, random_state=76, test_size=0.15, shuffle=True)

X_train = train.abstract
X_test = test.abstract

# define the pipeline
classifier = CalibratedClassifierCV(LinearSVC())

# for each category train the model and get accuracy
models = {}
preds = {}
features = {}
for category in categories:
    # give pipelines unique names. important!
    SVC_pipeline = Pipeline([
                (f'tfidf_{category}', TfidfVectorizer(stop_words=stop_words)),
                (f'clf_{category}', OneVsRestClassifier(classifier, n_jobs=1)),
            ])
    print('... Processing {}'.format(category))
    # train the model using X_dtm & y
    SVC_pipeline.fit(X_train, train[category])
    models[category] = SVC_pipeline
    # compute the testing accuracy
    prediction = SVC_pipeline.predict(X_test)
    preds[category] = prediction
    accuracy = accuracy_score(test[category], prediction)
    print("Accuracy: ", accuracy)
    # get most predictive features
    features[category] = feature_importance(SVC_pipeline)
    # 10 most important features by category
    features_df = pd.DataFrame(features)
    features_df.apply(lambda x: [d['token'] for d in x], axis=0)
    features_df.to_csv("most_predictive.csv")

    joblib.dump(SVC_pipeline, 'context_classifier/models/{}.joblib'.format(category))
