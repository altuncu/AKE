import joblib

models = {}

models['cs'] = joblib.load('context_classifier/models/cs.joblib')
models['bio'] = joblib.load('context_classifier/models/bio.joblib')
models['fin'] = joblib.load('context_classifier/models/fin.joblib')

def predict_tags(X):
    '''
    Predict tags for a given abstract.

    Args:
      - X (list): an iterable with text.
      - labels (pandas.Dataframe): label indicators for an abstract
    '''
    preds = []
    if type(X) is str: # convert into iterable if string
        X = [X]

    # get prediction from each model
    for c in models.keys():
        preds.append(models[c].predict(X))

    return [k for k,v in zip(list(models.keys()),preds) if v[0] > 0]
