from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer#, CountVectorizer
from sklearn.linear_model import SGDClassifier#
from sklearn.pipeline import Pipeline
from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np


def build_model():
    model = Pipeline([("tfidf_vectorizer", TfidfVectorizer(ngram_range=(1, 2))),('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=0.01, random_state=42,max_iter=5, tol=None))])
    sufficient = pd.read_csv("sufficient.csv") #unsere Vorverarbeiteten Daten
    sufficient.drop("Unnamed: 0", axis=1,inplace=True)
    sufficient= sufficient[["character", "tokens"]]
    y=np.array(sufficient.character.tolist())
    Xnew=sufficient.tokens.str.replace('[^\w\s]','')
    X_train,X_test, y_train,  y_test = train_test_split(Xnew, y,
                                                  stratify=y,
                                                  random_state=42,
                                                  test_size=0.1, shuffle=True)
    print("split complete")
    model.fit(X_train, y_train)
    print("sgd model fit complete")
    #model.pickle_clf()
    #model.pickle_vectorizer()
    #model.plot_roc(X_test, y_test)
    return model
if __name__ == "__main__":
    build_model()


app = Flask(__name__)
api = Api(app)

#model = Pipeline([("tfidf_vectorizer", TfidfVectorizer(ngram_range=(1, 2))),('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=0.01, random_state=42,max_iter=5, tol=None))])

#clf_path = 'lib/models/SentimentClassifier.pkl'
#with open(clf_path, 'rb') as f:
#    model.clf = pickle.load(f)

#vec_path = 'lib/models/TFIDFVectorizer.pkl'
#with open(vec_path, 'rb') as f:
#    model.vectorizer = pickle.load(f)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')



class PredictSentiment(Resource):
    def get(self):

        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']
        model_ergebnis=build_model()
        # vectorize the user's query and make a prediction
        #uq_vectorized = model.vectorizer_transform(np.array([user_query]))
        prediction = model_ergebnis.predict([user_query])
        #pred_proba = model.predict_proba(uq_vectorized)

        # Output either 'Negative' or 'Positive' along with the score
        #if prediction == 0:
        #    pred_text = 'Negative'
        #else:
        #    pred_text = 'Positive'

        # round the predict proba value and set to new variable
        #confidence = round(pred_proba[0], 3)

        # create JSON object
        output = {'prediction': prediction[0]}#, 'confidence': confidence}

        return output


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictSentiment, '/')


if __name__ == '__main__':
    app.run(debug=True)
