from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

def build_random_forest_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("rf_classifier",
         RandomForestClassifier(
             n_estimators=10,
             random_state=42
         ))
    ])


def build_svm_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("svm_classifier", SVC())
    ])



def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model