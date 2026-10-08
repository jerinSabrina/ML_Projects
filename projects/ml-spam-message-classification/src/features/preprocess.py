import pandas as pd
from sklearn.model_selection import train_test_split

#Step 4 — Move the preprocessing code
#Your notebook does two important preprocessing things before model training:
#- It checks class balance and finds 3672 ham vs 1499 spam, so the dataset is imbalanced.
#- It downsamples the ham class to 1499 rows, combines both classes into 2998 rows, then uses a 70/30 train-test split.

def balance_data(df):
    df_ham = df[df["label"] == "ham"]
    df_spam = df[df["label"] == "spam"]

    df_ham = df_ham.sample(
        n=df_spam.shape[0],
        random_state=42
    )

    balanced_data = pd.concat(
        [df_ham, df_spam],
        ignore_index=True
    )

    return balanced_data


def split_data(data):
    X = data["text"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=0,
        shuffle=True
    )

    return X_train, X_test, y_train, y_test