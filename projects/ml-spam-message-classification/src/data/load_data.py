from pathlib import Path

import pandas as pd


def load_data():
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "spam_ham_dataset.csv"

    df = pd.read_csv(data_path)

    return df


if __name__ == "__main__":
    df = load_data()

    print(df.head())
    print("\nShape:", df.shape)
    print("\nColumns:", df.columns.tolist())