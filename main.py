from src.data.load_data import load_data
from src.features.preprocess import balance_data, split_data
from src.models.train import (
    build_random_forest_model,
    build_svm_model,
    train_model,
)
from src.models.evaluate import evaluate_model
from src.models.predict import predict_message


df = load_data()

print("Original dataset:")
print(df.shape)

print("\nOriginal class distribution:")
print(df["label"].value_counts())


data = balance_data(df)

print("\nBalanced dataset:")
print(data.shape)

print("\nBalanced class distribution:")
print(data["label"].value_counts())


X_train, X_test, y_train, y_test = split_data(data)

print("\nTraining shapes:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting shapes:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


print("\nTraining Random Forest...")

rf_model = build_random_forest_model()
rf_model = train_model(
    rf_model,
    X_train,
    y_train
)

print("Random Forest training completed.")

print("\nTraining SVM...")

svm_model = build_svm_model()
svm_model = train_model(
    svm_model,
    X_train,
    y_train
)

print("SVM training completed.")

rf_accuracy = evaluate_model(
    rf_model,
    X_test,
    y_test,
    "Random Forest"
)


svm_accuracy = evaluate_model(
    svm_model,
    X_test,
    y_test,
    "SVM"
)

print("\n===== Model Comparison =====")

print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
print(f"SVM Accuracy: {svm_accuracy:.4f}")


test1 = (
    "Hi! I hope this mail finds you well. "
    "I just need the book you borrowed. Regards Sabrina."
)

test2 = (
    "Congratulations! You won a lottery ticket worth $1 million! "
    "To claim call on 22222."
)


print("\n===== Custom Message Predictions =====")

print("\nMessage 1:")
print(test1)

print("Random Forest:", predict_message(rf_model, test1))
print("SVM:", predict_message(svm_model, test1))


print("\nMessage 2:")
print(test2)

print("Random Forest:", predict_message(rf_model, test2))
print("SVM:", predict_message(svm_model, test2))