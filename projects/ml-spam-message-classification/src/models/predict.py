def predict_message(model, message):
    prediction = model.predict([message])

    return prediction[0]