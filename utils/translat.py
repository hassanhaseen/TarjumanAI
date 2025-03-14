def translate(preprocessed_text, model):
    translated_text = model.predict(preprocessed_text)
    return translated_text