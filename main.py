from flask import Flask, render_template, request
from predict import make_prediction_en, make_prediction_ru
from lingua import Language, LanguageDetectorBuilder


app = Flask(__name__)
languages = [Language.ENGLISH, Language.RUSSIAN]
detector = LanguageDetectorBuilder.from_languages(*languages).build()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']

        input_lang = detector.detect_language_of(message)

        if input_lang == Language.ENGLISH:
            pred = make_prediction_en(message)
            return render_template('index.html', prediction=pred)
        elif input_lang == Language.RUSSIAN:
            pred = make_prediction_ru(message)
            return render_template('index.html', prediction=pred)
    else:
        return render_template('index.html', prediction="Something went wrong")


if __name__ == '__main__':
    app.run(debug=True)
