from flask import Flask, request, render_template, url_for
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    output = prediction[0]
    to_text = {0: 'No intrusion detected', 1: 'Intrusion detected'}

    return render_template('index.html',
                           prediction_text='Result expected: {} : {}'.format(output, to_text[output]))


if __name__ == "__main__":
    app.run(debug=True)
