from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

model = joblib.load('../Models/heart_model.pkl')

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/heartApi', methods=["POST"])
def Api():
    try:
        age = request.args.get('age')
        sex = request.args.get('sex')
        cp = request.args.get('cp')
        trestbps = request.args.get('trestbps')
        chol = request.args.get('chol')
        fbs = request.args.get('fbs')
        restecg = request.args.get('restecg')
        thalach = request.args.get('thalach')
        exang = request.args.get('exang')
        oldpeak = request.args.get('oldpeak')
        slope = request.args.get('slope')
        ca = request.args.get('ca')
        thal = request.args.get('thal')

        input_data = [float(age), float(sex), float(cp), float(trestbps), float(chol), float(fbs), float(restecg),
                      float(thalach), float(exang), float(oldpeak), float(slope), float(ca), float(thal)]
        data_array = np.asarray(input_data)
        reshaped_data = data_array.reshape(1, -1)
        prediction_num = model.predict(reshaped_data)

        return jsonify({
            'prediction': int(prediction_num[0])
        }
        )
    except Exception as e:
        return jsonify({
            'error': str(e)
        }
        )


@app.route('/predict', methods=['POST'])
def predict():
    age = request.form['age']
    sex = request.form['sex']
    cp = request.form['cp']
    trestbps = request.form['trestbps']
    chol = request.form['chol']
    fbs = request.form['fbs']
    restecg = request.form['restecg']
    thalach = request.form['thalach']
    exang = request.form['exang']
    oldpeak = request.form['oldpeak']
    slope = request.form['slope']
    ca = request.form['ca']
    thal = request.form['thal']

    input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    data_array = np.asarray(input_data)
    reshaped_data = data_array.reshape(1, -1)
    prediction_num = model.predict(reshaped_data)
    if prediction_num == 0:
        prediction = 'You are not a heart patient'
    else:
        prediction = 'You may be a heart patient'

    return render_template('results.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True, port=3000)
