from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template("home.html")

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, -1)  # Membentuk array 2D dengan 1 baris dan 10 kolom
    loaded_model = joblib.load('model.sav')
    result = loaded_model.predict(to_predict)[0]  # Ambil hasil prediksi
    weather_mapping = {
        0: 'Cloudy',
        1: 'Rainy',
        2: 'Snowy',
        3: 'Sunny'
    }
    return weather_mapping.get(result, 'Unknown')  # Mengembalikan kategori cuaca

@app.route('/', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        try:
            to_predict_list = list(map(float, to_predict_list))  # Konversi ke float
        except ValueError as e:
            return render_template("home.html", result="Input tidak valid, pastikan semua data benar.")
        result = ValuePredictor(to_predict_list)
        return render_template("home.html", result=result)
    return render_template("home.html")

if __name__ == '__main__':
    app.run(debug=True)
