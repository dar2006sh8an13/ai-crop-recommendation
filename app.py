from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
mx = pickle.load(open('minmaxscaler.pkl','rb'))


def _scalers_look_degenerate(minmax_scaler: MinMaxScaler, standard_scaler: StandardScaler) -> bool:
    try:
        if hasattr(minmax_scaler, "data_range_") and np.allclose(minmax_scaler.data_range_, 0):
            return True
        if hasattr(standard_scaler, "var_") and np.allclose(standard_scaler.var_, 0):
            return True
    except Exception:
        return True
    return False


def _rebuild_scalers_from_csv() -> tuple[MinMaxScaler, StandardScaler]:
    df = pd.read_csv('Crop_recommendation.csv')
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].to_numpy(dtype=float)

    # Match the notebook split: test_size=0.2, random_state=42
    X_train, _X_test = train_test_split(X, test_size=0.2, random_state=42)

    rebuilt_mx = MinMaxScaler()
    X_train_mx = rebuilt_mx.fit_transform(X_train)

    rebuilt_sc = StandardScaler()
    rebuilt_sc.fit(X_train_mx)

    return rebuilt_mx, rebuilt_sc


if _scalers_look_degenerate(mx, sc):
    mx, sc = _rebuild_scalers_from_csv()
    # Persist fixed scalers so next run is fast.
    pickle.dump(mx, open('minmaxscaler.pkl', 'wb'))
    pickle.dump(sc, open('standscaler.pkl', 'wb'))


app = Flask(__name__, template_folder="templetes")

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    N = float(request.form['Nitrogen'])
    P = float(request.form['Phosporus'])
    K = float(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['pH'])
    rainfall = float(request.form['Rainfall'])

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list, dtype=float).reshape(1, -1)

    mx_features = mx.transform(single_pred)
    sc_mx_features = sc.transform(mx_features)
    prediction = model.predict(sc_mx_features)

    # Most models in this project predict string labels like "rice", "papaya", etc.
    # Older versions used numeric class ids; support both.
    pred0 = prediction[0]

    crop_id_to_name = {
        1: "Rice",
        2: "Maize",
        3: "Jute",
        4: "Cotton",
        5: "Coconut",
        6: "Papaya",
        7: "Orange",
        8: "Apple",
        9: "Muskmelon",
        10: "Watermelon",
        11: "Grapes",
        12: "Mango",
        13: "Banana",
        14: "Pomegranate",
        15: "Lentil",
        16: "Blackgram",
        17: "Mungbean",
        18: "Mothbean",
        19: "Pigeonpeas",
        20: "Kidney Beans",
        21: "Chickpea",
        22: "Coffee",
    }

    crop_label_to_name = {
        "rice": "Rice",
        "maize": "Maize",
        "jute": "Jute",
        "cotton": "Cotton",
        "coconut": "Coconut",
        "papaya": "Papaya",
        "orange": "Orange",
        "apple": "Apple",
        "muskmelon": "Muskmelon",
        "watermelon": "Watermelon",
        "grapes": "Grapes",
        "mango": "Mango",
        "banana": "Banana",
        "pomegranate": "Pomegranate",
        "lentil": "Lentil",
        "blackgram": "Blackgram",
        "mungbean": "Mungbean",
        "mothbean": "Mothbean",
        "pigeonpeas": "Pigeonpeas",
        "kidneybeans": "Kidney Beans",
        "chickpea": "Chickpea",
        "coffee": "Coffee",
    }

    crop_name: str | None = None
    if isinstance(pred0, (int, np.integer)):
        crop_name = crop_id_to_name.get(int(pred0))
    else:
        crop_name = crop_label_to_name.get(str(pred0).strip().lower())

    if crop_name:
        result = f"{crop_name} is the best crop to be cultivated right there"
    else:
        result = f"Sorry, we could not determine the best crop to be cultivated. Model output: {pred0}"
    return render_template('index.html',result = result)


if __name__ == "__main__":
    app.run(debug=True)