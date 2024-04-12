from flask import Flask, render_template, request
import joblib
import numpy as np
import cv2

app = Flask(__name__)

# Muat kembali model dari file
loaded_model = joblib.load('knnmodel1.pkl')

def predict_label(image_path):
    # Baca gambar menggunakan OpenCV dan lakukan konversi warna dari BGR ke RGB
    new_image = cv2.imread(image_path)
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)

    # Normalisasi gambar baru (jika diperlukan) atau lakukan preprocessing tambahan

    # Ekstraksi rata-rata komponen RGB dari gambar
    avg_red = np.mean(new_image[:, :, 0])
    avg_green = np.mean(new_image[:, :, 1])
    avg_blue = np.mean(new_image[:, :, 2])

    # Normalisasi fitur-fitur RGB menggunakan min-max normalization
    min_values = np.array([0, 0, 0])  # Nilai minimum untuk setiap fitur (Red, Green, Blue)
    max_values = np.array([255, 255, 255])  # Nilai maksimum untuk setiap fitur (Red, Green, Blue)
    normalized_rgb_features = (np.array([avg_red, avg_green, avg_blue]) - min_values) / (max_values - min_values)

    # Lakukan prediksi pada gambar baru
    prediction_index = loaded_model.predict([normalized_rgb_features])

    # Tampilkan hasil prediksi
    if prediction_index.ndim == 1:  # Memastikan prediction_index adalah array 1 dimensi
        if prediction_index == 0:
            return 'unripe'
        elif prediction_index == 1:
            return 'halfripe'
        elif prediction_index == 2:
            return 'ripe'
        elif prediction_index == 3:
            return 'veryripe'
        elif prediction_index == 4:
            return 'rotten'
        else:
            return 'Tidak ada prediksi yang valid'
    else:
        return 'Tidak ada prediksi yang valid'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Simpan file yang diunggah oleh pengguna
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            image_path = 'static/' + uploaded_file.filename
            uploaded_file.save(image_path)
            # Lakukan prediksi label
            predicted_label = predict_label(image_path)
            return render_template('result.html', image_path=image_path, predicted_label=predicted_label)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port= 5000)