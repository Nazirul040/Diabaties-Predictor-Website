from flask import Flask, render_template, request, jsonify
import pickle,numpy as np

app = Flask(__name__)

with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def hello_world():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model or scaler not loaded properly. Check server logs.'}), 500

    try:
        features = [
            float(data['pregnancies']),
            float(data['glucose']),
            float(data['bloodpressure']),
            float(data['skinthickness']),
            float(data['insulin']),
            float(data['bmi']),
            float(data['dpf']),
            float(data['age'])
        ]
        input_data = np.array([features])
        input_scaled = scaler.transform(input_data)

        prediction_raw = model.predict(input_scaled)
        
        prediction = int(prediction_raw[0])

   
        return jsonify({'prediction': prediction})

    except KeyError as e:
      
        return jsonify({'error': f'Missing data for: {str(e)}'}), 400
    except Exception as e:
      
        print(f"An error occurred: {e}") # Log the error for debugging
        return jsonify({'error': 'An error occurred during prediction.'}), 500


if __name__ == "__main__":
    app.run(debug=True)