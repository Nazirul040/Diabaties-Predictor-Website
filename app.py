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
    # Ensure model and scaler were loaded correctly
    if model is None or scaler is None:
        return jsonify({'error': 'Model or scaler not loaded properly. Check server logs.'}), 500

    try:
        # 1. Get data from the POST request's JSON body
        data = request.get_json()
        
        # 2. Extract features in the correct order
        # This order MUST match the order of features your model was trained on.
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
        
        # Convert to a numpy array for the model
        input_data = np.array([features])

        # 3. Scale the input data using the loaded scaler
        input_scaled = scaler.transform(input_data)

        # 4. Make a prediction
        prediction_raw = model.predict(input_scaled)
        
        # Convert prediction to a standard Python integer
        prediction = int(prediction_raw[0])

        # 5. Return the result as a JSON object
        # The JavaScript in index.html is expecting this format.
        return jsonify({'prediction': prediction})

    except KeyError as e:
        # This error occurs if a key is missing in the incoming JSON
        return jsonify({'error': f'Missing data for: {str(e)}'}), 400
    except Exception as e:
        # Handle other potential errors (e.g., data conversion issues)
        print(f"An error occurred: {e}") # Log the error for debugging
        return jsonify({'error': 'An error occurred during prediction.'}), 500


if __name__ == "__main__":
    app.run(debug=True)