import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create a new Flask app
app = Flask(__name__)

# Load the trained model
# Make sure 'logistic_regression_model.pkl' is in the same directory as this script
try:
    model = pickle.load(open('logistic_regression_model.pkl', 'rb'))
except FileNotFoundError:
    print("Model file not found. Please ensure 'logistic_regression_model.pkl' is in the root directory.")
    model = None
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    model = None

@app.route('/')
def home():
    """
    Renders the main page of the web application.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the prediction request from the user.
    Takes form data, processes it, and returns the prediction.
    """
    if model is None:
        return render_template('index.html', prediction_text='Error: Model could not be loaded.')

    try:
        # Get all the input values from the form and convert them to float
        feature_values = [float(x) for x in request.form.values()]
        
        # Convert the list of features into a NumPy array for the model
        final_features = [np.array(feature_values)]
        
        # Make a prediction using the loaded model
        prediction = model.predict(final_features)

        # Determine the output message based on the prediction
        output = "The model predicts that the person has Heart Disease." if prediction[0] == 1 else "The model predicts that the person does not have Heart Disease."

        # Render the main page again, but this time with the prediction result
        return render_template('index.html', prediction_text=output)
    
    except ValueError:
        return render_template('index.html', prediction_text='Error: Please enter valid numerical values for all fields.')
    except Exception as e:
        return render_template('index.html', prediction_text=f'An error occurred during prediction: {e}')

if __name__ == "__main__":
    # Runs the Flask app
    app.run(debug=True)
