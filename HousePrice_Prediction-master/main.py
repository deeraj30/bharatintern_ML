from flask import Flask, render_template, request, jsonify  # type: ignore
import pandas as pd  # type: ignore
import pickle

app = Flask(__name__)

# Load the dataset and the pre-trained model
data = pd.read_csv('final_dataset.csv')
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))

@app.route('/')
def index():
    # Extract unique values for dropdown menus
    bedrooms = sorted(data['beds'].unique())
    bathrooms = sorted(data['baths'].unique())
    sizes = sorted(data['size'].unique())
    zip_codes = sorted(data['zip_code'].unique())

    return render_template('index.html', bedrooms=bedrooms, bathrooms=bathrooms, sizes=sizes, zip_codes=zip_codes)

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    bedrooms = request.form.get('beds')
    bathrooms = request.form.get('baths')
    size = request.form.get('size')
    zipcode = request.form.get('zip_code')

    # Create a DataFrame with the input data
    input_data = pd.DataFrame([[bedrooms, bathrooms, size, zipcode]],
                               columns=['beds', 'baths', 'size', 'zip_code'])

    print("Input Data:")
    print(input_data)

    # Convert input data to appropriate numeric types
    input_data['beds'] = pd.to_numeric(input_data['beds'], errors='coerce')
    input_data['baths'] = pd.to_numeric(input_data['baths'], errors='coerce')
    input_data['size'] = pd.to_numeric(input_data['size'], errors='coerce')
    input_data['zip_code'] = pd.to_numeric(input_data['zip_code'], errors='coerce')

    # Handle unknown categories in the input data
    for column in input_data.columns:
        unknown_categories = set(input_data[column]) - set(data[column].unique())
        if unknown_categories:
            print(f"Unknown categories in {column}: {unknown_categories}")
            input_data[column] = input_data[column].replace(unknown_categories, data[column].mode()[0])

    print("Processed Input Data:")
    print(input_data)

    # Predict the price
    prediction = pipe.predict(input_data)[0]

    return jsonify(price=str(prediction))  # Return as JSON for better integration

if __name__ == "__main__":
    app.run(debug=True, port=5000)
