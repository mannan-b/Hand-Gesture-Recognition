from flask import Flask, request, jsonify
import xgboost as xgb
import pandas as pd

app = Flask(__name__)

# Load the XGBoost model
xgb_model = xgb.Booster()
xgb_model.load_model('path/to/your/model.json')

@app.route('/classify', methods=['POST'])
def classify_transactions():
    data = request.get_json()
    # Convert data to DataFrame
    df = pd.DataFrame(data)
    
    # Preprocess data if necessary
    # Make predictions with the model
    predictions = xgb_model.predict(df)
    
    # Convert predictions to category labels
    categories = [get_category(pred) for pred in predictions]
    
    # Return categories
    return jsonify(categories)

def get_category(prediction):
    # Logic to map prediction values to category labels
    # For example:
    category_map = {
        0: 'shopping',
        1: 'income',
        2: 'food',
        # Add more mappings as needed
    }
    return category_map[prediction]

if __name__ == '__main__':
    app.run(debug=True)