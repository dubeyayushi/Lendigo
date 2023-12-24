import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    feature_order = ['Income', 'Age', 'Experience', 'Married/Single', 'House_Ownership', 'Car_Ownership', 'CURRENT_JOB_YRS', 'CURRENT_HOUSE_YRS']
    int_features = []
    for feature in feature_order:
        value = request.form.get(feature, 0)
        if value.isdigit() or value == 'on':
            # Replace 'on' with 0 for non-numeric fields
            int_features.append(int(value) if value.isdigit() else 0)

    final_features = [np.array(int_features)]
    print(final_features)
    prediction = model.predict(final_features)
    print(prediction)
    output = round(prediction[0], 2)

    if (output==0):
        return render_template('prediction-approved.html', prediction_text='Your loan is likely to be APPROVED!')
    
    else:
        return render_template('prediction-rejected.html', prediction_text='Your loan might not be approved')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    For direct API calls through request
    '''    
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)