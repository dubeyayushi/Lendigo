import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url, json={'Income': 80000, 'Age': 23, 'Experience': 3, 'Married/Single': 1, 'House_Ownership': 1, 'Car_Ownership': 0, 'CURRENT_JOB_YRS': 2, 'CURRENT_HOUSE_YRS': 2})

print(r.json())