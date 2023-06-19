from flask import Flask, request, render_template, redirect
import joblib
#import pickle
import json
import pandas as pd
# 콘다 가상환경 python=3.10에서 작업

app = Flask(__name__)

"""
with open('./models/apart_sale_xgb_model_P3_10.pkl', 'rb') as file:
    reg_model = pickle.load(file)
"""

reg_model = joblib.load('./models/apart_sale_xgb_model.pkl')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods = ['POST'])
def predict():
    data = request.get_json()
    df_input = pd.DataFrame(data)
    prediction = reg_model.predict(df_input)

    return json.dumps(list(prediction.astype(float)))

@app.route('/predict_web', methods=['GET','POST'])
def predict_web():
    # HTML 폼에서 전송된 데이터 추출
    year = int(request.form['measure_year'])
    month = int(request.form['measure_month'])
    #day = int(request.form['measure_day'])
    legal_dong = request.form['legal_dong']
    apartment_name = request.form['apartment_name']
    exclusive_area = float(request.form['exclusive_area'])
    floor = int(request.form['floor'])

    # 데이터 가공 및 예측
    data = {
        'Measure_year': [year],
        'Measure_month': [month],
        'Legal_dong': [legal_dong],
        'Apartment_name': [apartment_name],
        'Exclusive_area': [exclusive_area],
        'Floor': [floor]
    }
    df = pd.DataFrame(data)
    prediction = reg_model.predict(df)
    # 예측 결과를 HTML로 전달
    return render_template('index.html', prediction=prediction)

# 대시보드 보기
@app.route('/dashboard', methods=['GET','POST'])
def redirect_to_dashboard():
    return redirect("http://127.0.0.1:3000/dashboard/3-project4", code=302)

if __name__ == '__main__':
    app.run(debug=True)
