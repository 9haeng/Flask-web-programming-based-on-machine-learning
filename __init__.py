import os
import numpy as np
from flask import Flask, render_template, request, jsonify
import joblib

def create_app():
    app = Flask(__name__)

    @app.route('/', methods=['GET','POST'])
    def predict():
        if request.method == 'GET':
            return render_template('index2.html')

        if request.method == 'POST':
            import pandas as pd
            from category_encoders import OrdinalEncoder
            from xgboost import XGBRegressor
            import warnings
            warnings.simplefilter(action='ignore', category=FutureWarning)

            CSV_FILEPATH = os.path.join(os.getcwd(), 'nycairbnb.csv') 
            df = pd.read_csv(CSV_FILEPATH)
            target = 'price'
            X = df.drop(columns=['price','Unnamed: 0'])
            y = df[target]
            
            encoder = OrdinalEncoder()
            X_encoded = encoder.fit_transform(X)
         
            model = XGBRegressor(n_jobs=-1, n_estimators=200, max_depth=5, learning_rate=0.2, random_state=2)
            model.fit(X_encoded, y)

            neighbourhood_group = request.form.get('neighbourhood_group')
            room_type = request.form.get('room_type')
            Popularity = request.form.get('Popularity')
            Reviews = request.form.get('Reviews')
            required_nights = request.form.get('required_nights')

            collection = pd.DataFrame({"neighbourhood_group":[neighbourhood_group], "room_type":[room_type], "Popularity":[Popularity], "Reviews": [Reviews], "required_nights": [required_nights]}, index=[0])
                        
            X_pred = encoder.transform(collection)
            result = round(float(model.predict(X_pred)[0]),2)
            #result_true = round(result[0],2)
            return render_template('result.html', data = result)
    return app

if __name__ == "__main__":
  app = create_app()
  app.run(debug=True)
