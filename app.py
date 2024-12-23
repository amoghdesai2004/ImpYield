from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('your_dataset.csv')

# Verify the dataset is loaded correctly
print(df.head())

# Train the regression model for predicting rainfall
X_rainfall = df[['temperature', 'wind_speed']]
y_rainfall = df['rainfall']
rainfall_model = LinearRegression()
rainfall_model.fit(X_rainfall, y_rainfall)

# Train the regression model for predicting temperature
X_temperature = df[['rainfall', 'wind_speed']]
y_temperature = df['temperature']
temperature_model = LinearRegression()
temperature_model.fit(X_temperature, y_temperature)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_rainfall', methods=['GET', 'POST'])
def predict_rainfall():
    prediction = None
    error = None
    if request.method == 'POST':
        try:
            temperature = float(request.form['temperature'])
            wind_speed = float(request.form['wind_speed'])
            prediction = rainfall_model.predict([[temperature, wind_speed]])[0]
        except Exception as e:
            error = str(e)
    return render_template('predict_rainfall.html', prediction=prediction, error=error)

@app.route('/predict_temperature', methods=['GET', 'POST'])
def predict_temperature():
    prediction = None
    error = None
    if request.method == 'POST':
        try:
            rainfall = float(request.form['rainfall'])
            wind_speed = float(request.form['wind_speed'])
            prediction = temperature_model.predict([[rainfall, wind_speed]])[0]
        except Exception as e:
            error = str(e)
    return render_template('predict_temperature.html', prediction=prediction, error=error)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
