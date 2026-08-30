import joblib 
import pandas as pd

# Load the trained model
model = joblib.load("telecom_tower_model.pkl")

# Create new input data with the correct column names
new_data = pd.DataFrame({ 
    "temperature_c": [55], 
    "battery_health": [65], 
    "Power_Consumption_W": [2962], 
    "signal_drop_rate": [0.5], 
    "Fan_Speed_RPM": [2838], 
    "Humidity_Percent": [52], 
    "Traffic_Load": [2612], 
    "Tower_Age_Years": [4] 
})

# Make prediction
prediction = model.predict(new_data) 

if prediction[0] == 1: 
    print("Hardware Failure Predicted") 
else: 
    print("Tower is Healthy")
