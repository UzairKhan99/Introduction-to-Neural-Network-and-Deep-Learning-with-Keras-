import pandas as pd
import numpy as np
import keras
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input


# Load dataset
def regression_model():
    model = Sequential()
    model.add(Input(shape=(n_cols,)))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


concrete_data = pd.read_csv("concrete_data.csv")

predictors = concrete_data.drop("Strength", axis=1)
scaler = StandardScaler()
predictors_norm = (predictors - predictors.mean()) / predictors.std()

target = concrete_data["Strength"]
n_cols = predictors_norm.shape[1]
model = regression_model()
model.fit(predictions_norm, target, validation_spilit=0.3, epochs=100,verbose2)

print(predictors_norm.head())
