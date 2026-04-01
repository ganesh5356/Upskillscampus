import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

df = pd.read_csv("dataset/train.csv")

data = df["Vehicles"].values.reshape(-1,1)

scaler = MinMaxScaler()
data = scaler.fit_transform(data)

X=[]
y=[]

for i in range(10,len(data)):
    X.append(data[i-10:i])
    y.append(data[i])

X=np.array(X)
y=np.array(y)

model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(X.shape[1],1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer="adam",loss="mse")

model.fit(X,y,epochs=10)

print("Training complete")