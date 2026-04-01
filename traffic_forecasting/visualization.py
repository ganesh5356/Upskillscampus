import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/train.csv")

df["DateTime"] = pd.to_datetime(df["DateTime"])

df.set_index("DateTime", inplace=True)

plt.figure(figsize=(12,5))
plt.plot(df["Vehicles"])
plt.title("Traffic Pattern")
plt.show()