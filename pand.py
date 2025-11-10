import pandas as pd
import matplotlib.pyplot as plt 

df = pd.DataFrame({
    "city": ["LA", "SD", "SF", None, None],
    "temperature": [71.1, 78.2, 66.5, None, None],
    "rain": [0, 5, 2, None, None]
})

# DataFrame before dropping the None values
print(df)

# Dropping None values
df = df.dropna()
print(df)

# Let's add our temperature in Celsius
df["temp_c"] = (df["temperature"] -32) * 5/9

print(df["temp_c"])

# Plotting our temperature
# plt.plot(df["temperature"])
# plt.title("Temperature in F")
# plt.show()