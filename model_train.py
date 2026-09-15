import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score   
df = pd.read_csv("/Users/aryanguptasamanagmail.com/Desktop/ML OPS/House Price Prediction Dataset.csv")
print(type(df))
df.info()
df.isna().sum()
df.drop("Id", axis=1, inplace=True)
le = LabelEncoder()
df["Location"] = le.fit_transform(df["Location"])
df["Condition"] = le.fit_transform(df["Condition"])
df["Garage"] = le.fit_transform(df["Garage"])
print(le.classes_)
x = df.drop("Price", axis=1)
y = df["Price"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print(r2)
mse = mean_squared_error(y_test, y_pred)
print(mse)
