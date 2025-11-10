import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# This portion is just for generating the data to work with
# Don't worry about understanding this part 
rng = np.random.default_rng(0)
hours = rng.uniform(0, 10, size=200)
noise = rng.normal(0, 5, size=200)
score = 5 + 8*hours + noise

# Let's setup what our X and y are
X = hours.reshape(-1, 1)
y = score

# Now let's split our data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now let's create our Linear Regression model and fit it to our training data
model = LinearRegression().fit(X_train, y_train)

# Let's make predictions on our test set
predictions = model.predict(X_test)

# Finally, let's check out how accurate our model was by comparing our test data 
print("MSE:", mean_squared_error(y_test, predictions))

# This is just to visualize our data and the model's predictions
# You can uncomment the lines below to see what the plot looks like
# plt.scatter(X_test[:,0], y_test, label="actual")
# plt.plot(sorted(X_test[:,0]), sorted(predictions), label="pred")
# plt.legend(); plt.show()