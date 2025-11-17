from sklearn.datasets import load_iris # Gives us an sample dataset
from sklearn.neighbors import KNeighborsClassifier # Our KNN model
from sklearn.metrics import accuracy_score, classification_report # Helps us evaluate the model
from sklearn.model_selection import train_test_split # To split our data into training and testing sets like before

# Load the iris dataset
X, y = load_iris(return_X_y=True)

# Split the dataset into training and testing sets
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the model to the training data and make predictions on the test data
knn.fit(X_tr, y_tr)

# Make predictions
y_pred = knn.predict(X_te)

# Evaluate the model's performance
print("Accuracy:", accuracy_score(y_te, y_pred))
# Detailed classification report about precision, recall, f1-score for each class
print(classification_report(y_te, y_pred))
