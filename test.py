from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# load data
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# build pipeline
pipe = Pipeline([
    ("scaler", StandardScaler()),   # Step 1: scale features
    ("knn", KNeighborsClassifier(n_neighbors=5))  # Step 2: model
])

# fit pipeline
pipe.fit(X_train, y_train)

# evaluate
print("Train accuracy:", pipe.score(X_train, y_train))
print("Test accuracy:", pipe.score(X_test, y_test))
