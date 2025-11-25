from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

# for k in [1, 3, 5, 10, 105]:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     knn.fit(X_tr, y_tr)
#     print("k=",k,"train:",knn.score(X_tr,y_tr),"test:",knn.score(X_te,y_te))

results = []

for k in range(1, 105):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_tr, y_tr)
    results.append((k, knn.score(X_tr,y_tr), knn.score(X_te,y_te)))

# Print only a few interesting ones
for k, train, test in results:
    if k in [1, 2, 3, 5, 10, 20, 50, 100]:
        print(f"k={k:3d} | train={train:.3f} | test={test:.3f}")

# WE can visual the results by:
import matplotlib.pyplot as plt

ks = [r[0] for r in results]
train_scores = [r[1] for r in results]
test_scores = [r[2] for r in results]

plt.plot(ks, train_scores, label="Train Accuracy")
plt.plot(ks, test_scores, label="Test Accuracy")
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.legend()
plt.show()




