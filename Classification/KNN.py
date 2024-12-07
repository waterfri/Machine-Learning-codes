from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metric import accuracy_score

# load datasets
iris = load_iris()
X, y = iris.data, iris.target

# split datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# create model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# predict
y_pred = knn.predict(X_test)

# evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")