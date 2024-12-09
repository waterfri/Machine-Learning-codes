import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 3) # 100 samples 3 features
y = 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + np.random.randn(100) * 0.5 # Linear relation with noise

# split datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create a Ridge model
ridge = Ridge(alpha=1.0) # alpha = lambda, controls regularization strength, smaller values reduce regularization

# fit
ridge.fit(X_train, y_train)

# predict
y_pred = ridge.predict(X_test)

# evaluate
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error:{mse:.4f}")
print(f"Model Coefficients:{ridge.coef_}")
