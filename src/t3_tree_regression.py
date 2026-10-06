# Exam score vs. hours of sleep  |  Decision Tree regression
# Paste into a Jupyter cell. Needs numpy, scipy, scikit-learn, matplotlib.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# ---------- 1. Data (synthetic, modelled on the real-world pattern) ----------
rng = np.random.default_rng(7)
n = 60
def f(x):
    return np.select([x < 5, x < 7, x < 9], [57, 70, 83], 78)
x = rng.uniform(3, 11, n)
y = f(x) + rng.normal(0, 5.5, n)                    # additive noise
X = x.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)

# ---------- 2. Model ----------
# Decision tree regression: a staircase of if/else rules
from sklearn.tree import DecisionTreeRegressor, export_text

max_depth = 2
min_samples_leaf = 4

model = DecisionTreeRegressor(max_depth=max_depth,
                              min_samples_leaf=min_samples_leaf,
                              random_state=0)
model.fit(X_train, y_train)

print("number of leaves:", model.get_n_leaves())
print(export_text(model, feature_names=["x"], decimals=2))

# ---------- 3. Evaluate ----------
print("train MSE:", mean_squared_error(y_train, model.predict(X_train)))
print("test  MSE:", mean_squared_error(y_test, model.predict(X_test)))
print("test  R²  :", r2_score(y_test, model.predict(X_test)))
cv = KFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error")
print("5-fold CV MSE:", -scores.mean(), "+/-", scores.std())

# ---------- 4. Plot ----------
X_grid = np.linspace(3, 11, 400).reshape(-1, 1)
plt.scatter(X_train, y_train, s=18, label="train")
plt.scatter(X_test, y_test, s=18, facecolors="none", edgecolors="k", label="test")
plt.plot(X_grid, model.predict(X_grid), lw=2.5, label="Decision Tree fit")
plt.xlabel("Hours of sleep before exam"); plt.ylabel("Exam score (%)"); plt.legend(); plt.show()