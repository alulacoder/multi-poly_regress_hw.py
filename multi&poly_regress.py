import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error


df = pd.read_csv("data science/iris.csv")

# Select features and target
X = df[['sepal_width', 'petal_length', 'petal_width']]
y = df['sepal_length']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predictions
y_pred_linear = linear_model.predict(X_test)

# Evaluation
print("===== Multivariable Linear Regression =====")
print("R² Score:", r2_score(y_test, y_pred_linear))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_linear))
print("Intercept:", linear_model.intercept_)
print("Coefficients:")
for feature, coef in zip(X.columns, linear_model.coef_):
    print(f"  {feature}: {coef}")


poly_model = Pipeline([
    ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
    ('linear_regression', LinearRegression())
])

poly_model.fit(X_train, y_train)

# Predictions
y_pred_poly = poly_model.predict(X_test)

# Evaluation
print("\n===== Polynomial Regression (Degree 2) =====")
print("R² Score:", r2_score(y_test, y_pred_poly))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_poly))


print("\n===== Model Comparison =====")
print(f"Linear Regression R²: {r2_score(y_test, y_pred_linear):.4f}")
print(f"Polynomial Regression R²: {r2_score(y_test, y_pred_poly):.4f}")
