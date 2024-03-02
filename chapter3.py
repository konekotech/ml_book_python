from sklearn.datasets import load_iris
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

"""
(c) konekotech all rights reserved.
This script is the Python version of the R-Code `chapter3.R`.
This script shows how to perform a simple linear regression on the iris dataset.
The script uses the petal width and petal length to predict the petal length.
The script also shows how to perform forward selection based on AIC, and how to plot the residuals histograms.
"""

# Load the iris dataset
iris = load_iris()

# Convert to a pandas dataframe
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

# Show class names
classes = iris.target_names
print("Classes:", classes)
print("Features:", iris.feature_names)

# Select the data for the species versicolor
versicolor_data = df[df["target"] == 1]

# Show the first few rows
print(versicolor_data.head())

# Delete "target" column
del versicolor_data["target"]

# Make train and test data (Odd rows are train, even rows are test. Different from the R-Code because of different starting indices of the indexes.)
train = versicolor_data.iloc[1::2]
test = versicolor_data.iloc[::2]

# Drop NA
train = train.dropna()
test = test.dropna()

print(train)

# Train the model
X_train = train.drop("petal length (cm)", axis=1)
y_train = train["petal length (cm)"]

# Add a constant to the features matrix (required for statsmodels)
X_train_with_const = sm.add_constant(X_train)

# Fit the statsmodels OLS model
ols_model = sm.OLS(y_train, X_train_with_const).fit()

# Display residuals statistics
residuals = ols_model.resid
print("Residuals statistics:\n", residuals.describe())
print("Coefficients: ", ols_model.summary())

# Forward selection based on AIC.
# NOTICE: THIS IS NOT A STEPWISE SELECTION despite the result will be same to R-Code.
selected_features = []
best_aic = float("inf")

for feature in X_train.columns:
    current_features = selected_features + [feature]
    X_train_selected = sm.add_constant(X_train[current_features])
    model = sm.OLS(y_train, X_train_selected).fit()
    current_aic = model.aic
    if current_aic < best_aic:
        best_aic = current_aic
        selected_features = current_features

# Fit the final model
X_train_aic = sm.add_constant(X_train[selected_features])
aic_model = sm.OLS(y_train, X_train_aic).fit()

# Display summary statistics
print(aic_model.summary())

# Predict the train and test data
X_test = test.drop("petal length (cm)", axis=1)
y_test = test["petal length (cm)"]
X_test_selected = sm.add_constant(X_test[selected_features])
y_pred_test = aic_model.predict(X_test_selected)
y_pred_train = aic_model.predict(X_train_aic)

# Errors (train.dif and test.dif in R-Code)
train_errors = y_pred_train - y_train
test_errors = y_pred_test - y_test

# Plot histograms
plt.figure(figsize=(12, 6))

# Train error histogram
plt.subplot(1, 2, 1)
# Adopts the same range as the R-Code
plt.hist(
    train_errors, bins=10, range=(-0.6, 0.4), color="blue", alpha=0.7, edgecolor="black"
)
plt.title("Train Residuals Histogram")
plt.xlabel("Residuals")
plt.ylabel("Frequency")

# Test error histogram
plt.subplot(1, 2, 2)
# Adopts the same range as the R-Code
plt.hist(
    test_errors, bins=7, range=(-0.6, 0.8), color="green", alpha=0.7, edgecolor="black"
)
plt.title("Test Residuals Histogram")
plt.xlabel("Residuals")
plt.ylabel("Frequency")

plt.savefig("./out/iris.png")
