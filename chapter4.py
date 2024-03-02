import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd


"""
Copyright (c) konekotech all rights reserved.
This script is the Python version of the R-Code `chapter4.R`.
This script shows how to perform a simple neural network on the iris dataset.
This script uses the petal width and petal length to predict the class of the iris.
"""

print("Neural network for the iris dataset. Now loading...")

# Load the iris dataset
iris = load_iris()

# Convert to a pandas dataframe
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

# Make train and test data
train = df.iloc[~df.index.isin(range(4, len(df), 5))]
test = df.iloc[df.index.isin(range(4, len(df), 5))]

# Train the model
X_train = train.drop("target", axis=1)
y_train = train["target"]

# Build the model
model = Sequential()
model.add(Dense(units=8, input_dim=X_train.shape[1], activation="relu"))
model.add(Dense(units=3, activation="softmax"))

# Compile the model
model.compile(
    optimizer=SGD(learning_rate=0.01),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Fit the model
model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, verbose=0)

# Predict the test data
X_test = test.drop("target", axis=1)
y_test = test["target"]
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Compare predictions with actual values
result_table = pd.crosstab(
    index=predicted_classes,
    columns=y_test,
    rownames=["Predicted"],
    colnames=["Actual"],
)
result_table = result_table.rename(index={0: "setosa", 1: "versicolor", 2: "virginica"})
result_table.columns = ["setosa", "versicolor", "virginica"]
print(result_table)
