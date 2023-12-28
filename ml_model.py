import json
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Data validation function
def validate_data(data, target):
    if not data.any() or not target.any():
        raise ValueError("Data is empty or not valid")
    if data.shape[0] != target.shape[0]:
        raise ValueError("Mismatched number of samples and labels")

# Load dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Validate dataset
validate_data(X, y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize our classifier
clf = LogisticRegression(random_state=0)

# Train classifier
clf.fit(X_train, y_train)

# Predict on test set
y_pred = clf.predict(X_test)

# Evaluate the classifier
report = classification_report(y_test, y_pred, output_dict=True)
print(json.dumps(report, indent=2))

# Save the model
np.save('model.npy', clf.coef_)