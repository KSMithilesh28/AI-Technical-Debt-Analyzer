import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Create synthetic training data
np.random.seed(42)
n_samples = 100

# Feature names for technical debt analysis
feature_columns = ["lines_of_code", "loops_count", "functions_count", "comments_count"]

# Generate synthetic data
lines = np.random.randint(10, 500, n_samples)
loops = np.random.randint(0, 20, n_samples)
functions = np.random.randint(1, 30, n_samples)
comments = np.random.randint(0, 50, n_samples)

# Create a simple label: HIGH risk if lines > 300 or few comments, MEDIUM if moderate, LOW otherwise
labels = np.zeros(n_samples)
for i in range(n_samples):
    if lines[i] > 300 or comments[i] < 5:
        labels[i] = 2  # HIGH
    elif lines[i] > 100 or comments[i] < 10:
        labels[i] = 1  # MEDIUM
    else:
        labels[i] = 0  # LOW

# Create DataFrame
X_train = pd.DataFrame({
    feature_columns[0]: lines,
    feature_columns[1]: loops,
    feature_columns[2]: functions,
    feature_columns[3]: comments
})

# Train model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, labels)

# Save model and feature columns
joblib.dump(model, "model.pkl")
joblib.dump(feature_columns, "features.pkl")

print("Model and features saved successfully!")
