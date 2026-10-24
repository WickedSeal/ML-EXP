
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# Create Dataset Using DataFrame
# -------------------------------

# How to use it
# 1. Change the values in pd.DataFrame({...}) to your own dataset.
# 2. Set the target column when prompted, for example Buys.
# 3. The program trains the Random Forest, displays accuracy and the classification report, and predicts a new sample.

data = pd.DataFrame({
    "Age": [
        "Young", "Young", "Middle", "Old", "Old",
        "Old", "Middle", "Young", "Young", "Old",
        "Middle", "Middle", "Old", "Young", "Middle"
    ],
    "Income": [
        25000, 30000, 50000, 60000, 65000,
        45000, 52000, 28000, 35000, 70000,
        48000, 55000, 62000, 32000, 58000
    ],
    "Student": [
        "No", "Yes", "No", "No", "Yes",
        "Yes", "No", "Yes", "No", "Yes",
        "Yes", "No", "Yes", "No", "Yes"
    ],
    "Buys": [
        "No", "No", "Yes", "Yes", "Yes",
        "No", "Yes", "No", "No", "Yes",
        "Yes", "Yes", "Yes", "No", "Yes"
    ]
})

print("Dataset:")
print(data)

print("\nColumns:", list(data.columns))

# -------------------------------
# Select Target Column
# -------------------------------

target = input("\nEnter target column name: ")

if target not in data.columns:
    raise ValueError("Target column not found!")

data = data.dropna().copy()

X = data.drop(columns=[target])
y = data[target]

# Encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Convert categorical features into numeric columns
X = pd.get_dummies(X, drop_first=False)

# -------------------------------
# Split Dataset
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# -------------------------------
# Train Random Forest
# -------------------------------

clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

clf.fit(X_train, y_train)

# -------------------------------
# Evaluate Model
# -------------------------------

y_pred = clf.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=range(len(label_encoder.classes_)),
        target_names=label_encoder.classes_.astype(str),
        zero_division=0
    )
)

# -------------------------------
# Predict New Sample
# -------------------------------

print("\nEnter values for a new sample:")

sample = {}

for column in data.drop(columns=[target]).columns:
    value = input(f"{column}: ")

    if pd.api.types.is_numeric_dtype(data[column]):
        value = float(value)

    sample[column] = value

new_data = pd.DataFrame([sample])

# Encode categorical features consistently
new_data = pd.get_dummies(new_data, drop_first=False)
new_data = new_data.reindex(columns=X.columns, fill_value=0)

# Predict class
prediction = clf.predict(new_data)[0]
predicted_label = label_encoder.inverse_transform([prediction])[0]

print("\nPredicted class:", predicted_label)
