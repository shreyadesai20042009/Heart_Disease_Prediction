import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("Heart_Disease_Prediction.csv")
df['Heart Disease'] = df['Heart Disease'].map({"Presence": 1, "Absence": 0})

# Features and target
X = df.drop("Heart Disease", axis=1)
y = df["Heart Disease"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved.")
