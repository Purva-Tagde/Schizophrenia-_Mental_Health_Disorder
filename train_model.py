import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle

# Load dataset
df = pd.read_csv("dataset.csv")

# Clean target column
df['Schizophrenia'] = df['Schizophrenia'].str.strip()

# Convert target
df['Schizophrenia'] = df['Schizophrenia'].map({
    'Low Proneness': 0,
    'Elevated Proneness': 1,
    'Moderate Proneness': 1,
    'High Proneness': 1
})

# Remove invalid rows
df = df.dropna(subset=['Schizophrenia'])

# Fill missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Features
X = df[['Age', 'Fatigue', 'Slowing', 'Pain', 'Hygiene', 'Movement']]
y = df['Schizophrenia']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# 🔥 Graphs
plt.figure()
plt.bar(["Accuracy"], [accuracy])
plt.title("Model Accuracy")
plt.savefig("static/accuracy.png")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.savefig("static/confusion_matrix.png")

plt.figure()
plt.hist(y_pred, bins=2)
plt.title("Prediction Distribution")
plt.savefig("static/distribution.png")

# Save model
pickle.dump(model, open('model.pkl','wb'))

print("Model trained successfully!")