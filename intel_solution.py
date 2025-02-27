import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
data = {
    "Age": [22, 25, 35, 45, 50, 60, 30, 40],
    "EstimatedSalary": [30000, 45000, 60000, 80000, 150000, 120000, 50000, 70000],
    "Purchased": [0, 0, 1, 1, 1, 1, 0, 1]
}
df = pd.DataFrame(data)
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Decision Tree Model Accuracy:", accuracy)
new_customer = pd.DataFrame([[30, 75000]], columns=["Age", "EstimatedSalary"])
prediction = model.predict(new_customer)
print("Prediction for new customer:", "Purchased" if prediction[0] == 1 else "Not Purchased")
