# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# %%

df = pd.read_csv('Heart_Disease_Prediction_Filtered.csv')


df['Heart Disease'] = df['Heart Disease'].apply(lambda x: 1 if x == 'Presence' else 0)


X = df[['Cholesterol']]
y = df['Heart Disease']


logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X, y)

# Create a range of cholesterol values for prediction
cholesterol_range = np.linspace(df['Cholesterol'].min(), df['Cholesterol'].max(), 100).reshape(-1, 1)
predicted_probs = logistic_model.predict_proba(cholesterol_range)[:, 1]

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(cholesterol_range, predicted_probs, color='blue')
plt.axhline(0.5, color='red', linestyle='--', label='Threshold (0.5)')
plt.title('Cholesterol Levels vs Probability of Heart Disease')
plt.xlabel('Cholesterol Level')
plt.ylabel('Predicted Probability of Heart Disease')
plt.legend()
plt.grid()
plt.show()


indices = np.where(predicted_probs >= 0.5)[0]

if indices.size > 0:
    threshold_cholesterol = cholesterol_range[indices[0]][0]
    print(f"The cholesterol level at which the probability of heart disease is approximately 50% is: {threshold_cholesterol:.2f}")
else:
    print("No cholesterol level found where the probability of heart disease is 50% or higher.")
# %%