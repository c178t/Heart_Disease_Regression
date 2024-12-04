# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


df = pd.read_csv('Heart_Disease_Prediction_Filtered.csv')


df['Heart Disease'] = df['Heart Disease'].apply(lambda x: 1 if x == 'Presence' else 0)


df_male = df[df['Sex'] == 1]  
df_female = df[df['Sex'] == 0]  


def fit_and_evaluate(df, gender):
    X = df.iloc[:, 1:14]  
    y = df['Heart Disease']  


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logistic_model = LogisticRegression(max_iter=1000) 
    logistic_model.fit(X_train, y_train)

    y_pred = logistic_model.predict(X_test)
    y_prob = logistic_model.predict_proba(X_test)[:, 1]  # Get probabilities for positive class

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"Gender: {gender}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:\n", conf_matrix)
    print(f"ROC AUC Score: {roc_auc:.4f}")
    
    return logistic_model


model_male = fit_and_evaluate(df_male, "Male")
model_female = fit_and_evaluate(df_female, "Female")

coefficients_male = model_male.coef_[0]
coefficients_female = model_female.coef_[0]
print(f"Male Coefficients: {coefficients_male}")
print(f"Female Coefficients: {coefficients_female}")

# Visualize the relationship between cholesterol and heart disease probability for both genders
plt.figure(figsize=(12, 6))

# Male Cholesterol vs Predicted Probability
plt.subplot(1, 2, 1)
sns.scatterplot(x=df_male['Cholesterol'], y=model_male.predict_proba(df_male.iloc[:, 1:14])[:, 1], color='blue', label='Predicted Probability')
plt.axhline(0.5, color='red', linestyle='--', label='Threshold (0.5)')
plt.title('Male: Cholesterol vs Predicted Probability of Heart Disease')
plt.xlabel('Cholesterol Level')
plt.ylabel('Predicted Probability of Heart Disease')
plt.legend()
plt.grid()

# Female Cholesterol vs Predicted Probability
plt.subplot(1, 2, 2)
sns.scatterplot(x=df_female['Cholesterol'], y=model_female.predict_proba(df_female.iloc[:, 1:14])[:, 1], color='green', label='Predicted Probability')
plt.axhline(0.5, color='red', linestyle='--', label='Threshold (0.5)')
plt.title('Female: Cholesterol vs Predicted Probability of Heart Disease')
plt.xlabel('Cholesterol Level')
plt.ylabel('Predicted Probability of Heart Disease')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
# %%
