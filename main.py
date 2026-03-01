
# OBESITY RISK PREDICTION PROJECT


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier




# 1. LOAD DATA

df = pd.read_csv("data/obesity.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())


# 2. PREPROCESSING





y = df["NObeyesdad"]
X = df.drop("NObeyesdad", axis=1)

X = pd.get_dummies(X, drop_first=True)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# 3. BASELINE MODEL


print("\n===== Logistic Regression (Baseline) =====")

baseline = LogisticRegression(max_iter=2000)
baseline.fit(X_train, y_train)

y_pred_lr = baseline.predict(X_test)

acc_lr = accuracy_score(y_test, y_pred_lr)
print("Accuracy:", acc_lr)
print(classification_report(y_test, y_pred_lr))


# 4. RANDOM FOREST (Bagging)


print("\n===== Random Forest =====")

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

print("Accuracy:", acc_rf)


# 5. GRADIENT BOOSTING

print("\n===== Gradient Boosting =====")

gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)

y_pred_gb = gb.predict(X_test)
acc_gb = accuracy_score(y_test, y_pred_gb)

print("Accuracy:", acc_gb)




# 7. VOTING CLASSIFIER


print("\n===== Voting Classifier =====")

voting = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('gb', gb),
        ('lr', baseline)
    ],
    voting='soft'
)

voting.fit(X_train, y_train)
y_pred_vote = voting.predict(X_test)

acc_vote = accuracy_score(y_test, y_pred_vote)
print("Accuracy:", acc_vote)


# 8. STACKING


print("\n===== Stacking Classifier =====")

stack = StackingClassifier(
    estimators=[
        ('rf', rf),
        ('gb', gb)
    ],
    final_estimator=LogisticRegression()
)

stack.fit(X_train, y_train)
y_pred_stack = stack.predict(X_test)

acc_stack = accuracy_score(y_test, y_pred_stack)
print("Accuracy:", acc_stack)


# 9. MODEL COMPARISON


print("\n===== Model Comparison =====")

results = {
    "Logistic Regression": acc_lr,
    "Random Forest": acc_rf,
    "Gradient Boosting": acc_gb,
    
    "Voting": acc_vote,
    "Stacking": acc_stack
}

for model, acc in results.items():
    print(f"{model}: {acc:.4f}")


# 10. CONFUSION MATRIX (Best Model)


best_model_pred = y_pred_rf  

cm = confusion_matrix(y_test, best_model_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 11. FEATURE IMPORTANCE


importances = rf.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Features:")
print(importance_df.head(10))
