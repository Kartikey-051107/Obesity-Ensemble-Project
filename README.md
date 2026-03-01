# Obesity Risk Prediction using Ensemble Learning

## About the Project
In this project, I built a machine learning model to predict obesity risk categories using health and lifestyle data.

The goal was to compare a basic model with ensemble learning methods and see how performance improves.

---

## Dataset
The dataset contains:
- Age
- Gender
- Height
- Eating habits
- Physical activity
- Transportation type
- Other lifestyle features

Target column:
- `NObeyesdad` (Obesity category)

Dataset size:
- 20,758 rows
- 18 columns

---

## Models Used

1. Logistic Regression (Baseline)
2. Random Forest
3. Gradient Boosting
4. Voting Classifier
5. Stacking Classifier

---

## Results

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 86.8% |
| Random Forest | 89.7% |
| Gradient Boosting | 90.5% |
| Voting Classifier | 90.65% |
| Stacking Classifier | 90.48% |

The Voting Classifier performed the best.

---

## Conclusion
Ensemble methods performed better than the basic Logistic Regression model.  
Combining multiple models helped improve prediction accuracy.

---



