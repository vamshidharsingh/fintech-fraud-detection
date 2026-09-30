import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, average_precision_score
from xgboost import XGBClassifier
import joblib
import os
import json

df = pd.read_csv('data/transactions.csv')
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

model = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss')
model.fit(X_train, y_train)

preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

f1 = f1_score(y_test, preds)
pr_auc = average_precision_score(y_test, probs)

metrics = {
    'f1_score': float(f1),
    'pr_auc': float(pr_auc)
}

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/xgboost_model.joblib')
with open('models/metrics.json', 'w') as f:
    json.dump(metrics, f)

print(f"Model trained. F1: {f1:.4f}, PR-AUC: {pr_auc:.4f}")
print("Classification Report:")
print(classification_report(y_test, preds))
