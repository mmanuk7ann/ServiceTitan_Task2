import json
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Loading Logs
with open("logs.json", "r") as f:
    logs = json.load(f)

# Feature Enginering 
def extract_features(log):
    features = {}

    features['latency_ms'] = log.get('response_latency_ms', 0)

    feedback = log.get('user_feedback', '').lower()
    features['feedback'] = 1 if feedback == 'thumb_up' else 0

    # Query length in words
    query = log.get('query', '')
    features['query_len'] = len(query.split())

    # Retrieved chunks
    chunks = log.get('retrieved_chunks', [])
    sources = [chunk.get('source') for chunk in chunks]

    # Token count based on content word length
    tokens = [len(chunk.get('content', '').split()) for chunk in chunks if chunk.get('content')]

    features['avg_doc_tokens'] = np.mean(tokens) if tokens else 0
    features['total_tokens'] = np.sum(tokens) if tokens else 0
    features['num_docs'] = len(sources)

    # One-hot encode source types
    for src in ['Engineering Wiki', 'Archived Design Docs (PDFs)', 'Confluence']:
        features[f'doc_from_{src}'] = sources.count(src)

    return features


df = pd.DataFrame([extract_features(log) for log in logs])
df.dropna(inplace=True)

# Output Folder
os.makedirs("data", exist_ok=True)
df.to_csv("data/logs_features.csv", index=False)


# Classification: Feedback Prediction 
X_clf = df.drop(columns=['feedback', 'latency_ms'])
y_clf = df['feedback']

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.3, random_state=42)

clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
clf.fit(X_train_clf, y_train_clf)
y_pred_clf = clf.predict(X_test_clf)

print("Feedback Classification Report:")
print(classification_report(y_test_clf, y_pred_clf))

# Feature importance
clf_feature_importance = pd.Series(clf.feature_importances_, index=X_clf.columns)
plt.figure(figsize=(8, 5))
clf_feature_importance.sort_values().plot(kind='barh', title='Feature Importance: Feedback Prediction')
plt.tight_layout()
plt.savefig("data/feedback_feature_importance.png")
plt.close()


# Regression: Latency Prediction 
X_reg = df.drop(columns=['latency_ms', 'feedback'])
y_reg = df['latency_ms']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

reg = LinearRegression()
reg.fit(X_train_reg, y_train_reg)
y_pred_reg = reg.predict(X_test_reg)

rmse = mean_squared_error(y_test_reg, y_pred_reg, squared=False)
print(f"Latency RMSE: {rmse:.2f} ms")

# Regression coefficients
reg_coef = pd.Series(reg.coef_, index=X_reg.columns)
plt.figure(figsize=(8, 5))
reg_coef.sort_values().plot(kind='barh', title='Linear Regression Coefficients: Latency')
plt.tight_layout()
plt.savefig("data/latency_regression_coefficients.png")
plt.close()

# Correlation Heatmap 
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("data/feature_correlation_heatmap.png")
plt.close()


# Task 2: Cost of increasing k=4 → k=10
chunks_added = 6  
tokens_per_chunk = 400
queries_per_month = 100_000
token_cost_per_million = 3.00

total_tokens_added = chunks_added * tokens_per_chunk * queries_per_month
monthly_cost_increase = (total_tokens_added / 1_000_000) * token_cost_per_million

print(f" Monthly Gen Cost Increase (Option B): ${monthly_cost_increase:.2f}")


# Summary Statistics
p99_sla = 3500
slow_pct = (df['latency_ms'] > p99_sla).mean() * 100
pdf_bad_pct = df[df['feedback'] == 0]['doc_from_Archived Design Docs (PDFs)'].gt(0).mean() * 100

print(f"Queries slower than 3.5s: {slow_pct:.1f}%")
print(f"Bad feedback responses including PDFs: {pdf_bad_pct:.1f}%")



