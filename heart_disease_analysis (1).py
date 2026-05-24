# ============================================================
# HEART DISEASE PREDICTION - Data Science Internship Project
# Company: Thirnex | Domain: Healthcare / Clinical AI
# ============================================================

# ── 1. IMPORTS ──────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_auc_score)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── 2. DATA LOADING ──────────────────────────────────────────
print("=" * 60)
print("HEART DISEASE PREDICTION - THIRNEX INTERNSHIP PROJECT")
print("=" * 60)

df = pd.read_csv('heart_disease.csv')
print(f"\n[Dataset Loaded]")
print(f"  Shape    : {df.shape[0]} records × {df.shape[1]} columns")
print(f"  Features : {list(df.columns[:-1])}")
print(f"  Target   : 'target' (0=No Disease, 1=Heart Disease)\n")
print(df.head())

# ── 3. DATA CLEANING & EDA ───────────────────────────────────
print("\n[Missing Values]")
print(df.isnull().sum())

print("\n[Descriptive Statistics]")
print(df.describe().round(2))

print("\n[Target Distribution]")
print(df['target'].value_counts())
print(f"  Disease prevalence: {df['target'].mean()*100:.1f}%")

# ── 4. VISUALIZATIONS ────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')

# 4a. Age Distribution
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(df[df['target']==0]['age'], bins=20, alpha=0.7,
        color='#1a73e8', label='No Disease')
ax.hist(df[df['target']==1]['age'], bins=20, alpha=0.7,
        color='#e8431a', label='Heart Disease')
ax.set_title('Age Distribution by Heart Disease Status', fontweight='bold')
ax.set_xlabel('Age'); ax.set_ylabel('Count')
ax.legend(); plt.tight_layout()
plt.savefig('viz1_age_distribution.png', dpi=150); plt.close()
print("\n[Fig 1 Saved] Age Distribution")

# 4b. Correlation Heatmap
fig, ax = plt.subplots(figsize=(11, 8))
mask = np.triu(np.ones_like(df.corr(), dtype=bool))
sns.heatmap(df.corr(), mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, square=True, ax=ax)
ax.set_title('Feature Correlation Heatmap', fontweight='bold')
plt.tight_layout()
plt.savefig('viz2_correlation_heatmap.png', dpi=150); plt.close()
print("[Fig 2 Saved] Correlation Heatmap")

# 4c. Cholesterol vs Max Heart Rate
fig, ax = plt.subplots(figsize=(9, 6))
for t, label, c in [(0,'No Disease','#1a73e8'), (1,'Heart Disease','#e8431a')]:
    m = df['target'] == t
    ax.scatter(df[m]['cholesterol'], df[m]['max_heart_rate'],
               alpha=0.5, color=c, label=label, s=40)
ax.set_xlabel('Cholesterol (mg/dl)'); ax.set_ylabel('Max Heart Rate')
ax.set_title('Cholesterol vs Max Heart Rate', fontweight='bold')
ax.legend(); plt.tight_layout()
plt.savefig('viz5_cholesterol_hr.png', dpi=150); plt.close()
print("[Fig 3 Saved] Cholesterol vs Heart Rate")

# ── 5. MODEL BUILDING ────────────────────────────────────────
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_s, y_train)
rf_pred = rf.predict(X_test_s)
rf_acc  = accuracy_score(y_test, rf_pred)
rf_auc  = roc_auc_score(y_test, rf.predict_proba(X_test_s)[:, 1])

# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)
lr_pred = lr.predict(X_test_s)
lr_acc  = accuracy_score(y_test, lr_pred)
lr_auc  = roc_auc_score(y_test, lr.predict_proba(X_test_s)[:, 1])

# ── 6. EVALUATION ────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL EVALUATION RESULTS")
print("=" * 60)
print(f"\nRandom Forest     → Accuracy: {rf_acc:.4f} | AUC: {rf_auc:.4f}")
print(f"Logistic Regression → Accuracy: {lr_acc:.4f} | AUC: {lr_auc:.4f}")

print("\n[Classification Report — Random Forest]")
print(classification_report(y_test, rf_pred,
      target_names=['No Disease', 'Heart Disease']))

# Confusion matrix plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (pred, title) in zip(axes, [
        (rf_pred, f'Random Forest ({rf_acc:.1%})'),
        (lr_pred, f'Logistic Regression ({lr_acc:.1%})')]):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Disease','Disease'],
                yticklabels=['No Disease','Disease'])
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig('viz4_confusion_matrix.png', dpi=150); plt.close()
print("[Fig 4 Saved] Confusion Matrices")

# Feature Importance
feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
fig, ax = plt.subplots(figsize=(9, 6))
feat_imp.plot(kind='barh', ax=ax, color='#1a73e8')
ax.set_title('Feature Importance (Random Forest)', fontweight='bold')
ax.set_xlabel('Importance Score'); plt.tight_layout()
plt.savefig('viz3_feature_importance.png', dpi=150); plt.close()
print("[Fig 5 Saved] Feature Importance")

# ── 7. PREDICTIONS (Sample) ──────────────────────────────────
print("\n[Sample Predictions on Test Set]")
sample = X_test.iloc[:10].copy()
sample['actual']    = y_test.iloc[:10].values
sample['predicted'] = rf_pred[:10]
sample['confidence'] = rf.predict_proba(X_test_s[:10])[:, 1].round(3)
print(sample[['age','cholesterol','max_heart_rate','actual','predicted','confidence']].to_string())

print("\n[Key Insights]")
print(f"  1. Top risk factor: {feat_imp.index[-1].replace('_',' ').title()}")
print(f"  2. Patients >55 show {df[df['age']>55]['target'].mean()*100:.0f}% disease rate")
print(f"  3. Model achieves {rf_acc:.1%} accuracy — clinically actionable")
print(f"  4. AUC-ROC: {rf_auc:.3f} — strong discriminative ability")

print("\n[Analysis Complete]")
