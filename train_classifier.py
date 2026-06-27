import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics         import (accuracy_score, classification_report,
                                     confusion_matrix)
from sklearn.preprocessing   import LabelEncoder
from collections             import Counter

# ── Load & Validate Data ───────────────────────────────────────────────────────
print("[INFO] Loading data from data.pickle...")
data_dict = pickle.load(open('./data.pickle', 'rb'))

raw_data   = data_dict['data']
raw_labels = data_dict['labels']

EXPECTED_FEATURES = 84  # 2 hands × 21 landmarks × 2 coords (x, y)

# Filter out any malformed samples (safety net)
filtered = [(d, l) for d, l in zip(raw_data, raw_labels) if len(d) == EXPECTED_FEATURES]
skipped  = len(raw_data) - len(filtered)

if skipped:
    print(f"[WARNING] Dropped {skipped} samples with incorrect feature length.")

if len(filtered) == 0:
    raise ValueError("No valid samples found. Re-run feature extraction.")

data   = np.array([d for d, _ in filtered])
labels = np.array([l for _, l in filtered])

print(f"  Dataset shape     : {data.shape}")
print(f"  Total samples     : {len(labels)}")
print(f"  Feature size      : {data.shape[1]}  ✓" if data.shape[1] == EXPECTED_FEATURES
      else f"  Feature size      : {data.shape[1]}  ✗ Expected {EXPECTED_FEATURES}")
print(f"  Class distribution: {dict(sorted(Counter(labels).items()))}")


# ── Encode Labels ──────────────────────────────────────────────────────────────
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
print(f"\n[INFO] Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")


# ── Train / Test Split ─────────────────────────────────────────────────────────
x_train, x_test, y_train, y_test = train_test_split(
    data, labels_encoded,
    test_size=0.2,
    shuffle=True,
    stratify=labels_encoded,
    random_state=42
)

print(f"\n[INFO] Train: {len(x_train)} samples | Test: {len(x_test)} samples")


# ── Hyperparameter Tuning ──────────────────────────────────────────────────────
print("\n[INFO] Running GridSearchCV (this may take a moment)...")

param_grid = {
    'n_estimators'     : [100, 200, 300],
    'max_depth'        : [None, 10, 20],
    'min_samples_split': [2, 5],
    'max_features'     : ['sqrt', 'log2']
}

grid_search = GridSearchCV(
    estimator  = RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_grid = param_grid,
    cv         = 5,
    scoring    = 'accuracy',
    n_jobs     = -1,
    verbose    = 1,
    refit      = True   # refit=True means grid_search.best_estimator_ is ready to use
)

grid_search.fit(x_train, y_train)

model         = grid_search.best_estimator_   # reuse — no need to retrain separately
best_params   = grid_search.best_params_
best_cv_score = grid_search.best_score_

print(f"\n[TUNING] Best Parameters  : {best_params}")
print(f"[TUNING] Best CV Accuracy : {best_cv_score * 100:.2f}%")


# ── Evaluation ────────────────────────────────────────────────────────────────
y_pred        = model.predict(x_test)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"\n[RESULT] Test Set Accuracy : {test_accuracy * 100:.2f}%")
print("\n[REPORT] Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))


# ── Confusion Matrix ───────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix — Two-Hand Gesture Classifier', fontsize=14)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
print("[INFO] Saved confusion_matrix.png")


# ── Feature Importance ────────────────────────────────────────────────────────
# Correct naming for interleaved format: x0,y0,x1,y1... per hand
def make_feature_names(prefix, n=21):
    return [f"{prefix}_lm{i}_{axis}" for i in range(n) for axis in ('x', 'y')]

feature_names = make_feature_names('L') + make_feature_names('R')
# = ['L_lm0_x', 'L_lm0_y', ..., 'L_lm20_y', 'R_lm0_x', ..., 'R_lm20_y']

importances = model.feature_importances_
top_idx     = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(13, 5))
plt.bar(range(20), importances[top_idx], color='steelblue')
plt.xticks(range(20), [feature_names[i] for i in top_idx], rotation=45, ha='right')
plt.title('Top 20 Most Important Landmarks', fontsize=14)
plt.ylabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()
print("[INFO] Saved feature_importance.png")


# ── Save Model ────────────────────────────────────────────────────────────────
with open('model.p', 'wb') as f:
    pickle.dump({
        'model'         : model,
        'label_encoder' : le,
        'best_params'   : best_params,
        'best_cv_score' : best_cv_score,
        'test_accuracy' : test_accuracy,
        'feature_names' : feature_names
    }, f)

print("\n[SAVED] model.p")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("           TRAINING SUMMARY")
print("=" * 50)
print(f"  Total Samples       : {len(labels)}")
print(f"  Feature Vector Size : {data.shape[1]}  (84 = 2 hands × 21 × 2)")
print(f"  Classes             : {list(le.classes_)}")
print(f"  Best CV Accuracy    : {best_cv_score * 100:.2f}%")
print(f"  Test Accuracy       : {test_accuracy * 100:.2f}%")
print(f"  Best Parameters     : {best_params}")
print("=" * 50)