#!/usr/bin/env python3
"""
Comprehensive Diabetes Prediction & Progression Analysis
Includes Regression & Classification Models with Full Analysis
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
)
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. SETUP
# ============================================================================
print("="*80)
print("DIABETES PREDICTION & PROGRESSION ANALYSIS - Complete Project")
print("="*80)

export_dir = r'd:\Machine Learning\Projects\Phase_1\Exports'
os.makedirs(export_dir, exist_ok=True)

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.size'] = 10

print("\n✓ Environment initialized")

# ============================================================================
# 2. LOAD & PREPARE DATA
# ============================================================================
print("\n" + "-"*80)
print("SECTION 1: DATA LOADING & PREPARATION")
print("-"*80)

data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target_progression'] = data.target

print(f"✓ Dataset loaded")
print(f"  Shape: {df.shape}")
print(f"  Features: {df.shape[1] - 1}")
print(f"  Samples: {len(df)}")

# Create binary classification target
median_progression = df['target_progression'].median()
df['is_diabetic'] = (df['target_progression'] > median_progression).astype(int)

print(f"\n✓ Binary target created (median threshold: {median_progression:.2f})")
print(f"  Non-diabetic (0): {(df['is_diabetic'] == 0).sum()} samples")
print(f"  Diabetic (1): {(df['is_diabetic'] == 1).sum()} samples")

# Prepare X and y
X = df.drop(['target_progression', 'is_diabetic'], axis=1)
y_regression = df['target_progression']
y_classification = df['is_diabetic']

# Train-test split
X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X, y_regression, test_size=0.2, random_state=42
)
_, _, y_clf_train, y_clf_test = train_test_split(
    X, y_classification, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✓ Data split and scaled")
print(f"  Training set: {X_train_scaled.shape}")
print(f"  Test set: {X_test_scaled.shape}")

# ============================================================================
# 3. EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n" + "-"*80)
print("SECTION 2: EXPLORATORY DATA ANALYSIS")
print("-"*80)

# Feature distributions
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
fig.suptitle('Distribution of All Features', fontsize=16, fontweight='bold')
axes = axes.ravel()

for idx, col in enumerate(df.columns[:-2]):
    sns.histplot(df[col], kde=True, ax=axes[idx], color='steelblue')
    axes[idx].set_title(f'{col}')
    axes[idx].set_xlabel('')

sns.histplot(df['target_progression'], kde=True, ax=axes[10], color='coral')
axes[10].set_title('Target Progression')
axes[11].remove()
plt.tight_layout()
plt.savefig(f'{export_dir}/01_feature_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Feature distributions saved")

# Correlation matrix
fig, ax = plt.subplots(figsize=(12, 8))
correlation_matrix = df.drop('is_diabetic', axis=1).corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
plt.title('Correlation Matrix - All Features & Target', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{export_dir}/02_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Correlation heatmap saved")

# Feature vs Target relationships
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
fig.suptitle('Feature vs Target Progression Relationship', fontsize=16, fontweight='bold')
axes = axes.ravel()

for idx, col in enumerate(df.columns[:-2]):
    axes[idx].scatter(df[col], df['target_progression'], alpha=0.5, color='steelblue')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Target Progression')
    axes[idx].set_title(f'{col} vs Target')

axes[11].remove()
plt.tight_layout()
plt.savefig(f'{export_dir}/03_feature_target_relationships.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Feature relationships saved")

# ============================================================================
# 4. REGRESSION MODELS
# ============================================================================
print("\n" + "-"*80)
print("SECTION 3: REGRESSION MODELS (Diabetes Progression)")
print("-"*80)

print("\nTraining regression models...")

# Train models
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_reg_train)
y_pred_lr = lr_model.predict(X_test_scaled)

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_reg_train)
y_pred_ridge = ridge_model.predict(X_test_scaled)

lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train_scaled, y_reg_train)
y_pred_lasso = lasso_model.predict(X_test_scaled)

rf_reg_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_reg_model.fit(X_train_scaled, y_reg_train)
y_pred_rf_reg = rf_reg_model.predict(X_test_scaled)

gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train_scaled, y_reg_train)
y_pred_gb = gb_model.predict(X_test_scaled)

print("✓ All regression models trained")

# Evaluate
regression_results = {}
models_reg = {
    'Linear Regression': (lr_model, y_pred_lr),
    'Ridge Regression': (ridge_model, y_pred_ridge),
    'Lasso Regression': (lasso_model, y_pred_lasso),
    'Random Forest': (rf_reg_model, y_pred_rf_reg),
    'Gradient Boosting': (gb_model, y_pred_gb)
}

print("\n" + "="*70)
print(" REGRESSION MODELS PERFORMANCE")
print("="*70)

for model_name, (model, y_pred) in models_reg.items():
    mse = mean_squared_error(y_reg_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_reg_test, y_pred)
    r2 = r2_score(y_reg_test, y_pred)
    
    regression_results[model_name] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}
    
    print(f"\n{model_name}:")
    print(f"  MSE:  {mse:>10.2f}")
    print(f"  RMSE: {rmse:>10.2f}")
    print(f"  MAE:  {mae:>10.2f}")
    print(f"  R²:   {r2:>10.4f}")

print("\n" + "="*70)

# Visualize regression comparisons
results_df = pd.DataFrame(regression_results).T
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].barh(results_df.index, results_df['R2'], color='steelblue')
axes[0].set_xlabel('R² Score')
axes[0].set_title('Model Performance: R² Scores')
axes[0].invert_yaxis()

axes[1].barh(results_df.index, results_df['RMSE'], color='coral')
axes[1].set_xlabel('RMSE')
axes[1].set_title('Model Performance: RMSE (lower is better)')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig(f'{export_dir}/04_regression_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Regression comparison charts saved")

# Actual vs Predicted
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Regression Models: Actual vs Predicted Diabetes Progression', fontsize=14, fontweight='bold')

predictions_reg = [
    ('Linear Regression', y_pred_lr),
    ('Ridge Regression', y_pred_ridge),
    ('Lasso Regression', y_pred_lasso),
    ('Random Forest', y_pred_rf_reg),
    ('Gradient Boosting', y_pred_gb)
]

for idx, (name, y_pred) in enumerate(predictions_reg):
    ax = axes[idx // 3, idx % 3]
    ax.scatter(y_reg_test, y_pred, alpha=0.5, color='steelblue')
    ax.plot([y_reg_test.min(), y_reg_test.max()],
            [y_reg_test.min(), y_reg_test.max()],
            'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(f'{name} (R² = {r2_score(y_reg_test, y_pred):.3f})')
    ax.legend()

axes[1, 2].remove()
plt.tight_layout()
plt.savefig(f'{export_dir}/05_regression_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Actual vs Predicted regression plots saved")

# ============================================================================
# 5. CLASSIFICATION MODELS
# ============================================================================
print("\n" + "-"*80)
print("SECTION 4: CLASSIFICATION MODELS (Diabetic Status)")
print("-"*80)

print("\nTraining classification models...")

# Train models
log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
log_reg_model.fit(X_train_scaled, y_clf_train)
y_pred_log_reg = log_reg_model.predict(X_test_scaled)

rf_clf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_clf_model.fit(X_train_scaled, y_clf_train)
y_pred_rf_clf = rf_clf_model.predict(X_test_scaled)

print("✓ All classification models trained")

# Evaluate
classification_results = {}
models_clf = {
    'Logistic Regression': y_pred_log_reg,
    'Random Forest Classifier': y_pred_rf_clf
}

print("\n" + "="*70)
print(" CLASSIFICATION MODELS PERFORMANCE")
print("="*70)

for model_name, y_pred in models_clf.items():
    accuracy = accuracy_score(y_clf_test, y_pred)
    precision = precision_score(y_clf_test, y_pred)
    recall = recall_score(y_clf_test, y_pred)
    f1 = f1_score(y_clf_test, y_pred)
    
    classification_results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    
    print(f"\n{model_name}:")
    print(f"  Accuracy:  {accuracy:>10.4f}")
    print(f"  Precision: {precision:>10.4f}")
    print(f"  Recall:    {recall:>10.4f}")
    print(f"  F1-Score:  {f1:>10.4f}")

print("\n" + "="*70)

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Confusion Matrices - Classification Models', fontsize=14, fontweight='bold')

for idx, (model_name, y_pred) in enumerate(models_clf.items()):
    cm = confusion_matrix(y_clf_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Non-Diabetic', 'Diabetic'],
                yticklabels=['Non-Diabetic', 'Diabetic'])
    axes[idx].set_title(model_name)
    axes[idx].set_ylabel('Actual')
    axes[idx].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig(f'{export_dir}/06_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Confusion matrices saved")

# ============================================================================
# 6. FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n" + "-"*80)
print("SECTION 5: FEATURE IMPORTANCE ANALYSIS")
print("-"*80)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Feature Importance Analysis', fontsize=14, fontweight='bold')

# Random Forest Regressor
feature_importance_rf = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_reg_model.feature_importances_
}).sort_values('Importance', ascending=False)

axes[0].barh(feature_importance_rf['Feature'], feature_importance_rf['Importance'], color='steelblue')
axes[0].set_xlabel('Importance')
axes[0].set_title('Random Forest Regressor - Feature Importance')
axes[0].invert_yaxis()

# Random Forest Classifier
feature_importance_clf = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_clf_model.feature_importances_
}).sort_values('Importance', ascending=False)

axes[1].barh(feature_importance_clf['Feature'], feature_importance_clf['Importance'], color='coral')
axes[1].set_xlabel('Importance')
axes[1].set_title('Random Forest Classifier - Feature Importance')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig(f'{export_dir}/07_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Feature importance charts saved")

# ============================================================================
# 7. PREDICTIONS
# ============================================================================
print("\n" + "-"*80)
print("SECTION 6: SAMPLE PREDICTIONS")
print("-"*80)

predictions_summary = pd.DataFrame({
    'Actual_Progression': y_reg_test.values,
    'Predicted_Progression': y_pred_gb,
    'Actual_Diabetic': y_clf_test.values,
    'Predicted_Diabetic': y_pred_rf_clf
})

predictions_summary['Diabetic_Status'] = predictions_summary['Predicted_Diabetic'].map(
    {0: 'Non-Diabetic', 1: 'Diabetic'}
)
predictions_summary['Risk_Level'] = pd.cut(
    predictions_summary['Predicted_Progression'],
    bins=3,
    labels=['Low Risk', 'Medium Risk', 'High Risk']
)

print(f"\nFirst 20 predictions:")
print(predictions_summary.head(20).to_string())

print(f"\n\nPrediction Summary Statistics:")
print(f"  Average Predicted Progression: {predictions_summary['Predicted_Progression'].mean():.2f}")
print(f"  Predicted Diabetic (1): {(predictions_summary['Predicted_Diabetic'] == 1).sum()} patients")
print(f"  Predicted Non-Diabetic (0): {(predictions_summary['Predicted_Diabetic'] == 0).sum()} patients")
print(f"  Accuracy on Test Set: {accuracy_score(y_clf_test, y_pred_rf_clf):.4f}")

# ============================================================================
# 8. EXPORT DATA
# ============================================================================
print("\n" + "-"*80)
print("SECTION 7: EXPORTING DATA & REPORTS")
print("-"*80)

# Full dataset
df.to_csv(f'{export_dir}/diabetes_dataset_full.csv', index=False)
print(f"✓ Full dataset exported: diabetes_dataset_full.csv")

# Training set
train_data = pd.DataFrame(X_train_scaled, columns=X.columns)
train_data['target_progression'] = y_reg_train.values
train_data['is_diabetic'] = y_clf_train.values
train_data.to_csv(f'{export_dir}/diabetes_training_set.csv', index=False)
print(f"✓ Training set exported: diabetes_training_set.csv")

# Test set with predictions
test_data = pd.DataFrame(X_test_scaled, columns=X.columns)
test_data['actual_progression'] = y_reg_test.values
test_data['predicted_progression'] = y_pred_gb
test_data['actual_diabetic'] = y_clf_test.values
test_data['predicted_diabetic'] = y_pred_rf_clf
test_data.to_csv(f'{export_dir}/diabetes_test_set_with_predictions.csv', index=False)
print(f"✓ Test set with predictions exported: diabetes_test_set_with_predictions.csv")

# Predictions summary
predictions_summary.to_csv(f'{export_dir}/predictions_summary.csv', index=False)
print(f"✓ Predictions summary exported: predictions_summary.csv")

# Model performance reports
performance_report = pd.DataFrame(regression_results).T
performance_report.to_csv(f'{export_dir}/regression_model_performance.csv')
print(f"✓ Regression performance exported: regression_model_performance.csv")

clf_performance = pd.DataFrame(classification_results).T
clf_performance.to_csv(f'{export_dir}/classification_model_performance.csv')
print(f"✓ Classification performance exported: classification_model_performance.csv")

# Feature importance
feature_importance_rf.to_csv(f'{export_dir}/feature_importance_regression.csv', index=False)
feature_importance_clf.to_csv(f'{export_dir}/feature_importance_classification.csv', index=False)
print(f"✓ Feature importance exported")

# ============================================================================
# 9. SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print(" PROJECT SUMMARY & KEY FINDINGS")
print("="*80)

print("\n📊 DATA OVERVIEW:")
print(f"  • Total Samples: {len(df)}")
print(f"  • Features: {X.shape[1]}")
print(f"  • Target Range (Progression): {df['target_progression'].min():.1f} - {df['target_progression'].max():.1f}")

print("\n🎯 REGRESSION RESULTS (Diabetes Progression Prediction):")
best_reg_model = results_df.sort_values('R2', ascending=False).index[0]
best_reg_r2 = results_df.sort_values('R2', ascending=False)['R2'].iloc[0]
best_reg_rmse = results_df.sort_values('R2', ascending=False)['RMSE'].iloc[0]
print(f"  • Best Model: {best_reg_model}")
print(f"  • R² Score: {best_reg_r2:.4f}")
print(f"  • RMSE: {best_reg_rmse:.2f}")

print("\n🏥 CLASSIFICATION RESULTS (Diabetic Status Prediction):")
clf_results_df = pd.DataFrame(classification_results).T
best_clf_model = clf_results_df.sort_values('Accuracy', ascending=False).index[0]
best_clf_acc = clf_results_df.sort_values('Accuracy', ascending=False)['Accuracy'].iloc[0]
print(f"  • Best Model: {best_clf_model}")
print(f"  • Accuracy: {best_clf_acc:.4f}")

print("\n📈 TOP 3 PREDICTIVE FEATURES (Regression):")
for idx, row in feature_importance_rf.head(3).iterrows():
    print(f"  {idx+1}. {row['Feature']}: {row['Importance']:.4f}")

print("\n✅ MODELS TRAINED:")
print("  Regression: Linear, Ridge, Lasso, Random Forest, Gradient Boosting")
print("  Classification: Logistic Regression, Random Forest")

print("\n💾 DATA EXPORTED (all in Exports folder):")
print("  • diabetes_dataset_full.csv")
print("  • diabetes_training_set.csv")
print("  • diabetes_test_set_with_predictions.csv")
print("  • predictions_summary.csv")
print("  • regression_model_performance.csv")
print("  • classification_model_performance.csv")
print("  • Feature importance CSV files")
print("  • 7 visualization PNG files")

print("\n" + "="*80)
print("✓ ANALYSIS COMPLETE!")
print("="*80)
print(f"\nAll outputs saved to: {export_dir}")
