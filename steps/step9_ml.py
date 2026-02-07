import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import io
import gc
import joblib
from datetime import datetime
from typing import List, Dict, Tuple, Any

# Import utilities
from ..utils.dependency_utils import SKLEARN_AVAILABLE, XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE, graceful_fallback
from ..utils.history_utils import log_action
from ..utils.memory_utils import clear_model_cache

# ============================================
# NEW: TARGET VALIDATION AND PROBLEM DETECTION
# ============================================

def detect_problem_type(y: pd.Series) -> Tuple[str, str, Dict]:
    """Auto-detect problem type based on target variable characteristics"""
    n_unique = y.nunique()
    n_samples = len(y)
    
    # Check if target appears to be categorical
    if n_unique <= 10:
        return "classification", f"Categorical target with {n_unique} classes", {"classes": n_unique}
    
    # Check if target appears to be continuous
    elif n_unique > 20 and pd.api.types.is_numeric_dtype(y):
        # Check if values look like regression (continuous)
        unique_ratio = n_unique / n_samples
        if unique_ratio > 0.3:  # More than 30% unique values
            return "regression", f"Continuous target with {n_unique} unique values", {"unique_ratio": unique_ratio}
    
    # Check for integer targets that might be counts or IDs
    elif pd.api.types.is_integer_dtype(y):
        if n_unique > 50 and n_unique < n_samples * 0.8:
            return "regression", f"Integer target (possible count/regression) with {n_unique} unique values", {}
    
    # Default to classification for small unique counts
    if n_unique <= 20:
        return "classification", f"Categorical target with {n_unique} classes", {"classes": n_unique}
    
    return "unknown", f"Target has {n_unique} unique values", {}

def validate_problem_target_compatibility(problem_type: str, y: pd.Series) -> Tuple[bool, List[str]]:
    """Validate if selected problem type matches target variable"""
    warnings = []
    
    n_unique = y.nunique()
    n_samples = len(y)
    
    if problem_type.lower() == 'classification':
        # For classification, target should be categorical
        if n_unique > 50:
            warnings.append(f"⚠️ Classification with {n_unique} unique values - consider regression instead")
        
        if n_unique > n_samples * 0.5:
            warnings.append(f"⚠️ High cardinality ({n_unique}/{n_samples} unique) - might be regression problem")
        
        # Check for numeric but categorical data
        if pd.api.types.is_numeric_dtype(y) and n_unique <= 10:
            warnings.append("⚠️ Numeric target with few unique values - ensure these are categorical labels")
        
        if n_unique < 2:
            return False, ["Need at least 2 classes for classification"]
            
    else:  # regression
        # For regression, target should be continuous
        if n_unique <= 5:
            warnings.append(f"⚠️ Regression with only {n_unique} unique values - consider classification")
        
        if not pd.api.types.is_numeric_dtype(y):
            return False, ["Regression requires numeric target variable"]
    
    return True, warnings

# ============================================
# ENHANCED METRICS AND RELIABILITY SCORES
# ============================================

def build_metrics_table(results: List[Dict], problem_type: str) -> pd.DataFrame:
    """Build a canonical metrics DataFrame (single source of truth)"""
    rows = []
    
    for r in results:
        if 'error' in r:
            continue
        
        base = {
            "Model": r["model_name"],
            "Training Time (s)": r["training_time"],
            "CV Mean": r.get("cv_mean", np.nan),
            "CV Std": r.get("cv_std", np.nan)
        }
        
        if problem_type == "classification":
            base.update({
                "Accuracy": r.get("accuracy", np.nan),
                "Precision": r.get("precision", np.nan),
                "Recall": r.get("recall", np.nan),
                "F1": r.get("f1", np.nan),
                "ROC_AUC": r.get("roc_auc", np.nan)
            })
            
            # Add classification-specific error metrics
            if 'confusion_matrix' in r:
                cm = r['confusion_matrix']
                if cm.shape == (2, 2):  # Binary classification
                    tn, fp, fn, tp = cm.ravel()
                    base["FPR"] = fp / (fp + tn) if (fp + tn) > 0 else np.nan
                    base["FNR"] = fn / (fn + tp) if (fn + tp) > 0 else np.nan
                    
            # Add confidence-based metrics if available
            if 'y_pred_proba' in r:
                base["Avg Confidence"] = r['y_pred_proba'].mean()
                    
        else:  # regression
            base.update({
                "R2": r.get("r2", np.nan),
                "RMSE": r.get("rmse", np.nan),
                "MAE": r.get("mae", np.nan),
                "MAPE": r.get("mape", np.nan) if not pd.isna(r.get("mape")) else np.nan
            })
        
        # ✅ CRITICAL FIX: Actually append the row
        rows.append(base)
    
    return pd.DataFrame(rows)

def add_reliability_scores(df: pd.DataFrame, problem_type: str) -> pd.DataFrame:
    """Add risk-adjusted reliability scores"""
    df = df.copy()
    
    if problem_type == "classification":
        # Risk-adjusted reliability: penalize instability
        if 'F1' in df.columns and 'CV Std' in df.columns:
            reliability = df["F1"] - 0.5 * df["CV Std"].fillna(0)
            # Clip to [0, 1] for interpretability
            df["Reliability Score"] = np.clip(reliability, 0, 1)
        elif 'Accuracy' in df.columns and 'CV Std' in df.columns:
            reliability = df["Accuracy"] - 0.5 * df["CV Std"].fillna(0)
            df["Reliability Score"] = np.clip(reliability, 0, 1)
    else:  # regression
        # Risk-adjusted reliability for regression
        if 'R2' in df.columns and 'CV Std' in df.columns:
            reliability = df["R2"] - 0.5 * df["CV Std"].fillna(0)
            # Clip to [-1, 1] for R²-like interpretability
            df["Reliability Score"] = np.clip(reliability, -1, 1)
    
    return df

def compute_error_analysis(y_true, y_pred, problem_type: str, y_pred_proba=None) -> Dict:
    """Compute comprehensive error analysis"""
    if problem_type == "regression":
        residuals = y_true - y_pred
        abs_errors = np.abs(residuals)
        
        # Safe percentage error (avoid division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            # Only compute MAPE when all values are non-zero
            non_zero_mask = np.abs(y_true) > 1e-10
            if non_zero_mask.any():
                pct_errors = np.abs(residuals[non_zero_mask]) / np.abs(y_true[non_zero_mask])
                mean_pct_error = np.mean(pct_errors) * 100
            else:
                pct_errors = np.full_like(residuals, np.nan)
                mean_pct_error = np.nan
        
        return {
            "residuals": residuals,
            "abs_errors": abs_errors,
            "pct_errors": pct_errors,
            "mean_abs_error": np.mean(abs_errors),
            "median_abs_error": np.median(abs_errors),
            "pct_90_error": np.percentile(abs_errors, 90),
            "mean_pct_error": mean_pct_error,
            "worst_indices": np.argsort(abs_errors)[-10:][::-1] if len(abs_errors) > 0 else []
        }
    else:
        # For classification, analyze confidence vs correctness
        correct = y_true == y_pred
        wrong_indices = np.where(~correct)[0]
        
        error_analysis = {
            "wrong_predictions": wrong_indices.size,
            "error_rate": wrong_indices.size / len(y_true),
            "wrong_indices": wrong_indices,
            "correct": correct
        }
        
        # Add confidence analysis if probabilities available
        if y_pred_proba is not None:
            # For binary classification, get probability of predicted class
            if len(y_pred_proba.shape) == 2 and y_pred_proba.shape[1] == 2:
                # Binary classification
                pred_class_proba = np.where(y_pred == 1, y_pred_proba[:, 1], y_pred_proba[:, 0])
            elif len(y_pred_proba.shape) == 2:
                # Multi-class: get probability of predicted class
                pred_class_proba = y_pred_proba[np.arange(len(y_pred)), y_pred]
            else:
                # Already probabilities for predicted class
                pred_class_proba = y_pred_proba
            
            error_analysis.update({
                "pred_confidences": pred_class_proba,
                "avg_confidence": pred_class_proba.mean(),
                "wrong_high_confidence": ((~correct) & (pred_class_proba > 0.8)).sum() if wrong_indices.size > 0 else 0,
                "correct_low_confidence": (correct & (pred_class_proba < 0.5)).sum() if correct.any() else 0
            })
        
        return error_analysis

def generate_model_recommendations(metrics_df: pd.DataFrame, problem_type: str) -> Dict[str, pd.Series]:
    """Generate top model recommendations based on different criteria"""
    recommendations = {}
    
    if metrics_df.empty:
        return recommendations
    
    # Filter out models with NaN reliability scores
    valid_df = metrics_df.dropna(subset=['Reliability Score'])
    
    if valid_df.empty:
        return recommendations
    
    # 1. Best Overall Model (highest risk-adjusted reliability)
    recommendations["best_overall"] = valid_df.sort_values(
        "Reliability Score", ascending=False
    ).iloc[0]
    
    # 2. Fastest Model (lowest training time)
    recommendations["best_fast"] = valid_df.sort_values(
        "Training Time (s)"
    ).iloc[0]
    
    # 3. Most Stable Model (lowest CV std)
    if 'CV Std' in valid_df.columns:
        recommendations["most_stable"] = valid_df.sort_values(
            "CV Std"
        ).iloc[0]
    
    # 4. Metric-specific recommendations
    if problem_type == "classification":
        if 'Precision' in valid_df.columns:
            recommendations["best_precision"] = valid_df.sort_values(
                "Precision", ascending=False
            ).iloc[0]
        if 'Recall' in valid_df.columns:
            recommendations["best_recall"] = valid_df.sort_values(
                "Recall", ascending=False
            ).iloc[0]
        if "ROC_AUC" in valid_df.columns and not valid_df["ROC_AUC"].isna().all():
            recommendations["best_auc"] = valid_df.sort_values(
                "ROC_AUC", ascending=False
            ).iloc[0]
    else:  # regression
        if 'RMSE' in valid_df.columns:
            recommendations["lowest_error"] = valid_df.sort_values(
                "RMSE"
            ).iloc[0]
        if 'R2' in valid_df.columns:
            recommendations["best_r2"] = valid_df.sort_values(
                "R2", ascending=False
            ).iloc[0]
    
    return recommendations

# ============================================
# DATA LEAKAGE DETECTION
# ============================================

def detect_data_leakage(X: pd.DataFrame, y: pd.Series, feature_cols: List[str]) -> List[str]:
    """Detect potential data leakage issues"""
    warnings = []
    
    # 1. Check for high correlation with target
    if pd.api.types.is_numeric_dtype(y):
        for col in X.select_dtypes(include=[np.number]).columns:
            try:
                corr = abs(X[col].corr(y))
                if corr > 0.95:
                    warnings.append(f"⚠️ High correlation ({corr:.3f}) between '{col}' and target - possible leakage")
            except:
                pass
    
    # 2. Check for ID-like columns
    for col in feature_cols:
        n_unique = X[col].nunique()
        if n_unique > len(X) * 0.9 and n_unique > 100:  # More than 90% unique values
            warnings.append(f"⚠️ '{col}' has {n_unique} unique values ({n_unique/len(X)*100:.1f}%) - possible ID column")
    
    # 3. Check for date-like patterns
    date_keywords = ['date', 'time', 'timestamp', 'year', 'month', 'day']
    for col in feature_cols:
        if any(keyword in col.lower() for keyword in date_keywords):
            warnings.append(f"⚠️ '{col}' appears to be a date/time column - ensure proper feature engineering")
    
    # 4. Check for constant or near-constant features
    for col in feature_cols:
        if X[col].nunique() <= 1:
            warnings.append(f"⚠️ '{col}' is constant or near-constant")
    
    return warnings

# ============================================
# MODEL-AWARE SCALING DETECTION
# ============================================

def should_scale_features(model_name: str, scale_features: bool) -> bool:
    """Determine if features should be scaled for this model"""
    if not scale_features:
        return False
    
    # Models that don't benefit from scaling
    no_scale_models = [
        'Random Forest', 'Decision Tree', 'Gradient Boosting',
        'AdaBoost', 'XGBoost', 'LightGBM'
    ]
    
    # Models that need scaling
    need_scale_models = [
        'SVM', 'SVR', 'Neural Network', 'K-Neighbors', 'K-NeighborsRegressor',
        'Logistic Regression', 'Ridge', 'Lasso', 'ElasticNet'
    ]
    
    for no_scale in no_scale_models:
        if no_scale in model_name:
            return False
    
    for need_scale in need_scale_models:
        if need_scale in model_name:
            return True
    
    # Default to scaling for unknown models
    return True

# ============================================
# CROSS-VALIDATION SCORING ADAPTATION
# ============================================

def get_cv_scoring(problem_type: str, y: pd.Series = None) -> str:
    """Get appropriate scoring metric for cross-validation"""
    if problem_type == "classification":
        if y is not None:
            n_classes = y.nunique()
            class_dist = y.value_counts()
            
            # Check for imbalance
            if n_classes == 2:
                return 'roc_auc'
            elif class_dist.iloc[0] > len(y) * 0.8:  # Severe imbalance
                return 'f1_weighted'
            else:
                return 'accuracy'
        return 'accuracy'
    else:  # regression
        return 'r2'

# ============================================
# CORE ML FUNCTIONS (UPDATED)
# ============================================

def validate_classification_data(X: pd.DataFrame, y: pd.Series) -> Tuple[bool, List[str]]:
    """Validate data for classification tasks"""
    warnings = []
    
    # Check class distribution
    class_counts = y.value_counts()
    total_samples = len(y)
    
    if len(class_counts) < 2:
        return False, ["Need at least 2 classes for classification"]
    
    # Check for severe class imbalance
    majority_pct = (class_counts.iloc[0] / total_samples) * 100
    if majority_pct > 90:
        warnings.append(f"Severe class imbalance: {majority_pct:.1f}% in majority class")
    
    # Check minimum samples per class
    min_samples = class_counts.min()
    if min_samples < 10:
        warnings.append(f"Very small class: only {min_samples} samples")
    
    # Check feature variance
    constant_features = X.columns[X.nunique() <= 1]
    if len(constant_features) > 0:
        warnings.append(f"Constant features found: {', '.join(constant_features[:3])}")
    
    return True, warnings

def validate_regression_data(X: pd.DataFrame, y: pd.Series) -> Tuple[bool, List[str]]:
    """Validate data for regression tasks"""
    warnings = []
    
    # Check target variance
    target_var = y.var()
    if target_var < 1e-10:
        return False, ["Target variable is constant"]
    
    # Check target distribution
    y_skew = y.skew()
    if abs(y_skew) > 3:
        warnings.append(f"Highly skewed target (skewness: {y_skew:.2f})")
    
    # Check for outliers in target
    q1, q3 = y.quantile(0.25), y.quantile(0.75)
    iqr = q3 - q1
    outliers = ((y < (q1 - 1.5 * iqr)) | (y > (q3 + 1.5 * iqr))).sum()
    if outliers / len(y) > 0.05:
        warnings.append(f"{outliers} outliers in target ({outliers/len(y)*100:.1f}%)")
    
    # Check feature variance
    constant_features = X.columns[X.nunique() <= 1]
    if len(constant_features) > 0:
        warnings.append(f"Constant features found: {', '.join(constant_features[:3])}")
    
    return True, warnings

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name, problem_type, cv_folds=5):
    """Train model and return comprehensive evaluation metrics with error analysis"""
    try:
        # Debug: Check data types
        # st.write(f"DEBUG: Training {model_name}")
        # st.write(f"X_train type: {type(X_train)}, shape: {X_train.shape if hasattr(X_train, 'shape') else 'No shape'}")
        # st.write(f"y_train type: {type(y_train)}, shape: {y_train.shape if hasattr(y_train, 'shape') else 'No shape'}")
        
        # Train model
        start_time = datetime.now()
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Get probabilities if available (for classification error analysis)
        y_pred_proba = None
        if problem_type == 'classification' and hasattr(model, 'predict_proba'):
            try:
                y_pred_proba = model.predict_proba(X_test)
            except:
                y_pred_proba = None
        
        # Error analysis
        error_analysis = compute_error_analysis(y_test, y_pred, problem_type, y_pred_proba)
        
        results = {
            'model_name': model_name,
            'model_object': model,
            'training_time': training_time,
            'predictions': y_pred,
            'y_test': y_test,
            'error_analysis': error_analysis
        }
        
        if y_pred_proba is not None:
            results['y_pred_proba'] = y_pred_proba
        
        if problem_type == 'classification':
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
            from sklearn.model_selection import cross_val_score
            
            # Classification metrics
            results['accuracy'] = accuracy_score(y_test, y_pred)
            results['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            results['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            results['f1'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Confusion matrix
            results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
            
            # ROC AUC (for binary classification)
            if len(np.unique(y_test)) == 2 and y_pred_proba is not None:
                try:
                    results['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
                except:
                    results['roc_auc'] = np.nan
            
            # Cross-validation with adaptive scoring
            cv_scoring = get_cv_scoring('classification', y_train)
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=cv_scoring)
                results['cv_mean'] = cv_scores.mean()
                results['cv_std'] = cv_scores.std()
            except Exception as cv_e:
                # st.warning(f"CV failed for {model_name}: {str(cv_e)}")
                results['cv_mean'] = np.nan
                results['cv_std'] = np.nan
            
        else:  # regression
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            from sklearn.model_selection import cross_val_score
            
            # Regression metrics
            results['r2'] = r2_score(y_test, y_pred)
            results['mse'] = mean_squared_error(y_test, y_pred)
            results['rmse'] = np.sqrt(results['mse'])
            results['mae'] = mean_absolute_error(y_test, y_pred)
            
            # SAFE MAPE calculation (only when no near-zero values)
            y_test_abs = np.abs(y_test)
            safe_mask = y_test_abs > 1e-10
            
            if safe_mask.any() and safe_mask.sum() > len(y_test) * 0.8:  # At least 80% safe values
                try:
                    mape = np.mean(np.abs((y_test[safe_mask] - y_pred[safe_mask]) / y_test[safe_mask])) * 100
                    results['mape'] = mape
                except:
                    results['mape'] = np.nan
            else:
                results['mape'] = np.nan
            
            # Cross-validation with R2 scoring
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2')
                results['cv_mean'] = cv_scores.mean()
                results['cv_std'] = cv_scores.std()
            except Exception as cv_e:
                # st.warning(f"CV failed for {model_name}: {str(cv_e)}")
                results['cv_mean'] = np.nan
                results['cv_std'] = np.nan
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            results['feature_importance'] = model.feature_importances_
        elif hasattr(model, 'coef_'):
            results['feature_importance'] = np.abs(model.coef_).flatten()
        
        return results
        
    except Exception as e:
        st.warning(f"⚠️ Model {model_name} failed: {str(e)}")
        # Debug: Show more details about the error
        # st.write(f"Error details: {type(e).__name__}")
        # st.write(f"X_train type: {type(X_train)}")
        # st.write(f"y_train type: {type(y_train)}")
        return {'model_name': model_name, 'error': str(e)}

def get_classification_models():
    """Return classification models dictionary"""
    models = {}
    
    if SKLEARN_AVAILABLE:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.svm import SVC
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.neural_network import MLPClassifier
        
        models.update({
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'SVM': SVC(probability=True, random_state=42),
            'Naive Bayes': GaussianNB(),
            'K-Neighbors': KNeighborsClassifier(n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
            'AdaBoost': AdaBoostClassifier(random_state=42, n_estimators=50),
            'Neural Network': MLPClassifier(max_iter=1000, random_state=42, hidden_layer_sizes=(100,))
        })
    
    if XGBOOST_AVAILABLE:
        try:
            import xgboost as xgb
            models['XGBoost'] = xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')
        except:
            pass
    
    if LIGHTGBM_AVAILABLE:
        try:
            import lightgbm as lgb
            models['LightGBM'] = lgb.LGBMClassifier(random_state=42, n_estimators=100, n_jobs=-1)
        except:
            pass
    
    return models

def get_regression_models():
    """Return regression models dictionary"""
    models = {}
    
    if SKLEARN_AVAILABLE:
        from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.svm import SVR
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.neural_network import MLPRegressor
        
        models.update({
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(random_state=42, alpha=1.0),
            'Lasso Regression': Lasso(random_state=42, alpha=0.1),
            'ElasticNet': ElasticNet(random_state=42, alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
            'SVR': SVR(),
            'K-Neighbors': KNeighborsRegressor(n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42, n_estimators=100),
            'AdaBoost': AdaBoostRegressor(random_state=42, n_estimators=50),
            'Neural Network': MLPRegressor(max_iter=1000, random_state=42, hidden_layer_sizes=(100,))
        })
    
    if XGBOOST_AVAILABLE:
        try:
            import xgboost as xgb
            models['XGBoost'] = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        except:
            pass
    
    if LIGHTGBM_AVAILABLE:
        try:
            import lightgbm as lgb
            models['LightGBM'] = lgb.LGBMRegressor(random_state=42, n_estimators=100, n_jobs=-1)
        except:
            pass
    
    return models

# ============================================
# MAIN STEP9_ML FUNCTION (FIXED FOR DATA TYPE ISSUES)
# ============================================

def step9_ml():
    st.header("Step 9 · ML & AutoML")
    st.markdown(f"**Step 9 of 12**")
    
    # ============================================
    # CRITICAL: SESSION STATE INITIALIZATION
    # ============================================
    if "model_results" not in st.session_state:
        st.session_state.model_results = []
    
    if "trained_models" not in st.session_state:
        st.session_state.trained_models = {}
    
    if "training_done" not in st.session_state:
        st.session_state.training_done = False
    
    if "current_problem_type" not in st.session_state:
        st.session_state.current_problem_type = None
    
    if st.session_state.df is None:
        st.warning("Please upload a dataset first!")
        return
    
    df = st.session_state.df
    
    st.markdown("**Train multiple models, compare performance, and get AI-powered insights**")
    
    if not SKLEARN_AVAILABLE:
        graceful_fallback('scikit-learn', 'Machine Learning')
        return
    
    # Get feature engineering columns (or all if not set)
    if st.session_state.feature_engineering_columns['selected']:
        available_features = [col for col in st.session_state.feature_engineering_columns['selected'] if col in df.columns]
        st.info(f"💡 Using {len(available_features)} features from Feature Engineering step")
    else:
        available_features = df.columns.tolist()
        st.warning("⚠️ No features selected in Feature Engineering. Using all columns.")
    
    # ============================================
    # PROBLEM SETUP WITH AUTO-DETECTION
    # ============================================
    st.subheader("🎯 Problem Setup")
    
    # Auto-detect problem type based on target
    col1, col2 = st.columns(2)
    
    with col1:
        # Get target column first
        target_col = st.selectbox(
            "Select Target Column:",
            [col for col in df.columns if col in available_features or col not in available_features],
            help="The column you want to predict"
        )
    
    if not target_col:
        st.warning("Please select a target column")
        return
    
    y = df[target_col]
    
    # Auto-detect problem type
    detected_type, detection_reason, detection_info = detect_problem_type(y)
    
    with col2:
        # Problem type selection with auto-suggestion
        if detected_type != "unknown":
            default_index = 0 if detected_type == "classification" else 1
            problem_type = st.radio(
                "Problem Type:",
                ['Classification', 'Regression'],
                index=default_index,
                help=f"Auto-detected as {detected_type}: {detection_reason}"
            )
        else:
            problem_type = st.radio(
                "Problem Type:",
                ['Classification', 'Regression'],
                help="Classification for categorical targets, Regression for continuous targets"
            )
    
    # Store current problem type
    st.session_state.current_problem_type = problem_type.lower()
    
    # Immediate validation of problem-target compatibility
    is_compatible, compatibility_warnings = validate_problem_target_compatibility(problem_type.lower(), y)
    
    if not is_compatible:
        st.error("❌ Problem-target compatibility failed")
        for warning in compatibility_warnings:
            st.error(warning)
        
        # Suggest alternative
        if detected_type != "unknown" and detected_type != problem_type.lower():
            st.info(f"💡 Suggested: Try {detected_type.capitalize()} instead")
        
        return
    
    for warning in compatibility_warnings:
        st.warning(warning)
    
    # ============================================
    # FEATURE SELECTION WITH LEAKAGE DETECTION
    # ============================================
    feature_cols = st.multiselect(
        "Select Feature Columns (X):",
        [col for col in available_features if col != target_col],
        default=[col for col in available_features if col != target_col][:min(10, len(available_features)-1)],
        help="Columns to use for prediction"
    )
    
    if not feature_cols:
        st.warning("Please select at least one feature column")
        return
    
    X = df[feature_cols]
    
    # Data Leakage Detection
    leakage_warnings = detect_data_leakage(X, y, feature_cols)
    if leakage_warnings:
        with st.expander("⚠️ Data Leakage Warnings", expanded=True):
            for warning in leakage_warnings:
                st.warning(warning)
    
    # ============================================
    # DATA VALIDATION
    # ============================================
    st.subheader("🔍 Data Validation")
    
    # Check for missing values
    if X.isnull().any().any() or y.isnull().any():
        st.error("❌ Missing values detected! Please clean your data first in the Missing Values step.")
        return
    
    # Check for non-numeric features
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        st.error(f"❌ Non-numeric features detected: {non_numeric}. Please encode them in Feature Engineering step.")
        return
    
    # Run appropriate validation
    if problem_type.lower() == 'classification':
        is_valid, warnings = validate_classification_data(X, y)
        validation_type = "Classification"
    else:
        is_valid, warnings = validate_regression_data(X, y)
        validation_type = "Regression"
    
    if not is_valid:
        st.error(f"❌ {validation_type} data validation failed")
        for warning in warnings:
            st.error(warning)
        return
    
    for warning in warnings:
        st.warning(warning)
    
    # ============================================
    # AI DATA ANALYSIS
    # ============================================
    with st.expander("🔍 AI Data Analysis", expanded=True):
        st.markdown("### Dataset Characteristics")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Samples", f"{len(X):,}")
        with c2:
            st.metric("Features", len(feature_cols))
        with c3:
            if problem_type.lower() == 'classification':
                st.metric("Classes", y.nunique())
            else:
                st.metric("Target Range", f"{y.min():.2f} - {y.max():.2f}")
        with c4:
            st.metric("Memory", f"{X.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Target distribution
        st.markdown("### 🎯 Target Distribution")
        if problem_type.lower() == 'classification':
            fig = px.bar(y.value_counts().reset_index(), 
                        x='index', y=y.name,
                        title=f"Distribution of {target_col}")
        else:
            fig = px.histogram(y, title=f"Distribution of {target_col}")
        st.plotly_chart(fig, use_container_width=True)
        
        # AI Recommendations
        st.markdown("### 🤖 AI Recommendations")
        recommendations = []
        
        # Dataset size recommendations
        if len(X) < 100:
            recommendations.append("⚠️ Small dataset (<100 samples). Consider: KNN, Naive Bayes, or simple models")
        elif len(X) < 1000:
            recommendations.append("👍 Medium dataset. Recommended: Random Forest, SVM, or ensemble methods")
        else:
            recommendations.append("🌟 Large dataset. All models will work well. Try XGBoost/LightGBM for best performance")
        
        # Feature recommendations
        if len(feature_cols) > 50:
            recommendations.append("⚠️ Many features (>50). Consider feature selection or dimensionality reduction")
        
        # Class imbalance check
        if problem_type.lower() == 'classification':
            class_dist = y.value_counts(normalize=True)
            if class_dist.iloc[0] > 0.8:
                recommendations.append(f"⚠️ Severe class imbalance ({class_dist.iloc[0]*100:.1f}% in majority class)")
                recommendations.append("💡 Consider: SMOTE, class weights, or ensemble methods")
        
        for rec in recommendations:
            if "⚠️" in rec:
                st.warning(rec)
            elif "🌟" in rec:
                st.success(rec)
            else:
                st.info(rec)
    
    # ============================================
    # TRAIN/TEST SPLIT CONFIGURATION
    # ============================================
    st.subheader("🔀 Train/Test Split")
    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Test Size (%)", 10, 40, 20, 5) / 100
    with col2:
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
    with col3:
        random_state = st.number_input("Random State", 0, 100, 42)
    
    # ============================================
    # MODEL SELECTION WITH PROBLEM-TYPE FILTERING
    # ============================================
    model_selection_mode = st.radio(
        "Selection Mode:",
        ['🤖 AutoML (Try All Models)', '🎯 Manual Selection', '⚡ Quick Test (Top 3)'],
        horizontal=True
    )
    
    selected_models = []
    
    # Get appropriate models based on problem type
    if problem_type.lower() == 'classification':
        all_models_dict = get_classification_models()
        available_models = list(all_models_dict.keys())
    else:
        all_models_dict = get_regression_models()
        available_models = list(all_models_dict.keys())
    
    if model_selection_mode == '🤖 AutoML (Try All Models)':
        st.info(f"🚀 AutoML will train and compare ALL available {problem_type.lower()} models")
        selected_models = available_models
        st.write(f"**Will train {len(selected_models)} models:** {', '.join(selected_models)}")
    
    elif model_selection_mode == '⚡ Quick Test (Top 3)':
        st.info("⚡ Quick test with 3 most reliable models")
        if problem_type.lower() == 'classification':
            selected_models = ['Logistic Regression', 'Random Forest', 'XGBoost' if 'XGBoost' in available_models else 'Gradient Boosting']
        else:
            selected_models = ['Linear Regression', 'Random Forest', 'XGBoost' if 'XGBoost' in available_models else 'Gradient Boosting']
        selected_models = [m for m in selected_models if m in available_models]
        st.write(f"**Models:** {', '.join(selected_models)}")
    
    else:  # Manual Selection
        st.markdown(f"**Select {problem_type.lower()} models to train:**")
        
        # Show only models compatible with problem type
        cols = st.columns(3)
        for idx, model_name in enumerate(available_models):
            with cols[idx % 3]:
                if st.checkbox(model_name, value=True, key=f"model_{model_name}"):
                    selected_models.append(model_name)
    
    if not selected_models:
        st.warning("Please select at least one model")
        return
    
    # ============================================
    # ADVANCED OPTIONS WITH MODEL-AWARE SCALING
    # ============================================
    with st.expander("⚙️ Advanced Options"):
        col1, col2 = st.columns(2)
        with col1:
            scale_features = st.checkbox("Scale Features (Auto-disabled for tree models)", value=True)
            hyperparameter_tuning = st.checkbox("Enable Hyperparameter Tuning (Slower)", value=False)
        with col2:
            ensemble_stacking = st.checkbox("Create Ensemble Stack (Advanced)", value=False)
            save_models = st.checkbox("Save Trained Models", value=True)
    
    # ============================================
    # TRAIN MODELS BUTTON (FIXED FOR DATA TYPE ISSUES)
    # ============================================
    st.divider()
    
    # Clear previous results if starting new training
    if st.button("🚀 Train Models", type="primary", use_container_width=True):
        # Reset session state for new training
        st.session_state.model_results = []
        st.session_state.training_done = False
        
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        # Prepare data - CONVERT TO NUMPY ARRAYS FOR TRAINING
        X_np = X.values if hasattr(X, 'values') else X
        y_np = y.values if hasattr(y, 'values') else y
        
        # Debug: Check data types
        # st.write(f"X_np type: {type(X_np)}, shape: {X_np.shape}")
        # st.write(f"y_np type: {type(y_np)}, shape: {y_np.shape}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_np, y_np, test_size=test_size, random_state=random_state
        )
        
        # Convert back to DataFrame for feature names if needed
        X_train_df = pd.DataFrame(X_train, columns=feature_cols) if hasattr(X, 'columns') else X_train
        X_test_df = pd.DataFrame(X_test, columns=feature_cols) if hasattr(X, 'columns') else X_test
        
        # Initialize scaler (will be used conditionally)
        scaler = StandardScaler() if scale_features else None
        
        # Train models sequentially with memory management
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, model_name in enumerate(selected_models):
            if model_name not in all_models_dict:
                st.warning(f"⚠️ Model '{model_name}' not available, skipping...")
                continue
            
            status_text.text(f"Training {model_name}... ({idx+1}/{len(selected_models)})")
            
            model = all_models_dict[model_name]
            
            # Apply model-aware scaling
            X_train_scaled = X_train_df.copy()
            X_test_scaled = X_test_df.copy()
            
            if scaler is not None and should_scale_features(model_name, scale_features):
                try:
                    # Convert to numpy for scaling
                    if hasattr(X_train_scaled, 'values'):
                        X_train_np = X_train_scaled.values
                        X_test_np = X_test_scaled.values
                    else:
                        X_train_np = X_train_scaled
                        X_test_np = X_test_scaled
                    
                    X_train_scaled_np = scaler.fit_transform(X_train_np)
                    X_test_scaled_np = scaler.transform(X_test_np)
                    
                    # Convert back to DataFrame if original was DataFrame
                    if hasattr(X_train_df, 'columns'):
                        X_train_scaled = pd.DataFrame(
                            X_train_scaled_np,
                            columns=feature_cols
                        )
                        X_test_scaled = pd.DataFrame(
                            X_test_scaled_np,
                            columns=feature_cols
                        )
                    else:
                        X_train_scaled = X_train_scaled_np
                        X_test_scaled = X_test_scaled_np
                        
                except Exception as e:
                    st.warning(f"⚠️ Scaling failed for {model_name}: {str(e)}")
                    # Continue without scaling
                    X_train_scaled = X_train_df
                    X_test_scaled = X_test_df
            
            # Hyperparameter tuning
            if hyperparameter_tuning and model_name in ['Random Forest', 'XGBoost', 'LightGBM', 'Gradient Boosting']:
                st.info(f"🔧 Tuning {model_name}...")
                from sklearn.model_selection import GridSearchCV
                
                param_grid = {}
                if 'Random Forest' in model_name:
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [None, 10, 20],
                        'min_samples_split': [2, 5]
                    }
                elif 'XGBoost' in model_name or 'LightGBM' in model_name or 'Gradient Boosting' in model_name:
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 5, 7],
                        'learning_rate': [0.01, 0.1, 0.3]
                    }
                
                if param_grid:
                    scoring = get_cv_scoring(problem_type.lower(), y_train)
                    try:
                        grid_search = GridSearchCV(model, param_grid, cv=3, scoring=scoring, n_jobs=-1)
                        grid_search.fit(X_train_scaled, y_train)
                        model = grid_search.best_estimator_
                        st.success(f"✅ Best params: {grid_search.best_params_}")
                    except Exception as e:
                        st.warning(f"⚠️ Hyperparameter tuning failed for {model_name}: {str(e)}")
            
            # Train and evaluate
            try:
                # Ensure we're passing numpy arrays or DataFrames, not strings
                # st.write(f"DEBUG: X_train_scaled type: {type(X_train_scaled)}")
                # st.write(f"DEBUG: y_train type: {type(y_train)}")
                
                result = train_and_evaluate_model(
                    model, X_train_scaled, X_test_scaled, y_train, y_test,
                    model_name, problem_type.lower(), cv_folds
                )
                
                if 'error' not in result:
                    results.append(result)
                    
                    # Save model if requested
                    if save_models:
                        st.session_state.trained_models[model_name] = result['model_object']
                    
                    # Show quick result
                    if problem_type.lower() == 'classification':
                        st.success(f"✅ {model_name}: Accuracy = {result['accuracy']:.4f}, F1 = {result['f1']:.4f}")
                    else:
                        st.success(f"✅ {model_name}: R² = {result['r2']:.4f}, RMSE = {result['rmse']:.4f}")
                else:
                    st.warning(f"❌ {model_name}: {result['error']}")
                    
            except Exception as e:
                st.error(f"❌ {model_name}: Training failed - {str(e)}")
                import traceback
                st.error(f"Traceback: {traceback.format_exc()}")
            
            # Clear memory after each model
            gc.collect()
            progress_bar.progress((idx + 1) / len(selected_models))
        
        # ============================================
        # SET TRAINING COMPLETE FLAG
        # ============================================
        status_text.text("✅ Training complete!")
        
        st.session_state.model_results = results
        st.session_state.training_done = True
        
        successful_models = len([r for r in results if 'error' not in r])
        if successful_models > 0:
            st.balloons()
            st.success(f"🎉 Training finished! {successful_models} model(s) evaluated successfully.")
        else:
            st.error(f"❌ All models failed! Check data preprocessing and feature selection.")
        
        # Ensemble Stacking
        if ensemble_stacking and successful_models >= 3:
            st.info("🔗 Creating ensemble stack...")
            try:
                from sklearn.ensemble import StackingClassifier, StackingRegressor
                
                base_models = [(r['model_name'], r['model_object']) for r in results[:3] if 'error' not in r]
                if problem_type.lower() == 'classification':
                    from sklearn.linear_model import LogisticRegression
                    stack = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression())
                else:
                    from sklearn.linear_model import Ridge
                    stack = StackingRegressor(estimators=base_models, final_estimator=Ridge())
                
                stack_result = train_and_evaluate_model(
                    stack, X_train_scaled, X_test_scaled, y_train, y_test,
                    'Ensemble Stack', problem_type.lower(), cv_folds
                )
                results.append(stack_result)
                st.session_state.model_results = results
                st.success("✅ Ensemble model created!")
            except Exception as e:
                st.warning(f"Ensemble creation failed: {str(e)}")
    
    # ============================================
    # DISPLAY RESULTS (CRITICAL FIX FOR UI RENDERING)
    # ============================================
    
    # Check if we should show results from previous training
    show_results = (
        st.session_state.training_done and 
        st.session_state.current_problem_type == problem_type.lower() and
        len(st.session_state.model_results) > 0
    )
    
    if show_results:
        st.divider()
        
        # Filter out failed models
        valid_results = [r for r in st.session_state.model_results if 'error' not in r]
        
        if len(valid_results) == 0:
            st.error("❌ No models trained successfully. Check the warnings above for errors.")
            return
        
        # Step 1: Build canonical metrics table
        metrics_df = build_metrics_table(valid_results, problem_type.lower())
        
        if not metrics_df.empty:
            # Step 2: Add risk-adjusted reliability scores
            metrics_df = add_reliability_scores(metrics_df, problem_type.lower())
            
            # Step 3: Show Performance Summary Table
            st.subheader("📊 Model Performance Summary")
            
            # Create display version with rounded values
            display_df = metrics_df.copy()
            
            # Format numeric columns
            numeric_cols = display_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col in ["Training Time (s)"]:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}s")
                elif col in ["CV Mean", "CV Std", "Reliability Score"]:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}")
                elif col in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "R2"]:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
                elif col in ["RMSE", "MAE", "MAPE"]:
                    if col == "MAPE":
                        # Show MAPE only if not NaN
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%" if not pd.isna(x) else "N/A")
                    else:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
                elif col in ["FPR", "FNR", "Avg Confidence"]:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}")
            
            # Sort by Reliability Score (descending)
            sorted_display_df = display_df.sort_values("Reliability Score", ascending=False)
            
            # Display the table with all metrics
            st.dataframe(
                sorted_display_df,
                use_container_width=True,
                height=min(400, 50 + 40 * len(sorted_display_df))
            )
            
            # Step 4: Generate recommendations
            recommendations = generate_model_recommendations(metrics_df, problem_type.lower())
            
            # Step 5: Show Recommendation Cards
            if recommendations:
                st.subheader("🤖 AutoML Recommendations")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if "best_overall" in recommendations:
                        best = recommendations["best_overall"]
                        st.success(
                            f"""
                            🏆 **Best Overall Model**
                            **{best['Model']}**
                            
                            Reliability Score: {best['Reliability Score']:.3f}
                            Best balance of performance + stability.
                            """
                        )
                
                with col2:
                    if "best_fast" in recommendations:
                        fastest = recommendations["best_fast"]
                        st.info(
                            f"""
                            ⚡ **Fastest Model**
                            **{fastest['Model']}**
                            
                            Training Time: {fastest['Training Time (s)']:.2f}s  
                            Recommended for real-time / frequent retraining.
                            """
                        )
                
                with col3:
                    if problem_type.lower() == "classification":
                        if "best_precision" in recommendations:
                            precise = recommendations["best_precision"]
                            st.warning(
                                f"""
                                🎯 **Best for Precision**
                                **{precise['Model']}**
                                
                                Precision: {precise['Precision']:.3f}  
                                Use when false positives are costly.
                                """
                            )
                    else:
                        if "lowest_error" in recommendations:
                            low_error = recommendations["lowest_error"]
                            st.warning(
                                f"""
                                🎯 **Lowest Error Model**
                                **{low_error['Model']}**
                                
                                RMSE: {low_error['RMSE']:.3f}  
                                Use when prediction accuracy matters most.
                                """
                            )
            
            # ============================================
            # ERROR ANALYSIS SECTION
            # ============================================
            if valid_results:
                st.subheader("🔬 Error Analysis")
                
                # Model selection for detailed analysis
                valid_model_names = metrics_df['Model'].tolist()
                if valid_model_names:
                    selected_model_name = st.selectbox(
                        "Select model for detailed analysis:",
                        valid_model_names
                    )
                    
                    selected_result = next((r for r in valid_results 
                                          if r['model_name'] == selected_model_name), None)
                    
                    if selected_result:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Performance Visualization
                            if problem_type.lower() == 'classification':
                                metric_to_plot = 'Accuracy'
                                title = "Model Accuracy Comparison"
                            else:
                                metric_to_plot = 'R2'
                                title = "Model R² Comparison"
                            
                            fig = px.bar(metrics_df, x='Model', y=metric_to_plot,
                                        title=title,
                                        color=metric_to_plot, color_continuous_scale='Viridis')
                            fig.update_layout(xaxis_tickangle=-45)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Reliability Score comparison
                            fig = px.bar(metrics_df.sort_values('Reliability Score', ascending=False),
                                        x='Model', y='Reliability Score',
                                        title="Model Reliability Scores",
                                        color='Reliability Score', color_continuous_scale='RdYlGn')
                            fig.update_layout(xaxis_tickangle=-45)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed Error Analysis
                        if 'error_analysis' in selected_result:
                            error_stats = selected_result['error_analysis']
                            
                            if problem_type.lower() == 'regression':
                                st.markdown("### 📊 Residual Analysis")
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Residual distribution
                                    fig = px.histogram(
                                        x=error_stats['residuals'],
                                        nbins=50,
                                        title=f"Residual Distribution - {selected_model_name}",
                                        labels={'x': 'Residual (Actual - Predicted)'}
                                    )
                                    fig.add_vline(x=0, line_dash="dash", line_color="red")
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with col2:
                                    # Residuals vs Actual
                                    fig = px.scatter(
                                        x=selected_result['y_test'],
                                        y=error_stats['residuals'],
                                        title=f"Residuals vs Actual - {selected_model_name}",
                                        labels={'x': 'Actual', 'y': 'Residual'}
                                    )
                                    fig.add_hline(y=0, line_dash="dash", line_color="red")
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Error statistics
                                st.markdown("### 📈 Error Statistics")
                                err_col1, err_col2, err_col3, err_col4 = st.columns(4)
                                with err_col1:
                                    st.metric("Mean Abs Error", f"{error_stats['mean_abs_error']:.4f}")
                                with err_col2:
                                    st.metric("Median Abs Error", f"{error_stats['median_abs_error']:.4f}")
                                with err_col3:
                                    st.metric("90th %ile Error", f"{error_stats['pct_90_error']:.4f}")
                                with err_col4:
                                    if not pd.isna(error_stats['mean_pct_error']):
                                        st.metric("Mean % Error", f"{error_stats['mean_pct_error']:.1f}%")
                                
                                # Worst predictions
                                worst_idx = error_stats['worst_indices']
                                if len(worst_idx) > 0:
                                    st.markdown("### ⚠️ Worst Predictions (Top 10)")
                                    worst_df = pd.DataFrame({
                                        'Actual': selected_result['y_test'].iloc[worst_idx] if hasattr(selected_result['y_test'], 'iloc') else [selected_result['y_test'][i] for i in worst_idx],
                                        'Predicted': [selected_result['predictions'][i] for i in worst_idx],
                                        'Error': [error_stats['abs_errors'][i] for i in worst_idx],
                                        'Index': worst_idx
                                    })
                                    st.dataframe(worst_df, use_container_width=True)
                            
                            else:  # Classification error analysis
                                st.markdown("### 📊 Classification Error Analysis")
                                
                                if 'confusion_matrix' in selected_result:
                                    cm = selected_result['confusion_matrix']
                                    fig, ax = plt.subplots(figsize=(8, 6))
                                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                                    plt.title(f"Confusion Matrix - {selected_model_name}")
                                    plt.ylabel('True Label')
                                    plt.xlabel('Predicted Label')
                                    st.pyplot(fig)
                                    plt.close()
                                
                                # Confidence analysis if available
                                if 'y_pred_proba' in selected_result and error_stats.get('pred_confidences') is not None:
                                    st.markdown("### 🎯 Confidence Analysis")
                                    
                                    conf_col1, conf_col2, conf_col3 = st.columns(3)
                                    with conf_col1:
                                        st.metric("Avg Confidence", f"{error_stats.get('avg_confidence', 0):.3f}")
                                    with conf_col2:
                                        wrong_high = error_stats.get('wrong_high_confidence', 0)
                                        total_wrong = error_stats.get('wrong_predictions', 0)
                                        if total_wrong > 0:
                                            pct_high = (wrong_high / total_wrong) * 100
                                            st.metric("Wrong but Confident", f"{pct_high:.1f}%")
                                    with conf_col3:
                                        correct_low = error_stats.get('correct_low_confidence', 0)
                                        total_correct = len(selected_result['y_test']) - error_stats.get('wrong_predictions', 0)
                                        if total_correct > 0:
                                            pct_low = (correct_low / total_correct) * 100
                                            st.metric("Correct but Unsure", f"{pct_low:.1f}%")
                                    
                                    # Confidence distribution for wrong predictions
                                    if error_stats.get('wrong_indices') is not None and len(error_stats['wrong_indices']) > 0:
                                        wrong_confidences = error_stats['pred_confidences'][error_stats['wrong_indices']]
                                        fig = px.histogram(
                                            x=wrong_confidences,
                                            nbins=20,
                                            title="Confidence Distribution for Wrong Predictions",
                                            labels={'x': 'Confidence'},
                                            color_discrete_sequence=['red']
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Feature Importance
                        if 'feature_importance' in selected_result:
                            st.markdown("### 📊 Feature Importance")
                            importance_df = pd.DataFrame({
                                'Feature': feature_cols,
                                'Importance': selected_result['feature_importance']
                            }).sort_values('Importance', ascending=False)
                            
                            fig = px.bar(importance_df.head(15), x='Importance', y='Feature',
                                        orientation='h', title=f"Top 15 Features - {selected_model_name}")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Predictions vs Actual (Regression)
                        if problem_type.lower() == 'regression':
                            st.markdown("### 📉 Predictions vs Actual")
                            pred_df = pd.DataFrame({
                                'Actual': selected_result['y_test'],
                                'Predicted': selected_result['predictions']
                            })
                            fig = px.scatter(pred_df, x='Actual', y='Predicted',
                                            title=f"Predictions vs Actual - {selected_model_name}",
                                            trendline="ols")
                            fig.add_trace(go.Scatter(x=[selected_result['y_test'].min(), selected_result['y_test'].max()],
                                                    y=[selected_result['y_test'].min(), selected_result['y_test'].max()],
                                                    mode='lines', name='Perfect Prediction',
                                                    line=dict(color='red', dash='dash')))
                            st.plotly_chart(fig, use_container_width=True)
            
            # ============================================
            # EXPORT SECTION
            # ============================================
            st.divider()
            st.subheader("📥 Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Export predictions
                if st.button("📊 Export Predictions (CSV)"):
                    if selected_result and 'error' not in selected_result:
                        pred_export = pd.DataFrame({
                            'Actual': selected_result['y_test'],
                            f'{selected_model_name}_Predicted': selected_result['predictions']
                        })
                        csv = pred_export.to_csv(index=False)
                        st.download_button(
                            label="Download Predictions",
                            data=csv,
                            file_name=f"predictions_{selected_model_name}.csv",
                            mime="text/csv"
                        )
            
            with col2:
                # Export trained model
                if st.button("💾 Export Trained Model"):
                    if selected_result and 'error' not in selected_result:
                        model_bytes = io.BytesIO()
                        joblib.dump(selected_result['model_object'], model_bytes)
                        model_bytes.seek(0)
                        st.download_button(
                            label="Download Model (.pkl)",
                            data=model_bytes,
                            file_name=f"model_{selected_model_name}.pkl",
                            mime="application/octet-stream"
                        )
            
            with col3:
                # Export full report with recommendations
                if st.button("📄 Export Full Report"):
                    best_overall = recommendations.get("best_overall", metrics_df.iloc[0] if not metrics_df.empty else None)
                    
                    if best_overall is not None:
                        report = f"""
Machine Learning Model Report
=============================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PROBLEM SETUP
-------------
Problem Type: {problem_type}
Target: {target_col}
Features: {', '.join(feature_cols)}
CV Folds: {cv_folds}

MODEL PERFORMANCE SUMMARY
-------------------------
Total Models Trained: {len(metrics_df)}
"""
                        
                        # Add all model metrics
                        report += "\nDetailed Metrics:\n"
                        for _, row in metrics_df.sort_values('Reliability Score', ascending=False).iterrows():
                            report += f"\n{row['Model']}:\n"
                            report += f"  Reliability Score: {row['Reliability Score']:.3f}\n"
                            if problem_type.lower() == 'classification':
                                report += f"  Accuracy: {row['Accuracy']:.4f}\n"
                                report += f"  F1 Score: {row['F1']:.4f}\n"
                            else:
                                report += f"  R²: {row['R2']:.4f}\n"
                                report += f"  RMSE: {row['RMSE']:.4f}\n"
                            report += f"  Training Time: {row['Training Time (s)']:.2f}s\n"
                        
                        # Add recommendations
                        if recommendations:
                            report += f"""
AUTOML RECOMMENDATIONS
----------------------
"""
                            if "best_overall" in recommendations:
                                report += f"""🏆 Best Overall Model: {recommendations['best_overall']['Model']}
   Reliability Score: {recommendations['best_overall']['Reliability Score']:.3f}
   Recommended for: General purpose deployment
"""
                            if "best_fast" in recommendations:
                                report += f"""
⚡ Fastest Model: {recommendations['best_fast']['Model']}
   Training Time: {recommendations['best_fast']['Training Time (s)']:.2f}s
   Recommended for: Real-time applications, frequent retraining
"""
                        
                        st.download_button(
                            label="Download Report (.txt)",
                            data=report,
                            file_name=f"ml_automl_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
    
    # ============================================
    # MEMORY MANAGEMENT
    # ============================================
    with st.expander("🧠 Memory Management", expanded=False):
        st.markdown("### Clear Model Cache")
        if st.button("🗑️ Clear All Trained Models"):
            clear_model_cache()
            st.session_state.training_done = False  # Reset flag
            st.session_state.model_results = []  # Clear results
            st.rerun()
    
    # ============================================
    # NAVIGATION
    # ============================================
    st.markdown("---")
    if st.button("Go to Next Step", type="primary", use_container_width=True):
        st.session_state.current_step = min(12, st.session_state.current_step + 1)
        st.rerun()