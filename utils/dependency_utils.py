# Dependency checks
from typing import Dict
import streamlit as st

# Optional imports with fallbacks
try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

try:
    from sklearn.ensemble import (IsolationForest, RandomForestClassifier, RandomForestRegressor, 
                                   GradientBoostingClassifier, GradientBoostingRegressor, 
                                   AdaBoostClassifier, AdaBoostRegressor, VotingClassifier, VotingRegressor,
                                   StackingClassifier, StackingRegressor)
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, PolynomialFeatures
    from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier, KNeighborsRegressor
    from sklearn.linear_model import (LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet)
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                                 confusion_matrix, classification_report, roc_auc_score, roc_curve,
                                 mean_squared_error, mean_absolute_error, r2_score)
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from ydata_profiling import ProfileReport
    PROFILING_AVAILABLE = True
except ImportError:
    PROFILING_AVAILABLE = False

# ---------------------------------------------------------------------
# Dependency checking system
# ---------------------------------------------------------------------
def check_dependencies() -> Dict[str, bool]:
    """Check availability of all optional packages"""
    dependencies = {
        'scikit-learn': SKLEARN_AVAILABLE,
        'rapidfuzz': RAPIDFUZZ_AVAILABLE,
        'ydata_profiling': PROFILING_AVAILABLE,
        'xgboost': XGBOOST_AVAILABLE,
        'lightgbm': LIGHTGBM_AVAILABLE,
        'joblib': 'joblib' in globals() or 'joblib' in locals()
    }
    return dependencies

def graceful_fallback(package_name: str, feature_name: str) -> None:
    """Show user-friendly message for missing dependencies"""
    st.warning(f"⚠️ **{feature_name}** requires `{package_name}`")
    st.info(f"Install with: `pip install {package_name}`")
    if package_name == 'scikit-learn':
        st.info("Alternative: Use manual outlier detection methods")
    elif package_name == 'rapidfuzz':
        st.info("Alternative: Use exact text matching instead of fuzzy")
    elif package_name == 'ydata_profiling':
        st.info("Alternative: Use the built-in data overview features")
    elif package_name in ['xgboost', 'lightgbm']:
        st.info("Alternative: Use Random Forest or Gradient Boosting from scikit-learn")