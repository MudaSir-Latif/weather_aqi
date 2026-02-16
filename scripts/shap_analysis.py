"""
Standalone SHAP + LIME explainability analysis for AQI prediction models.

Loads features from Hopsworks Feature Store, trains a raw XGBoost booster,
then computes SHAP values and LIME explanations.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import shap
import joblib

from src.feature_store import FeatureStore
from src.config import ModelConfig

# ---------------------------------------------------------------------------
# 1. Load features from Hopsworks (fallback to local CSV)
# ---------------------------------------------------------------------------
print("=" * 60)
print("SHAP + LIME Explainability Analysis")
print("=" * 60)

config = ModelConfig()
df = pd.DataFrame()

try:
    fs = FeatureStore()
    if fs.fs:
        df = fs.read_feature_group()
        if not df.empty:
            print(f"[OK] Loaded {len(df)} records from Hopsworks feature group.")
        else:
            print("[WARN] Hopsworks returned empty DataFrame.")
    else:
        print("[WARN] Could not connect to Hopsworks.")
except Exception as e:
    print(f"[WARN] Hopsworks loading failed: {e}")

if df.empty:
    csv_path = Path("data/processed/engineered_features.csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"[OK] Loaded {len(df)} records from {csv_path}")
    else:
        raise RuntimeError("No features available from Hopsworks or CSV.")

# ---------------------------------------------------------------------------
# 2. Prepare features and target
# ---------------------------------------------------------------------------
target_col = "aqi"
drop_cols = [target_col, "time", "aqi_category"]
X = df.drop(columns=[c for c in drop_cols if c in df.columns])
# Ensure all columns are numeric
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
y = pd.to_numeric(df[target_col], errors="coerce").fillna(0)

print(f"Features shape: {X.shape}  |  Target shape: {y.shape}")
print(f"Feature dtypes:\n{X.dtypes.value_counts()}")

# ---------------------------------------------------------------------------
# 3. Train XGBRegressor for SHAP analysis
# ---------------------------------------------------------------------------
from xgboost import XGBRegressor
import re as _re

xgb_model = XGBRegressor(
    objective="reg:squarederror",
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42,
    base_score=0.5,
)
xgb_model.fit(X, y)
print("[OK] Trained XGBRegressor for SHAP analysis.")


# Monkey-patch SHAP's XGBTreeModelLoader to handle bracket-wrapped floats
# that newer XGBoost versions emit (e.g. "[5E-1]" instead of "0.5").
_bracket_float_re = _re.compile(r'^\[(.+)\]$')

_orig_init = shap.explainers._tree.XGBTreeModelLoader.__init__

def _patched_xgb_loader_init(self, xgb_model_obj):
    """Wrap original __init__ – unwrap bracket notation in learner_model_param."""
    import json as _json
    try:
        cfg = _json.loads(xgb_model_obj.save_config())
        lmp = cfg.get("learner", {}).get("learner_model_param", {})
        for k, v in lmp.items():
            m = _bracket_float_re.match(str(v))
            if m:
                lmp[k] = str(float(m.group(1)))
        xgb_model_obj.load_config(_json.dumps(cfg))
    except Exception:
        pass
    _orig_init(self, xgb_model_obj)

shap.explainers._tree.XGBTreeModelLoader.__init__ = _patched_xgb_loader_init

# ---------------------------------------------------------------------------
# 4. SHAP analysis
# ---------------------------------------------------------------------------
print("\n--- SHAP Analysis ---")
X_shap = X  # default: use full X for SHAP
try:
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X)
except Exception as _tree_err:
    print(f"[WARN] TreeExplainer failed ({_tree_err}), using permutation-based Explainer")
    explainer = shap.Explainer(xgb_model.predict, X.iloc[:100])
    sv = explainer(X.iloc[:200])
    shap_values = sv.values
    X_shap = X.iloc[:200]

if isinstance(shap_values, list):
    shap_values = shap_values[0]
shap_values = np.array(shap_values)
print(f"[OK] SHAP values computed. Shape: {shap_values.shape}")

# Save SHAP values
out_dir = Path("data/processed")
out_dir.mkdir(parents=True, exist_ok=True)

shap_df = pd.DataFrame(shap_values, columns=X_shap.columns)
shap_df.to_csv(out_dir / "shap_values.csv", index=False)
print(f"[OK] SHAP values saved to {out_dir / 'shap_values.csv'}")

# Feature importance from SHAP (mean absolute SHAP value)
shap_importance = pd.DataFrame({
    "feature": X.columns,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0),
}).sort_values("mean_abs_shap", ascending=False)

print("\nTop 15 features by SHAP importance:")
print(shap_importance.head(15).to_string(index=False))

shap_importance.to_csv(out_dir / "shap_feature_importance.csv", index=False)
print(f"[OK] SHAP feature importance saved to {out_dir / 'shap_feature_importance.csv'}")

# Save SHAP summary plot
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_shap, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary_plot.png", dpi=150)
    plt.close()
    print(f"[OK] SHAP summary plot saved to {out_dir / 'shap_summary_plot.png'}")
except Exception as e:
    print(f"[WARN] Could not generate SHAP plot: {e}")

# ---------------------------------------------------------------------------
# 5. LIME analysis
# ---------------------------------------------------------------------------
print("\n--- LIME Analysis ---")
try:
    import lime
    import lime.lime_tabular

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X.values,
        feature_names=list(X.columns),
        mode="regression",
        random_state=42,
    )

    # Explain a few representative samples (median-AQI, high-AQI, low-AQI)
    sample_indices = {
        "median_aqi": int((y - y.median()).abs().idxmin()),
        "high_aqi": int(y.idxmax()),
        "low_aqi": int(y.idxmin()),
    }

    def booster_predict(data: np.ndarray) -> np.ndarray:
        return xgb_model.predict(data)

    lime_results = {}
    for label, idx in sample_indices.items():
        exp = lime_explainer.explain_instance(
            X.iloc[idx].values,
            booster_predict,
            num_features=10,
        )
        lime_results[label] = exp.as_list()
        print(f"\nLIME explanation for {label} (AQI={y.iloc[idx]:.1f}):")
        for feat, weight in exp.as_list():
            print(f"  {feat:>40s}  weight={weight:+.4f}")

        # Save LIME plot
        try:
            fig = exp.as_pyplot_figure()
            fig.tight_layout()
            fig.savefig(out_dir / f"lime_{label}.png", dpi=150)
            plt.close(fig)
            print(f"  [OK] LIME plot saved to {out_dir / f'lime_{label}.png'}")
        except Exception:
            pass

    # Save LIME results as CSV
    lime_rows = []
    for label, features in lime_results.items():
        for feat, weight in features:
            lime_rows.append({"sample": label, "feature_rule": feat, "weight": weight})
    lime_df = pd.DataFrame(lime_rows)
    lime_df.to_csv(out_dir / "lime_explanations.csv", index=False)
    print(f"\n[OK] LIME explanations saved to {out_dir / 'lime_explanations.csv'}")

except ImportError:
    print("[WARN] lime package not installed. Run: pip install lime")
except Exception as e:
    print(f"[WARN] LIME analysis failed: {e}")

print("\n" + "=" * 60)
print("Explainability analysis complete!")
print("=" * 60)
