import warnings
warnings.filterwarnings("ignore")

import io, os, ssl
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

st.set_page_config(
    page_title="Bank Telemarketing — ML Pipeline",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 Bank Telemarketing — Subscription Predictor")
st.caption("Automated ML pipeline: feature selection → model comparison → Optuna tuning → production artifact")
st.divider()


@st.cache_data(show_spinner=False)
def load_data(uploaded=None):
    if uploaded is not None:
        df = pd.read_csv(uploaded, sep=";")
        df["y"] = (df["y"] == "yes").astype(int)
        return df

    for path in ["bank-full.csv", "bankfull.csv", "bank_full.csv"]:
        if os.path.exists(path):
            df = pd.read_csv(path, sep=";")
            df["y"] = (df["y"] == "yes").astype(int)
            return df

    ssl._create_default_https_context = ssl._create_unverified_context
    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-full.csv",
        sep=";"
    )
    df["y"] = (df["y"] == "yes").astype(int)
    return df


def preprocess(df):
    df = df.copy()
    for col in df.columns:
        if col == "y":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "unknown")
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df


def variance_filter(X, threshold):
    sel = VarianceThreshold(threshold=threshold)
    sel.fit(X)
    mask = sel.get_support()
    return X.loc[:, mask].copy(), list(X.columns[~mask])


def correlation_filter(X, threshold):
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [c for c in upper.columns if (upper[c] > threshold).any()]
    return X.drop(columns=drop).copy(), drop, corr


MODELS = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       lambda: DecisionTreeClassifier(random_state=42),
    "Random Forest":       lambda: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting":   lambda: GradientBoostingClassifier(random_state=42),
    "KNN":                 lambda: KNeighborsClassifier(),
}


def run_cv(X, y, n_folds):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = {}
    for name, fn in MODELS.items():
        scores = cross_val_score(fn(), Xs, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        results[name] = {"mean": float(scores.mean()), "std": float(scores.std())}
    return results, scaler


def build_model(name, params):
    kw = dict(params)
    if name != "KNN":
        kw["random_state"] = 42
    if name == "Logistic Regression":
        kw["max_iter"] = 1000
    return {
        "Logistic Regression": LogisticRegression,
        "Decision Tree":       DecisionTreeClassifier,
        "Random Forest":       RandomForestClassifier,
        "Gradient Boosting":   GradientBoostingClassifier,
        "KNN":                 KNeighborsClassifier,
    }[name](**kw)


def tune(X, y, model_name, scaler, n_trials):
    Xs = scaler.transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        if model_name == "Logistic Regression":
            p = {"C": trial.suggest_float("C", 1e-4, 10.0, log=True)}
        elif model_name == "Random Forest":
            p = {
                "n_estimators":      trial.suggest_int("n_estimators", 50, 300),
                "max_depth":         trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
            }
        elif model_name == "Gradient Boosting":
            p = {
                "n_estimators":  trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "max_depth":     trial.suggest_int("max_depth", 3, 10),
                "subsample":     trial.suggest_float("subsample", 0.6, 1.0),
            }
        elif model_name == "KNN":
            p = {
                "n_neighbors": trial.suggest_int("n_neighbors", 3, 30),
                "weights":     trial.suggest_categorical("weights", ["uniform", "distance"]),
            }
        else:
            p = {
                "max_depth":         trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            }
        m = build_model(model_name, p)
        return float(cross_val_score(m, Xs, y, cv=cv, scoring="roc_auc", n_jobs=-1).mean())

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


# sidebar
with st.sidebar:
    st.header("Settings")

    uploaded = st.file_uploader("Upload bank-full.csv", type="csv")
    st.divider()

    st.subheader("Feature Selection")
    var_thresh  = st.slider("Variance threshold",    0.0,  0.5,  0.01, 0.005)
    corr_thresh = st.slider("Correlation threshold", 0.70, 0.99, 0.90, 0.01)
    st.divider()

    st.subheader("Cross-validation")
    n_folds = st.slider("Folds", 3, 10, 5)
    st.divider()

    st.subheader("Optuna")
    n_trials = st.slider("Trials", 10, 100, 30)
    st.divider()

    run_btn = st.button("Run pipeline", type="primary", use_container_width=True)


# tabs
tab1, tab2, tab3, tab4 = st.tabs(["Data", "Feature Selection", "Model Selection", "Results"])

# session state init
for k, v in dict(
    done=False, df_raw=None, X=None, y=None,
    X_final=None, var_removed=[], corr_removed=[], corr_mat=None,
    cv_results=None, scaler=None, best_name=None, study=None, n_folds=5,
).items():
    if k not in st.session_state:
        st.session_state[k] = v


if run_btn:
    with st.spinner("Loading data..."):
        df_raw = load_data(uploaded)
        df = preprocess(df_raw)
        X = df.drop(columns=["y"])
        y = df["y"]

    with st.spinner("Running feature selection..."):
        X_var, var_removed = variance_filter(X, var_thresh)
        X_final, corr_removed, corr_mat = correlation_filter(X_var, corr_thresh)

    with st.spinner("Cross-validating models..."):
        cv_results, scaler = run_cv(X_final, y, n_folds)

    best_name = max(cv_results, key=lambda k: cv_results[k]["mean"])

    with st.spinner(f"Tuning {best_name} with Optuna ({n_trials} trials)..."):
        study = tune(X_final, y, best_name, scaler, n_trials)

    st.session_state.update(
        done=True, df_raw=df_raw, X=X, y=y,
        X_final=X_final, var_removed=var_removed,
        corr_removed=corr_removed, corr_mat=corr_mat,
        cv_results=cv_results, scaler=scaler,
        best_name=best_name, study=study, n_folds=n_folds,
    )
    st.rerun()


# TAB 1 — DATA
with tab1:
    if not st.session_state["done"]:
        st.info("Configure the settings on the left and click **Run pipeline**.")
        st.markdown("""
The dataset is from a Portuguese bank's direct marketing campaign (phone calls, 2008–2010).
The goal is to predict whether a client will subscribe to a term deposit — a classic binary classification problem.

- 45,211 rows, 16 features
- Target: `y` (1 = subscribed, 0 = didn't)
- Class imbalance: ~12% positive class
        """)
    else:
        df_raw = st.session_state["df_raw"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{df_raw.shape[0]:,}")
        col2.metric("Features", df_raw.shape[1] - 1)
        col3.metric("Positive class", f"{(df_raw['y'] == 1).sum():,}")
        col4.metric("Missing values", int(df_raw.isnull().sum().sum()))

        vc = df_raw["y"].value_counts()
        pct = vc.get(1, 0) / vc.sum() * 100
        st.warning(
            f"Class imbalance: {pct:.1f}% positive ({vc.get(1,0):,} out of {vc.sum():,}). "
            "Using stratified k-fold CV and ROC-AUC to account for this."
        )
        st.info(
            "`duration` (call length in seconds) is a data leakage risk — you only know it after "
            "the call ends, so it can't inform the decision to call. Kept here for demo purposes."
        )

        st.subheader("Sample rows")
        st.dataframe(df_raw.head(50), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Column info")
            info = pd.DataFrame({
                "dtype":  df_raw.dtypes.astype(str),
                "nulls":  df_raw.isnull().sum(),
                "unique": df_raw.nunique(),
            })
            st.dataframe(info, use_container_width=True)
        with c2:
            st.subheader("Target distribution")
            vc2 = df_raw["y"].value_counts().reset_index()
            vc2.columns = ["subscribed", "count"]
            vc2["subscribed"] = vc2["subscribed"].map({1: "yes", 0: "no"})
            fig = px.bar(vc2, x="subscribed", y="count", color="subscribed",
                         color_discrete_map={"yes": "#5C6BC0", "no": "#ef9a9a"},
                         text="count")
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)


# TAB 2 — FEATURE SELECTION
with tab2:
    if st.session_state["done"]:
        X       = st.session_state["X"]
        X_final = st.session_state["X_final"]
        var_rem = st.session_state["var_removed"]
        cor_rem = st.session_state["corr_removed"]
        cor_mat = st.session_state["corr_mat"]

        st.subheader("Variance threshold")
        st.caption(
            "Removes features that are nearly constant across all rows. "
            "A constant feature adds no predictive signal."
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Before", X.shape[1])
        c2.metric("Dropped", len(var_rem))
        c3.metric("After", X.shape[1] - len(var_rem))

        if var_rem:
            st.warning(f"Dropped: {', '.join(var_rem)}")
        else:
            st.success("No features removed.")

        variances = X.var().reset_index()
        variances.columns = ["feature", "variance"]
        variances["dropped"] = variances["feature"].isin(var_rem)
        variances = variances.sort_values("variance")
        fig = px.bar(variances, x="variance", y="feature", orientation="h",
                     color="dropped",
                     color_discrete_map={False: "#5C6BC0", True: "#e53935"},
                     title=f"Feature variances (threshold = {var_thresh})")
        fig.add_vline(x=var_thresh, line_dash="dash", line_color="#e53935")
        fig.update_layout(height=max(300, 26 * X.shape[1]), margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Correlation filter")
        st.caption(
            "Removes one feature from each highly-correlated pair. "
            "Correlated features are redundant — they tell the model the same thing twice."
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Before", X.shape[1] - len(var_rem))
        c2.metric("Dropped", len(cor_rem))
        c3.metric("Final", X_final.shape[1])

        if cor_rem:
            st.warning(f"Dropped: {', '.join(cor_rem)}")
        else:
            st.success("No features removed.")

        fig2 = px.imshow(cor_mat, color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1, aspect="auto",
                         title=f"Correlation matrix (threshold = {corr_thresh})")
        fig2.update_layout(margin=dict(t=40))
        st.plotly_chart(fig2, use_container_width=True)

        st.success(f"{X.shape[1]} → {X_final.shape[1]} features kept.")


# TAB 3 — MODEL SELECTION
with tab3:
    if st.session_state["done"]:
        cv_res    = st.session_state["cv_results"]
        best_name = st.session_state["best_name"]
        study     = st.session_state["study"]
        n_folds   = st.session_state["n_folds"]
        baseline  = cv_res[best_name]["mean"]

        st.subheader("Cross-validated model comparison")
        st.caption(
            f"{n_folds}-fold stratified CV with ROC-AUC scoring. "
            "Accuracy is misleading here due to class imbalance — a model predicting 'no' every time "
            "scores 88% accuracy but catches zero subscribers."
        )

        res_df = pd.DataFrame([
            {"model": k, "roc_auc": v["mean"], "std": v["std"]}
            for k, v in cv_res.items()
        ]).sort_values("roc_auc", ascending=False)

        fig = px.bar(res_df, x="model", y="roc_auc", error_y="std",
                     color="roc_auc", color_continuous_scale="Blues",
                     title="ROC-AUC by model")
        fig.update_layout(yaxis_range=[0.5, 1.0], margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(res_df.round(4), use_container_width=True, hide_index=True)

        st.info(f"Best: **{best_name}** (ROC-AUC = {baseline:.4f})")

        st.divider()
        st.subheader(f"Optuna tuning — {best_name}")
        st.caption(
            "Bayesian hyperparameter search via Tree-structured Parzen Estimator (TPE). "
            "Smarter than grid search — uses past trial results to focus on promising regions."
        )

        tuned = study.best_value
        delta = tuned - baseline
        c1, c2, c3 = st.columns(3)
        c1.metric("Baseline", f"{baseline:.4f}")
        c2.metric("Tuned",    f"{tuned:.4f}")
        c3.metric("Delta",    f"{'+' if delta >= 0 else ''}{delta:.4f}", delta=f"{delta:.4f}")

        trials_df = pd.DataFrame([
            {"trial": t.number, "score": t.value}
            for t in study.trials if t.value is not None
        ])
        trials_df["best"] = trials_df["score"].cummax()

        c1, c2 = st.columns([2, 1])
        with c1:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=trials_df["trial"], y=trials_df["score"],
                                      mode="markers", name="trial",
                                      marker=dict(color="#9575CD", opacity=0.5, size=5)))
            fig2.add_trace(go.Scatter(x=trials_df["trial"], y=trials_df["best"],
                                      mode="lines", name="best so far",
                                      line=dict(color="#e53935", width=2)))
            fig2.update_layout(title="Optimization history",
                               xaxis_title="trial", yaxis_title="roc_auc",
                               margin=dict(t=30))
            st.plotly_chart(fig2, use_container_width=True)
        with c2:
            st.markdown("**Best params**")
            st.json(study.best_params)


# TAB 4 — RESULTS
with tab4:
    if st.session_state["done"]:
        best_name = st.session_state["best_name"]
        study     = st.session_state["study"]
        scaler    = st.session_state["scaler"]
        X_final   = st.session_state["X_final"]
        y         = st.session_state["y"]

        # train on full data
        final_model = build_model(best_name, study.best_params)
        final_model.fit(scaler.transform(X_final), y)

        # hold-out eval
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_final, y, test_size=0.2, random_state=42, stratify=y
        )
        eval_model = build_model(best_name, study.best_params)
        eval_model.fit(scaler.transform(X_tr), y_tr)
        y_pred = eval_model.predict(scaler.transform(X_te))
        y_prob = eval_model.predict_proba(scaler.transform(X_te))[:, 1]

        c1, c2, c3 = st.columns(3)
        c1.metric("Model", best_name)
        c2.metric("ROC-AUC (tuned)", f"{study.best_value:.4f}")
        c3.metric("Features", X_final.shape[1])

        if hasattr(final_model, "feature_importances_"):
            st.subheader("Feature importance")
            imp = pd.DataFrame({
                "feature":    X_final.columns,
                "importance": final_model.feature_importances_,
            }).sort_values("importance")
            fig = px.bar(imp, x="importance", y="feature", orientation="h",
                         color="importance", color_continuous_scale="Blues",
                         title="Feature importance (tuned model)")
            fig.update_layout(height=max(300, 26 * len(imp)), margin=dict(t=30))
            st.plotly_chart(fig, use_container_width=True)

        elif hasattr(final_model, "coef_"):
            st.subheader("Coefficients")
            coefs = final_model.coef_.flatten() if final_model.coef_.ndim > 1 else final_model.coef_
            coef_df = pd.DataFrame({
                "feature": X_final.columns, "coef": coefs
            }).sort_values("coef")
            fig = px.bar(coef_df, x="coef", y="feature", orientation="h",
                         color="coef", color_continuous_scale="RdBu", title="Coefficients")
            fig.update_layout(margin=dict(t=30))
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Evaluation on 20% hold-out set")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Confusion matrix**")
            cm = confusion_matrix(y_te, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
                               x=["Pred: No", "Pred: Yes"],
                               y=["Actual: No", "Actual: Yes"],
                               color_continuous_scale="Purples")
            fig_cm.update_layout(margin=dict(t=20))
            st.plotly_chart(fig_cm, use_container_width=True)

        with c2:
            st.markdown("**ROC curve**")
            fpr, tpr, _ = roc_curve(y_te, y_prob)
            auc = roc_auc_score(y_te, y_prob)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                         name=f"model (AUC={auc:.3f})",
                                         line=dict(color="#5C6BC0", width=2)))
            fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                         name="random",
                                         line=dict(dash="dash", color="gray")))
            fig_roc.update_layout(xaxis_title="FPR", yaxis_title="TPR",
                                  margin=dict(t=20))
            st.plotly_chart(fig_roc, use_container_width=True)

        st.markdown("**Classification report**")
        report = classification_report(y_te, y_pred, output_dict=True)
        rep_df = pd.DataFrame(report).T.round(3)
        rep_df = rep_df[rep_df.index.isin(["0", "1", "macro avg", "weighted avg"])]
        rep_df.index = ["no (0)", "yes (1)", "macro avg", "weighted avg"]
        st.dataframe(rep_df, use_container_width=True)

        st.divider()
        st.subheader("Export model")

        bundle = {
            "model":      final_model,
            "scaler":     scaler,
            "features":   list(X_final.columns),
            "model_name": best_name,
            "params":     study.best_params,
            "roc_auc":    study.best_value,
        }
        buf = io.BytesIO()
        joblib.dump(bundle, buf)
        buf.seek(0)

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download model (.joblib)",
                data=buf,
                file_name=f"bank_{best_name.replace(' ','_').lower()}.joblib",
                mime="application/octet-stream",
                use_container_width=True,
                type="primary",
            )
        with c2:
            st.download_button(
                "Download requirements.txt",
                data="streamlit>=1.32.0\npandas>=2.0.0\nnumpy>=1.24.0\nscikit-learn>=1.3.0\noptuna>=3.5.0\nplotly>=5.18.0\njoblib>=1.3.0\n",
                file_name="requirements.txt",
                mime="text/plain",
                use_container_width=True,
            )

        st.markdown("""
```python
import joblib, pandas as pd

bundle = joblib.load("bank_model.joblib")
X_new = pd.DataFrame(...)  # columns must match bundle["features"]
X_scaled = bundle["scaler"].transform(X_new[bundle["features"]])

preds = bundle["model"].predict(X_scaled)          # 0 or 1
probs = bundle["model"].predict_proba(X_scaled)[:, 1]  # subscription likelihood
```
""")