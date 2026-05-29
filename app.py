import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from datetime import datetime
from scipy import stats

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Self-Evolving Intelligence", layout="wide")
st.title("Self-Evolving Data Quality & ML Intelligence")
st.markdown("---")

# ----------------------------
# SESSION STATE
# ----------------------------
if 'evolution_log' not in st.session_state:
    st.session_state.evolution_log = []
if 'best_model_trained' not in st.session_state:
    st.session_state.best_model_trained = None
if 'custom_datasets' not in st.session_state:
    st.session_state.custom_datasets = {}
if 'training_columns' not in st.session_state:
    st.session_state.training_columns = None
if 'label_encoder' not in st.session_state:
    st.session_state.label_encoder = None
if 'drift_retrain_triggered' not in st.session_state:
    st.session_state.drift_retrain_triggered = False

# ----------------------------
# PIPELINE
# ----------------------------
def get_pipeline(model_obj, numeric_features, categorical_features):
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipe, numeric_features),
        ('cat', cat_pipe, categorical_features)
    ])
    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', model_obj)
    ])

# ----------------------------
# TRAIN MODELS HELPER
# ----------------------------
def train_models(X, y, num_cols, cat_cols, do_tune=False):
    """Train models and return best model info."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None
    )

    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=2000)
    }

    best_acc = -1
    best_model = None
    best_model_name = None
    best_preds = None
    best_metrics = None

    for model_name, model_obj in models.items():
        pipe = get_pipeline(model_obj, num_cols, cat_cols)

        if do_tune and model_name == "Random Forest":
            grid = GridSearchCV(
                pipe,
                {
                    'model__n_estimators': [50, 100, 200],
                    'model__max_depth': [None, 10, 20]
                },
                cv=3
            )
            grid.fit(X_train, y_train)
            pipe = grid.best_estimator_
        else:
            pipe.fit(X_train, y_train)

        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average='weighted', zero_division=0)
        recall = recall_score(y_test, preds, average='weighted', zero_division=0)
        f1 = f1_score(y_test, preds, average='weighted', zero_division=0)

        if acc > best_acc:
            best_acc = acc
            best_model = pipe
            best_model_name = model_name
            best_preds = preds
            best_metrics = (precision, recall, f1)

    return best_model, best_model_name, best_acc, best_preds, best_metrics, y_test

# ----------------------------
# DISPLAY MODEL RESULTS
# ----------------------------
def display_model_results(best_model_name, best_acc, best_preds, best_metrics, y_test, best_model, num_cols, label_encoder):
    st.success(f"Best Model: **{best_model_name}**")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{best_acc:.2%}")
    col2.metric("Precision", f"{best_metrics[0]:.2f}")
    col3.metric("Recall", f"{best_metrics[1]:.2f}")
    col4.metric("F1 Score", f"{best_metrics[2]:.2f}")

    # Confusion Matrix
    st.markdown("#### Confusion Matrix")
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_test, best_preds)

    labels = None
    if label_encoder is not None:
        try:
            labels = label_encoder.classes_
        except Exception:
            labels = None

    sns.heatmap(
        cm, annot=True, fmt="d", ax=ax,
        cmap="Blues",
        xticklabels=labels if labels is not None else "auto",
        yticklabels=labels if labels is not None else "auto"
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    plt.close(fig)

    # Feature Importance
    if best_model_name == "Random Forest":
        st.markdown("#### Top Feature Importances")
        try:
            feat_names = best_model.named_steps['preprocessor'].get_feature_names_out()
            feat_names = [n.split("__")[-1] for n in feat_names]
            importance = best_model.named_steps['model'].feature_importances_
            feat_df = pd.DataFrame({
                "Feature": feat_names,
                "Importance": importance
            }).sort_values(by="Importance", ascending=False).head(10)
            st.bar_chart(feat_df.set_index("Feature"))
        except Exception as e:
            st.warning(f"Could not extract feature importances: {e}")

# ----------------------------
# ANOMALY EXPLAINER
# ----------------------------
def explain_anomalies(X_original_num, X_full, anomaly_mask, num_cols):
    """Show why each anomaly was flagged."""
    anomaly_indices = np.where(anomaly_mask == -1)[0]
    if len(anomaly_indices) == 0:
        st.info("No anomalies detected.")
        return

    st.markdown(f"**{len(anomaly_indices)} anomalies detected** out of {len(X_original_num)} rows.")

    # Compute z-scores for numeric columns
    col_means = X_original_num[num_cols].mean()
    col_stds = X_original_num[num_cols].std().replace(0, 1e-9)

    anomaly_rows = X_full.iloc[anomaly_indices].copy()
    anomaly_num = X_original_num.iloc[anomaly_indices][num_cols]
    z_scores = ((anomaly_num - col_means) / col_stds).abs()

    # Build explanation table
    explanation_rows = []
    for idx_pos, orig_idx in enumerate(anomaly_indices):
        row_z = z_scores.iloc[idx_pos]
        top_features = row_z.nlargest(3)
        reasons = []
        for feat, z in top_features.items():
            actual_val = anomaly_num.iloc[idx_pos][feat]
            direction = "high" if anomaly_num.iloc[idx_pos][feat] > col_means[feat] else "low"
            reasons.append(f"{feat}={actual_val:.2f} (z={z:.1f}, {direction})")
        explanation_rows.append({
            "Row Index": orig_idx,
            "Top Reasons (by z-score)": " | ".join(reasons),
            "Max Z-Score": f"{row_z.max():.2f}"
        })

    exp_df = pd.DataFrame(explanation_rows)
    st.dataframe(exp_df, use_container_width=True)

    # Visual: z-score heatmap for anomalies
    st.markdown("**Z-Score Heatmap for Anomalous Rows**")
    z_display = z_scores.copy()
    z_display.index = [f"Row {i}" for i in anomaly_indices]

    top_cols = z_scores.mean().nlargest(min(10, len(num_cols))).index.tolist()
    z_display = z_display[top_cols]

    fig, ax = plt.subplots(figsize=(min(12, len(top_cols) * 1.2 + 2), max(3, len(anomaly_indices) * 0.4 + 1)))
    sns.heatmap(
        z_display, annot=True, fmt=".1f", cmap="RdYlGn_r",
        linewidths=0.5, ax=ax, cbar_kws={'label': 'Abs Z-Score'}
    )
    ax.set_title("Anomaly Z-Scores (higher = more unusual)")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Anomalous Row")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Distribution plots: show anomalies vs normal
    st.markdown("**Anomalous vs Normal — Distribution per Feature**")
    normal_mask = anomaly_mask == 1
    n_cols_plot = min(3, len(num_cols))
    top_drift_cols = z_scores.mean().nlargest(n_cols_plot).index.tolist()

    fig, axes = plt.subplots(1, n_cols_plot, figsize=(5 * n_cols_plot, 4))
    if n_cols_plot == 1:
        axes = [axes]

    for ax, col in zip(axes, top_drift_cols):
        normal_data = X_original_num[normal_mask][col].dropna()
        anomaly_data = X_original_num[anomaly_mask == -1][col].dropna()

        ax.hist(normal_data, bins=20, alpha=0.6, color='steelblue', label='Normal')
        ax.hist(anomaly_data, bins=10, alpha=0.8, color='crimson', label='Anomaly')
        ax.set_title(col)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.legend()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ----------------------------
# TABS
# ----------------------------
tab1, tab2, tab3 = st.tabs(["Evolution Engine", "Data Architect", "Evolution Log"])

# ----------------------------
# DATA ARCHITECT
# ----------------------------
with tab2:
    st.header("Dataset Builder")
    dataset_name_input = st.text_input("Dataset Name")

    if 'builder_df' not in st.session_state:
        st.session_state.builder_df = pd.DataFrame(
            [[25, "Male", 1], [30, "Female", 0]],
            columns=["Age", "Gender", "Target"]
        )

    edited_df = st.data_editor(st.session_state.builder_df, num_rows="dynamic")

    if st.button("💾 Save Dataset"):
        if dataset_name_input:
            st.session_state.custom_datasets[dataset_name_input] = edited_df
            st.success(f"Saved '{dataset_name_input}'")
        else:
            st.error("Enter a dataset name first")

# ----------------------------
# EVOLUTION LOG TAB
# ----------------------------
with tab3:
    st.header("📈 Evolution Progress Log")
    if st.session_state.evolution_log:
        log_df = pd.DataFrame(st.session_state.evolution_log)
        st.dataframe(log_df, use_container_width=True)

        st.markdown("#### Accuracy Over Runs")
        st.line_chart(log_df.set_index("Run")["Accuracy"])

        if "Drift_Retrain" in log_df.columns:
            retrain_runs = log_df[log_df["Drift_Retrain"] == True]
            if not retrain_runs.empty:
                st.info(f"🔄 Auto-retrain triggered {len(retrain_runs)} time(s) due to severe drift.")
    else:
        st.info("No evolution runs recorded yet. Run the Evolution Engine first.")

# ----------------------------
# EVOLUTION ENGINE
# ----------------------------
with tab1:

    with st.sidebar:
        st.header("⚙️ Control Panel")

        source = st.radio("Data Source", ["Upload CSV", "Custom Dataset"])

        df = None
        if source == "Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV", type="csv")
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
        else:
            if st.session_state.custom_datasets:
                selected_dataset = st.selectbox("Select Dataset", list(st.session_state.custom_datasets.keys()))
                df = st.session_state.custom_datasets[selected_dataset]
            else:
                st.warning("No datasets found. Build one in Data Architect.")

        use_iso = st.checkbox("Enable Anomaly Removal", True)
        show_anomaly_detail = st.checkbox("Show Anomaly Explanations", True)
        iso_contamination = st.slider("Anomaly Contamination %", 1, 20, 5) / 100

        run = st.button("⚙️ Run Evolution")
        tune = st.button("Run with Hyperparameter Tuning")

    if df is not None:
        drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'id']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        st.subheader("Data Quality Audit")
        target = st.selectbox("Select Target Variable", df.columns)
        df = df.dropna(subset=[target])

        col_a, col_b = st.columns(2)
        col_a.dataframe(df.head())

        missing_total = df.isnull().sum().sum()
        quality_score = 1 - (missing_total / df.size)
        col_b.metric("Data Quality Score", f"{quality_score:.2%}")
        col_b.metric("Total Rows", len(df))
        col_b.metric("Missing Values", int(missing_total))

        X = df.drop(columns=[target])
        y = df[target]
        X_original = X.copy()

        if len(df) < 10:
            st.error("Dataset too small — need at least 10 rows.")
        else:
            num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            cat_cols = X.select_dtypes(include=['object']).columns.tolist()

            if run or tune:
                with st.spinner("System evolving..."):

                    # ---- Anomaly Detection ----
                    anomaly_mask = np.ones(len(X))  # default: all normal

                    if use_iso and num_cols:
                        st.subheader("🔍 Anomaly Detection")
                        iso = IsolationForest(contamination=iso_contamination, random_state=42)
                        num_data_filled = X[num_cols].fillna(X[num_cols].median())
                        anomaly_mask = iso.fit_predict(num_data_filled)

                        if show_anomaly_detail:
                            with st.expander("🧬 Why Were These Rows Flagged as Anomalies?", expanded=True):
                                explain_anomalies(num_data_filled, X, anomaly_mask, num_cols)

                        X = X[anomaly_mask == 1]
                        y = y[anomaly_mask == 1]

                    # ---- Label Encoding ----
                    le = None
                    if y.dtype == 'object':
                        le = LabelEncoder()
                        y = le.fit_transform(y)
                        st.session_state.label_encoder = le
                    else:
                        st.session_state.label_encoder = None

                    # ---- Train ----
                    st.subheader("Model Training")
                    (
                        best_model, best_model_name, best_acc,
                        best_preds, best_metrics, y_test
                    ) = train_models(X, y, num_cols, cat_cols, do_tune=tune)

                    st.session_state.best_model_trained = best_model
                    st.session_state.training_columns = {
                        "num_cols": num_cols,
                        "cat_cols": cat_cols
                    }

                    # ---- Log evolution ----
                    run_number = len(st.session_state.evolution_log) + 1
                    st.session_state.evolution_log.append({
                        "Run": run_number,
                        "Timestamp": datetime.now().strftime("%H:%M:%S"),
                        "Model": best_model_name,
                        "Accuracy": round(best_acc, 4),
                        "Precision": round(best_metrics[0], 4),
                        "Recall": round(best_metrics[1], 4),
                        "F1": round(best_metrics[2], 4),
                        "Rows_After_Anomaly_Removal": int(X.shape[0]),
                        "Tuned": tune,
                        "Drift_Retrain": st.session_state.drift_retrain_triggered,
                    })
                    st.session_state.drift_retrain_triggered = False

                display_model_results(
                    best_model_name, best_acc, best_preds,
                    best_metrics, y_test, best_model, num_cols,
                    st.session_state.label_encoder
                )

            # ---- Prediction Interface ----
            if st.session_state.best_model_trained and st.session_state.training_columns:
                st.divider()
                st.subheader("Live Prediction")
                with st.expander("Enter values for a new prediction"):
                    tc = st.session_state.training_columns
                    input_data = {}

                    pred_col1, pred_col2 = st.columns(2)
                    all_input_cols = tc["num_cols"] + tc["cat_cols"]

                    for i, col in enumerate(all_input_cols):
                        target_col = pred_col1 if i % 2 == 0 else pred_col2
                        if col in tc["num_cols"]:
                            input_data[col] = target_col.number_input(f"{col}", value=0.0, key=f"pred_{col}")
                        else:
                            input_data[col] = target_col.text_input(f"{col}", value="", key=f"pred_{col}")

                    if st.button("Predict"):
                        try:
                            input_df = pd.DataFrame([input_data])
                            prediction = st.session_state.best_model_trained.predict(input_df)[0]
                            proba = None
                            try:
                                proba = st.session_state.best_model_trained.predict_proba(input_df)[0]
                            except Exception:
                                pass

                            le = st.session_state.label_encoder
                            if le is not None:
                                try:
                                    label = le.inverse_transform([int(prediction)])[0]
                                except Exception:
                                    label = prediction
                            else:
                                label = prediction

                            st.success(f"**Predicted Class: {label}**")
                            if proba is not None:
                                proba_df = pd.DataFrame({
                                    "Class": le.classes_ if le is not None else list(range(len(proba))),
                                    "Probability": proba
                                })
                                st.dataframe(proba_df)
                        except Exception as e:
                            st.error(f"Prediction error: {e}")

            # ---- Drift Detection ----
            if st.session_state.best_model_trained:
                st.divider()
                st.subheader("Multi-Dataset Drift Detection")

                drift_files = st.file_uploader(
                    "Upload New Datasets for Drift Comparison",
                    type="csv",
                    accept_multiple_files=True,
                    key="drift_uploader"
                )

                if drift_files:
                    for file_idx, drift_file in enumerate(drift_files):
                        st.markdown(f"### Dataset {file_idx + 1}: `{drift_file.name}`")
                        df_new = pd.read_csv(drift_file)

                        common_cols = [c for c in num_cols if c in df_new.columns]
                        if not common_cols:
                            st.error("No matching numeric columns found.")
                            continue

                        drift_results = []
                        severe_drift_found = False

                        for col in common_cols:
                            old_vals = X_original[col].dropna()
                            new_vals = df_new[col].dropna()

                            old_mean, new_mean = old_vals.mean(), new_vals.mean()
                            old_std, new_std = old_vals.std(), new_vals.std()

                            mean_shift = abs(old_mean - new_mean) / (abs(old_mean) + 1e-6)
                            std_shift = abs(old_std - new_std) / (abs(old_std) + 1e-6)
                            drift_score = (mean_shift + std_shift) / 2

                            # KS Test
                            ks_stat, ks_p = stats.ks_2samp(old_vals, new_vals)

                            if drift_score < 0.1 and ks_p > 0.05:
                                status = "✅ Stable"
                            elif drift_score < 0.25 and ks_p > 0.01:
                                status = "Moderate Drift"
                            else:
                                status = "Severe Drift"
                                severe_drift_found = True

                            drift_results.append({
                                "Feature": col,
                                "Drift Score": f"{drift_score * 100:.2f}%",
                                "KS Statistic": f"{ks_stat:.3f}",
                                "KS p-value": f"{ks_p:.4f}",
                                "Status": status
                            })

                        st.dataframe(pd.DataFrame(drift_results), use_container_width=True)

                        # Auto-retrain on severe drift
                        if severe_drift_found:
                            st.warning("Severe drift detected. Auto-retraining triggered...")

                            retrain_X = df_new.drop(columns=[target], errors='ignore')
                            if target in df_new.columns:
                                retrain_y = df_new[target].dropna()
                                retrain_X = retrain_X.loc[retrain_y.index]

                                if retrain_y.dtype == 'object':
                                    le_new = LabelEncoder()
                                    retrain_y = le_new.fit_transform(retrain_y)
                                    st.session_state.label_encoder = le_new

                                new_num = [c for c in num_cols if c in retrain_X.columns]
                                new_cat = [c for c in cat_cols if c in retrain_X.columns]

                                if len(retrain_X) >= 10 and new_num:
                                    (
                                        new_model, new_name, new_acc,
                                        new_preds, new_metrics, new_y_test
                                    ) = train_models(retrain_X, retrain_y, new_num, new_cat)

                                    st.session_state.best_model_trained = new_model
                                    st.session_state.drift_retrain_triggered = True

                                    run_number = len(st.session_state.evolution_log) + 1
                                    st.session_state.evolution_log.append({
                                        "Run": run_number,
                                        "Timestamp": datetime.now().strftime("%H:%M:%S"),
                                        "Model": new_name,
                                        "Accuracy": round(new_acc, 4),
                                        "Precision": round(new_metrics[0], 4),
                                        "Recall": round(new_metrics[1], 4),
                                        "F1": round(new_metrics[2], 4),
                                        "Rows_After_Anomaly_Removal": len(retrain_X),
                                        "Tuned": False,
                                        "Drift_Retrain": True,
                                    })

                                    st.success(f"Auto-retrain complete! New best model: **{new_name}** ({new_acc:.2%} accuracy)")
                                    display_model_results(
                                        new_name, new_acc, new_preds,
                                        new_metrics, new_y_test, new_model,
                                        new_num, st.session_state.label_encoder
                                    )
                                else:
                                    st.error("Drift dataset too small or missing numeric columns for retraining.")
                            else:
                                st.warning(f"Target column `{target}` not found in drift dataset — cannot auto-retrain.")

                        # Distribution plots per feature
                        feature_to_plot = st.selectbox(
                            f"Visualize feature distribution — Dataset {file_idx + 1}",
                            common_cols,
                            key=f"drift_feat_{file_idx}"
                        )

                        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

                        # KDE
                        sns.kdeplot(X_original[feature_to_plot].dropna(), label="Original", ax=axes[0], fill=True, alpha=0.4)
                        sns.kdeplot(df_new[feature_to_plot].dropna(), label="New", ax=axes[0], fill=True, alpha=0.4, color="crimson")
                        axes[0].set_title(f"{feature_to_plot} — Distribution Shift")
                        axes[0].legend()

                        # Box plot comparison
                        box_data = pd.DataFrame({
                            "Value": pd.concat([
                                X_original[feature_to_plot].dropna(),
                                df_new[feature_to_plot].dropna()
                            ], ignore_index=True),
                            "Dataset": (
                                ["Original"] * len(X_original[feature_to_plot].dropna()) +
                                ["New"] * len(df_new[feature_to_plot].dropna())
                            )
                        })
                        sns.boxplot(data=box_data, x="Dataset", y="Value", ax=axes[1], palette=["steelblue", "crimson"])
                        axes[1].set_title(f"{feature_to_plot} — Box Plot Comparison")

                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

    # Evolution log mini-preview at bottom
    if st.session_state.evolution_log:
        st.divider()
        st.subheader("📈 Evolution Progress")
        log_df = pd.DataFrame(st.session_state.evolution_log)
        st.line_chart(log_df.set_index("Run")["Accuracy"])
        st.caption("Full log available in the Evolution Log tab.")
