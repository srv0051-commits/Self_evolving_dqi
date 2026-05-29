import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Self-Evolving Intelligence", layout="wide")
st.title("🧠 Self-Evolving Data Quality & ML Intelligence")
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

# ----------------------------
# PIPELINE FUNCTION
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
# TABS
# ----------------------------
tab1, tab2 = st.tabs(["🚀 Evolution Engine", "🛠️ Data Architect"])

# ---------------------------------------------------------
# DATA ARCHITECT
# ---------------------------------------------------------
with tab2:
    st.header("🛠️ Dataset Builder")
    name = st.text_input("Dataset Name")

    if 'builder_df' not in st.session_state:
        st.session_state.builder_df = pd.DataFrame(
            [[25, "Male", 1], [30, "Female", 0]],
            columns=["Age", "Gender", "Target"]
        )

    edited_df = st.data_editor(st.session_state.builder_df, num_rows="dynamic")

    if st.button("💾 Save Dataset"):
        if name:
            st.session_state.custom_datasets[name] = edited_df
            st.success(f"Saved '{name}'")
        else:
            st.error("Enter dataset name")

# ---------------------------------------------------------
# EVOLUTION ENGINE
# ---------------------------------------------------------
with tab1:

    with st.sidebar:
        st.header("⚙️ Control Panel")

        source = st.radio("Data Source", ["Upload CSV", "Custom Dataset"])

        df = None
        if source == "Upload CSV":
            file = st.file_uploader("Upload CSV", type="csv")
            if file:
                df = pd.read_csv(file)
        else:
            if st.session_state.custom_datasets:
                selected = st.selectbox("Select Dataset", list(st.session_state.custom_datasets.keys()))
                df = st.session_state.custom_datasets[selected]
            else:
                st.warning("No datasets found")

        use_iso = st.checkbox("Enable Anomaly Removal", True)
        run = st.button("⚙️ Run Evolution")
        tune = st.button("🚀 Run Tuning")

    if df is not None:

        drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'id', 'Id']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        st.subheader("📊 Data Quality Audit")

        target = st.selectbox("Select Target Variable", df.columns, index=len(df.columns)-1)

        # Remove missing target rows
        initial_len = len(df)
        df = df.dropna(subset=[target])
        if len(df) < initial_len:
            st.warning(f"Removed {initial_len - len(df)} rows with missing target")

        c1, c2 = st.columns(2)
        c1.dataframe(df.head())

        quality = 1 - (df.isnull().sum().sum() / df.size)
        c2.metric("Data Quality Score", f"{quality:.2%}")
        c2.metric("Rows", len(df))

        X = df.drop(columns=[target])
        y = df[target]

        X_original = X.copy()

        if len(df) < 10:
            st.error("Dataset too small")
        elif y.nunique() > 10 and y.dtype != 'object':
            st.warning("Target looks continuous. This system is for classification.")
        else:
            num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            cat_cols = X.select_dtypes(include=['object']).columns.tolist()

            if run or tune:
                with st.spinner("System evolving..."):

                    # ANOMALY DETECTION
                    if use_iso and num_cols:
                        iso = IsolationForest(contamination=0.05, random_state=42)
                        num_data = X[num_cols].fillna(X[num_cols].median())
                        mask = iso.fit_predict(num_data)
                        X, y = X[mask == 1], y[mask == 1]
                        st.write(f"Anomalies removed: {(mask==-1).sum()}")

                    # TARGET ENCODING
                    if y.dtype == 'object' or y.dtype == 'bool':
                        y = LabelEncoder().fit_transform(y)

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )

                    models = {
                        "Random Forest": RandomForestClassifier(random_state=42),
                        "Logistic Regression": LogisticRegression(max_iter=2000)
                    }

                    best_acc = 0

                    for name, model in models.items():
                        pipe = get_pipeline(model, num_cols, cat_cols)

                        if tune and name == "Random Forest":
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
                            best_name = name
                            best_preds = preds
                            best_metrics = (precision, recall, f1)

                # RESULTS
                st.success(f"🏆 Best Model: {best_name}")
                st.metric("Accuracy", f"{best_acc:.2%}")
                st.metric("Best Model", best_name)

                st.write(f"Precision: {best_metrics[0]:.2f}")
                st.write(f"Recall: {best_metrics[1]:.2f}")
                st.write(f"F1 Score: {best_metrics[2]:.2f}")

                # CONFUSION MATRIX
                st.subheader("🔍 Confusion Matrix")
                fig, ax = plt.subplots()
                sns.heatmap(confusion_matrix(y_test, best_preds), annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

                # FEATURE IMPORTANCE
                if best_name == "Random Forest":
                    st.subheader("💡 Feature Importance")

                    names = best_model.named_steps['preprocessor'].get_feature_names_out()
                    names = [n.split("__")[-1] for n in names]

                    importance = best_model.named_steps['model'].feature_importances_

                    feat_df = pd.DataFrame({
                        "Feature": names,
                        "Importance": importance
                    }).sort_values(by="Importance", ascending=False).head(10)

                    st.bar_chart(feat_df.set_index("Feature"))

                # SAVE STATE
                st.session_state.best_model_trained = best_model
                st.session_state.training_columns = X.columns.tolist()

                st.session_state.evolution_log.append({
                    "Step": len(st.session_state.evolution_log)+1,
                    "Accuracy": best_acc
                })
# ----------------------------
# MULTI DATASET DRIFT DETECTION
# ----------------------------
if st.session_state.best_model_trained:

    st.divider()
    st.subheader("🕵️ Multi-Dataset Drift Detection")

    uploaded_files = st.file_uploader(
        "Upload One or More New Datasets",
        type="csv",
        accept_multiple_files=True,
        key="multi_drift"
    )

    if uploaded_files:

        for i, new_file in enumerate(uploaded_files):

            st.markdown(f"### 📂 Dataset {i+1}: {new_file.name}")

            df_new = pd.read_csv(new_file)

            drift_results = []

            # Find common columns
            common_cols = [col for col in num_cols if col in df_new.columns]

            if not common_cols:
                st.error("❌ No matching numeric columns found")
                continue

            for col in common_cols:

                old_mean = X_original[col].mean()
                new_mean = df_new[col].mean()

                old_std = X_original[col].std()
                new_std = df_new[col].std()

                mean_shift = abs(old_mean - new_mean) / (abs(old_mean) + 1e-6)
                std_shift = abs(old_std - new_std) / (abs(old_std) + 1e-6)

                drift_score = (mean_shift + std_shift) / 2

                if drift_score < 0.1:
                    status = "✅ Stable"
                elif drift_score < 0.25:
                    status = "⚠️ Moderate Drift"
                else:
                    status = "🚨 Severe Drift"

                drift_results.append({
                    "Feature": col,
                    "Drift Score": f"{drift_score*100:.2f}%",
                    "Status": status
                })

            drift_df = pd.DataFrame(drift_results)
            st.dataframe(drift_df)

            # 🚨 ALERT
            if any("Severe" in r["Status"] for r in drift_results):
                st.error(f"🚨 Dataset {i+1} has severe drift!")
            else:
                st.success(f"✅ Dataset {i+1} is mostly stable")

            # 📊 VISUAL COMPARISON
            st.subheader(f"📊 Distribution Comparison (Dataset {i+1})")

            feature_to_plot = st.selectbox(
                f"Select Feature for Dataset {i+1}",
                common_cols,
                key=f"feature_{i}"
            )

            fig, ax = plt.subplots()
            sns.kdeplot(X_original[feature_to_plot], label="Original", ax=ax)
            sns.kdeplot(df_new[feature_to_plot], label="New", ax=ax)
            ax.legend()
            st.pyplot(fig)
    if st.session_state.evolution_log:
        st.divider()
        st.subheader("📈 Evolution Progress")
        log = pd.DataFrame(st.session_state.evolution_log)
        st.line_chart(log.set_index("Step"))