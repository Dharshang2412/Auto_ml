import os
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib, io

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph
)

from reportlab.lib.styles import getSampleStyleSheet

from ml_utility import (
    read_data,
    preprocess_data,
    train_model,
    evaluate_model
)
from audit_agent import run_probes, get_agent_verdict
from profiling_agent import compute_profile, get_profile_narrative
def generate_pdf(report_text):

    pdf_path = "AutoML_Report.pdf"

    doc = SimpleDocTemplate(pdf_path)

    styles = getSampleStyleSheet()

    elements = [
        Paragraph(
            report_text.replace("\n", "<br/>"),
            styles["BodyText"]
        )
    ]

    doc.build(elements)

    return pdf_path


# Page Config
st.set_page_config(
    page_title="AutoML Pro",
    page_icon="🚀",
    layout="wide"
)


st.title("👾 AutoML Dashboard")

# Sidebar
st.sidebar.header("📂 Data Source")
mode = st.sidebar.radio("Choose Data Source", ["Upload File", "Built-in Dataset"])

df = None

# Upload CSV
file = st.sidebar.file_uploader(
    "Upload CSV or Excel File",
    type=["csv", "xlsx"]
)

if file:
    try:
        if file.name.endswith(".csv"):
            try:
                df = pd.read_csv(file, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(file, encoding="latin1")
        else:
            df = pd.read_excel(file)

        st.success("✅ File loaded successfully!")

    except Exception as e:
        st.error(f"Error loading file: {e}")
# Built-in Dataset
else:
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(working_dir)
    dataset_list = os.listdir(f"{parent_dir}/data")

    dataset = st.sidebar.selectbox("Select Dataset", dataset_list)
    if dataset:
        df = read_data(dataset)

# If data is loaded
if df is not None:

    tabs = st.tabs([
        "📊EDA",
        "🛡️Trust Auditor",
        "⚙️Training",
        "⚡Insights"

    ])

    if "results" not in st.session_state:
        with tabs[3]:
            st.info("👈 Train a model in the Training tab first to see results and insights here.")


    # ================== TRUST AUDITOR TAB (Agent) ==================
    with tabs[1]:
        st.subheader("🛡️ Data Quality & Trust Auditor Agent")
        st.caption(
            "Runs a battery of statistical checks (leakage, imbalance, cardinality, "
            "missingness, duplicates) and then reasons over the findings to flag risks "
            "before you spend time training models on broken data."
        )

        audit_target = st.selectbox(
            "Target Column to Audit",
            df.columns,
            key="audit_target_column"
        )

        audit_is_classification = (
            df[audit_target].dtype == "object"
            or (
                pd.api.types.is_integer_dtype(df[audit_target])
                and df[audit_target].nunique() < 20
            )
        )

        task_label = "Classification" if audit_is_classification else "Regression"
        st.info(f"Detected task type: **{task_label}**")

        if not os.environ.get("GEMINI_API_KEY"):
            st.warning(
                "No Gemini API key found. The audit will still run the statistical "
                "probes, but will fall back to a rule-based summary instead of the AI "
                "agent's reasoning. Set GEMINI_API_KEY in your .env file to enable full reasoning."
            )

        if st.button("🔍 Run Trust Audit"):
            with st.spinner("Running statistical probes..."):
                findings = run_probes(df, audit_target, audit_is_classification)

            dataset_meta = {
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "task_type": task_label,
                "target_column": audit_target
            }

            with st.spinner("Agent is reasoning over the findings..."):
                verdict = get_agent_verdict(findings, dataset_meta)

            st.session_state["audit_verdict"] = verdict
            st.session_state["audit_findings_raw"] = findings

        if "audit_verdict" in st.session_state:
            verdict = st.session_state["audit_verdict"]

            verdict_label = verdict.get("verdict", "unknown")
            verdict_display = {
                "safe_to_train": ("✅ Safe to Train", "success"),
                "train_with_caution": ("⚠️ Train with Caution", "warning"),
                "fix_before_training": ("🛑 Fix Before Training", "error"),
            }.get(verdict_label, (f"Verdict: {verdict_label}", "info"))

            label, style = verdict_display
            getattr(st, style)(label)

            st.write(verdict.get("summary", ""))

            ranked = verdict.get("ranked_findings", [])
            if ranked:
                st.subheader("📋 Findings, Ranked by Risk")

                risk_order = {"high": 0, "medium": 1, "low": 2}
                ranked_sorted = sorted(
                    ranked,
                    key=lambda f: risk_order.get(f.get("risk", "medium"), 1)
                )

                for f in ranked_sorted:
                    risk = f.get("risk", "medium")
                    icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(risk, "⚪")
                    col_label = f.get("column") or "Dataset-wide"
                    with st.expander(f"{icon} [{risk.upper()}] {col_label} — {f.get('issue', '')}"):
                        st.write(f.get("recommendation", ""))
            else:
                st.success("No risk findings to display — dataset looks clean.")

            with st.expander("🔬 Raw probe output (debug)"):
                st.json(st.session_state.get("audit_findings_raw", []))

    # ================== EDA TAB ==================
    with tabs[0]:
        st.subheader("📊 Exploratory Data Analysis")

        st.dataframe(df.head())

        # ================= Histogram =================
        col = st.selectbox("Select Column for Histogram", df.columns)
        st.subheader("Histogram")
        fig = px.histogram(df, x=col)
        st.plotly_chart(
            fig,
            use_container_width=True,
            key="histogram_chart"
        )
        # ================= Correlation Heatmap =================
        num_cols = df.select_dtypes(include='number').columns
        if len(num_cols) > 1:
            st.subheader(" Correlation Heatmap")
            fig = px.imshow(df[num_cols].corr(), text_auto=True)
        st.plotly_chart(
            fig,
            use_container_width=True,
            key="correlation_heatmap"
        )



    # ================== TRAINING TAB ==================
    with tabs[2]:
        st.subheader("🤖 Model Training")

        col1, col2, col3 = st.columns(3)

        with col1:
            target_column = st.selectbox("Target Column", df.columns)

        with col2:
            scaler_type = st.selectbox("Scaler", ["standard", "minmax"])

        with col3:
            model_name = st.text_input("Model Name", "best_model")

        if st.button("🚀 Train Models"):

            X_train, X_test, y_train, y_test, feature_names = preprocess_data(
                df,
                target_column,
                scaler_type
            )

            st.session_state["feature_names"] = feature_names

            is_classification = (
                    df[target_column].dtype == "object"
                    or (
                            pd.api.types.is_integer_dtype(df[target_column])
                            and df[target_column].nunique() < 20
                    )
            )
            if is_classification:

                from sklearn.linear_model import LogisticRegression
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.tree import DecisionTreeClassifier
                from sklearn.neighbors import KNeighborsClassifier
                from xgboost import XGBClassifier

                models = {
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "Random Forest": RandomForestClassifier(),
                    "Decision Tree": DecisionTreeClassifier(),
                    "KNN": KNeighborsClassifier(),
                    "XGBoost": XGBClassifier(
                        eval_metric="logloss",
                        random_state=42
                    )
                }

            else:
                from sklearn.linear_model import LinearRegression
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.tree import DecisionTreeRegressor
                from sklearn.neighbors import KNeighborsRegressor
                from xgboost import XGBRegressor

                models = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest Regressor": RandomForestRegressor(),
                    "Decision Tree Regressor": DecisionTreeRegressor(),
                    "KNN Regressor": KNeighborsRegressor(),
                    "XGBoost Regressor": XGBRegressor(
                        random_state=42
                    )
                }
            results = []
            best_model = None
            best_score = -999

            for name, model in models.items():
                trained_model = train_model(
                    X_train,
                    y_train,
                    model,
                    model_name
                )

                score = evaluate_model(
                    trained_model,
                    X_test,
                    y_test,
                    is_classification
                )

                # Convert to percentage
                score_percent = round(score * 100, 2)

                results.append([name, score_percent])

                if score > best_score:
                    best_score = score
                    best_model = trained_model

            result_df = pd.DataFrame(results, columns=["Model", "Accuracy (%)"])

            st.session_state["results"] = result_df
            st.session_state["best_model"] = best_model
            st.session_state["y_test"] = y_test
            st.session_state["X_test"] = X_test
            st.session_state["best_score"] = best_score
            st.session_state["is_classification"] = is_classification

            st.success("✅ Training Completed!")

        if "results" in st.session_state:
            st.divider()
            st.header("📈 Results Dashboard")

            result_df = st.session_state["results"]

            result_df = result_df.sort_values(
                by=result_df.columns[1],
                ascending=False
            ).reset_index(drop=True)

            result_df.insert(
                0,
                "Rank",
                range(1, len(result_df) + 1)
            )

            st.subheader("🏅 Model Leaderboard")

            st.dataframe(
                result_df,
                hide_index=True,
                use_container_width=True
            )

            metric_col = result_df.columns[2]

            leaderboard_fig = px.bar(
                result_df,
                x="Model",
                y=metric_col,
                color="Model",
                text=metric_col,
                title="Model Performance Comparison"
            )

            st.plotly_chart(
                leaderboard_fig,
                use_container_width=True,
                key="leaderboard_chart"
            )
            best_score = st.session_state["best_score"]
            model = st.session_state["best_model"]
            y_test = st.session_state["y_test"]
            X_test = st.session_state["X_test"]

            st.metric(
                "🏆 Best Score",
                f"{best_score * 100:.2f}%"
            )

            # ================== AI INSIGHTS TAB ==================

            with tabs[3]:


                st.subheader(" Data Profiling Card")
                st.caption(
                    "A quick health check on the dataset: missing data, duplicate rows, "
                    "skewed columns, outliers, and identifier-like columns -- pure statistics, "
                    "no model training involved."
                )

                if st.button("🔍 Generate Profile"):
                    with st.spinner("Computing dataset profile..."):
                        try:
                            profile = compute_profile(df)
                            st.session_state["data_profile"] = profile
                        except Exception as e:
                            st.error(f"Profiling failed: {e}")

                if "data_profile" in st.session_state:
                    profile = st.session_state["data_profile"]

                    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
                    with p_col1:
                        st.metric("Rows", profile["n_rows"])
                    with p_col2:
                        st.metric("Columns", profile["n_cols"])
                    with p_col3:
                        st.metric("Missing Data", f"{profile['overall_missing_pct']}%")
                    with p_col4:
                        st.metric("Duplicate Rows", profile["duplicate_rows"])

                    st.dataframe(
                        profile["profile_table"],
                        hide_index=True,
                        use_container_width=True
                    )

                    flag_cols = st.columns(4)
                    flag_labels = [
                        ("⚠️ High Missing", profile["flagged_high_missing"]),
                        ("🆔 ID-like Columns", profile["flagged_high_cardinality"]),
                        ("📐 Skewed Columns", profile["flagged_skewed"]),
                        ("📍 Notable Outliers", profile["flagged_outliers"]),
                    ]
                    for col_widget, (label, items) in zip(flag_cols, flag_labels):
                        with col_widget:
                            if items:
                                st.warning(f"**{label}**\n\n" + ", ".join(items))
                            else:
                                st.success(f"**{label}**\n\nNone")

                    if st.button("✨ Generate AI Summary"):
                        with st.spinner("Agent is summarizing the profile..."):
                            narrative = get_profile_narrative(profile)
                            st.session_state["profile_narrative"] = narrative

                    if "profile_narrative" in st.session_state:
                        nr = st.session_state["profile_narrative"]
                        st.info(f"**{nr.get('headline', '')}**")
                        st.write(nr.get("summary", ""))

            # ================= Feature Importance =================

            if hasattr(model, "feature_importances_"):
                importance_df = pd.DataFrame({
                    "Feature": st.session_state["feature_names"],
                    "Importance": model.feature_importances_
                })

                importance_df = importance_df.sort_values(
                    by="Importance",
                    ascending=False
                )

                st.subheader("🔍 Feature Importance")

                fig = px.bar(
                    importance_df.head(15),
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Top Important Features"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key="feature_importance_chart"

                )

                st.dataframe(importance_df)
                if hasattr(model, "feature_importances_"):
                    top_feature = importance_df.iloc[0]

                    st.metric(
                        "✦ Most Important Feature",
                        top_feature["Feature"]
                    )

            # ================= Confusion Matrix =================

            if st.session_state["is_classification"]:
                from sklearn.metrics import confusion_matrix

                y_pred = model.predict(X_test)

                cm = confusion_matrix(
                    y_test,
                    y_pred
                )

                fig = px.imshow(
                    cm,
                    text_auto=True,
                    title="Confusion Matrix"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key="confusion_matrix_chart"

                )

            # ================= Download Model =================

            bio = io.BytesIO()
            joblib.dump(model, bio)
            bio.seek(0)

            st.download_button(
                "⬇ Download Best Model",
                bio,
                file_name="best_model.pkl"
            )
            csv = result_df.to_csv(index=False)
