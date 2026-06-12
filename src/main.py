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
        "⚙️Training",
        "⚡Insights"

    ])


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
    with tabs[1]:
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

            with tabs[2]:

                st.subheader("✨ AI Result Summary")

                best_model_name = result_df.iloc[0]["Model"]

                summary = f"""
                Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.

                Best Model:
                {best_model_name}

                Performance:
                {best_score * 100:.2f}%

                The model comparison indicates that
                {best_model_name} achieved the highest performance.

                Recommendation:
                Deploy {best_model_name} for production use.
                """

                st.success(summary)
                st.subheader("📑 Executive Report")

                best_model_name = result_df.iloc[0]["Model"]

                report = f"""
                AUTO ML EXECUTIVE REPORT

                =================================

                Dataset Information

                Rows: {df.shape[0]}
                Columns: {df.shape[1]}

                =================================

                MODEL PERFORMANCE

                Best Model:
                {best_model_name}

                Best Score:
                {best_score * 100:.2f}%

                =================================

                BUSINESS RECOMMENDATION

                Deploy {best_model_name}
                for production use.

                =================================
                """

                st.text_area(
                    "Generated Report",
                    report,
                    height=300
                )
                st.download_button(
                    "📄 Download Report",
                    report,
                    file_name="AutoML_Report.txt",
                    mime="text/plain"
                )

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



