import os
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib, io

from ml_utility import (
    read_data,
    preprocess_data,
    train_model,
    evaluate_model
)

# Page Config
st.set_page_config(
    page_title="AutoML Pro",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 AutoML Dashboard")

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

    tabs = st.tabs(["📊 EDA", "🤖 Training", "📈 Results"])

    # ================== EDA TAB ==================
    with tabs[0]:
        st.subheader("📊 Exploratory Data Analysis")

        st.dataframe(df.head())

        # ================= Histogram =================
        col = st.selectbox("Select Column for Histogram", df.columns)
        st.subheader("Histogram")
        fig = px.histogram(df, x=col)
        st.plotly_chart(fig, use_container_width=True)

        # ================= Correlation Heatmap =================
        num_cols = df.select_dtypes(include='number').columns
        if len(num_cols) > 1:
            st.subheader(" Correlation Heatmap")
            fig = px.imshow(df[num_cols].corr(), text_auto=True)
            st.plotly_chart(fig, use_container_width=True)



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

            X_train, X_test, y_train, y_test = preprocess_data(
                df,
                target_column,
                scaler_type
            )

            is_classification = (
                df[target_column].dtype == "object"
                or df[target_column].nunique() < 20
            )

            if is_classification:
                from sklearn.linear_model import LogisticRegression
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.tree import DecisionTreeClassifier

                models = {
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "Random Forest": RandomForestClassifier(),
                    "Decision Tree": DecisionTreeClassifier()
                }
            else:
                from sklearn.linear_model import LinearRegression
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.tree import DecisionTreeRegressor

                models = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest Regressor": RandomForestRegressor(),
                    "Decision Tree Regressor": DecisionTreeRegressor()
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
                    y_test
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

            st.success("✅ Training Completed!")

    # ================== RESULTS TAB ==================
    with tabs[2]:
        st.subheader("📈 Results Dashboard")

        if "results" in st.session_state:

            result_df = st.session_state["results"]
            st.dataframe(result_df)

            fig = px.bar(result_df, x="Model", y="Accuracy (%)", color="Model")
            st.plotly_chart(fig, use_container_width=True)

            # Show best score
            best_score = st.session_state["best_score"]
            st.metric("🏆 Best Score", f"{best_score * 100:.2f}%")

            # Confusion Matrix (only classification)
            from sklearn.metrics import confusion_matrix

            y_test = st.session_state["y_test"]
            X_test = st.session_state["X_test"]
            model = st.session_state["best_model"]

            if y_test.dtype == "object" or y_test.nunique() < 20:
                y_pred = model.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)

                fig = px.imshow(cm, text_auto=True, title="Confusion Matrix")
                st.plotly_chart(fig)

            # Download model
            bio = io.BytesIO()
            joblib.dump(model, bio)
            bio.seek(0)

            st.download_button(
                "⬇ Download Best Model",
                bio,
                file_name="best_model.pkl"
            )

        else:
            st.info("⚠ Train the model first")