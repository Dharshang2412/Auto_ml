# 🚀 AutoML Pro Dashboard

An interactive **Auto Machine Learning Web App** built using **Streamlit** that allows users to upload datasets, perform Exploratory Data Analysis (EDA), train multiple ML models, and compare their performance — all in a few clicks.

---

## 🌟 Features

### 📂 Data Handling
- Upload **CSV & Excel (.xlsx)** files
- Supports built-in datasets
- Handles encoding issues automatically

### 📊 Exploratory Data Analysis (EDA)
- Dataset preview
- 📈 Histogram (feature distribution)
- 🔥 Correlation Heatmap
- 🥧 Pie Chart (categorical distribution)

### 🤖 Model Training
- Automatic detection:
  - Classification
  - Regression
- Multiple models trained at once:
  - Logistic Regression
  - Random Forest
  - Decision Tree
  - Linear Regression (for regression)

### 📈 Results Dashboard
- Model leaderboard
- Accuracy displayed in **percentage (%)**
- Interactive bar chart comparison
- Confusion matrix (for classification)
- Download trained model (`.pkl`)

---

## 🛠️ Tech Stack

- **Frontend/UI:** Streamlit  
- **Data Processing:** Pandas, NumPy  
- **Machine Learning:** Scikit-learn  
- **Visualization:** Plotly  
- **Model Saving:** Joblib  

---

## 📂 Project Structure


Auto_ml/
│
├── src/
│ ├── main.py # Streamlit App
│ └── ml_utility.py # ML pipeline functions
│
├── data/ # Sample datasets
├── trained_model/ # Saved models
├── requirements.txt
└── README.md


---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/Auto_ml.git
cd Auto_ml
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Run the app
streamlit run src/main.py
🌐 Deployment (Streamlit Cloud)
Push code to GitHub
Go to Streamlit Cloud
Click New App
Select your repo
Set:
Main file: src/main.py
Click Deploy
📸 Screenshots (Optional)

Add screenshots of your app here for better presentation

🎯 Use Cases
Quick ML prototyping
Learning machine learning workflows
Dataset exploration
Comparing ML models easily
🚀 Future Improvements
Hyperparameter tuning
Advanced metrics (F1, RMSE, MAE)
Feature importance visualization
Model history tracking
User authentication system
🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

📜 License

This project is open-source and available under the MIT License.

👨‍💻 Author

Dharshan G
GitHub: https://github.com/Dharshang2412
stremlit: https://automl-dharshan2412.streamlit.app/

⭐ Support

If you like this project, give it a ⭐ on GitHub!
