# 💻 AI Hardware Price Estimator

An end-to-end Machine Learning pipeline that predicts laptop market values with **81.3% cross-validated accuracy**. 

🔗 **Live Application:** [(https://rutman-predicts-hardware.streamlit.app/)]

## 🚀 Key Features & Engineering Highlights
* **Regex Feature Engineering:** Extracted exact hardware specs (SSD/HDD capacities, CPU brands, and screen types) from highly chaotic, messy text strings.
* **Robust Pipeline Structure:** Utilized Scikit-Learn `Pipeline` and `ColumnTransformer` to completely isolate preprocessing from model training, ensuring zero data leakage.
* **Hyperparameter Tuning:** Used `RandomizedSearchCV` across a 5-Fold Cross-Validation split to optimize the Random Forest Regressor and prevent overfitting.
* **Dynamic Production UI:** Built a Streamlit interface featuring real-time input validation and responsive hardware constraints.
