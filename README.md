
# 🏡 Airbnb Dynamic Price Prediction

A machine learning-powered web application that predicts Airbnb listing prices based on key features such as location, room type, availability, and minimum stay duration.

Built using **Streamlit**, **Scikit-learn**, and **Pandas**, this project demonstrates end-to-end data analysis, feature engineering, model training, and deployment.

---

## 🚀 Features

* 📊 Data preprocessing & cleaning
* 🧠 Machine Learning model (Linear Regression)
* 🔤 Encoding categorical variables
* 📈 Feature engineering (`price_per_night`)
* 🌐 Interactive UI using Streamlit
* 💰 Predict:

  * Price per night
  * Total cost for selected duration

---

## 📂 Project Structure

```
├── app.py                  # Streamlit application
├── AB_NYC_2019.csv        # Dataset
├── notebook.ipynb         # Data analysis & model experimentation
├── README.md              # Project documentation
```

---

## 📊 Dataset

* Source: Airbnb NYC 2019 Dataset
* Contains:

  * Neighbourhood group
  * Room type
  * Price
  * Minimum nights
  * Availability
  * Reviews

---

## ⚙️ Data Preprocessing

* Dropped unnecessary columns:

  * `name`, `id`, `host_name`, `last_review`
* Handled missing values:

  * `reviews_per_month → 0`
* Feature engineering:

  * `price_per_night = price / minimum_nights`
* Encoded categorical features:

  * `neighbourhood_group`
  * `room_type`

---

## 🧠 Machine Learning Model

* Model Used: **Linear Regression**
* Features:

  * `neighbourhood_group`
  * `room_type`
  * `minimum_nights`
  * `availability_365`
* Target:

  * `price_per_night`

Additional models explored in notebook:

* Ridge Regression
* Lasso Regression
* Decision Tree
* Random Forest

---

## 🖥️ Streamlit App

### User Inputs:

* Area (Neighbourhood Group)
* Room Type
* Minimum Nights
* Availability (days/year)

### Outputs:

* Predicted Price per Night 💵
* Total Cost for Stay 💰

---

## ▶️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/airbnb-price-prediction.git
cd airbnb-price-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install streamlit pandas scikit-learn
```

### 3. Run the App

```bash
streamlit run app.py
```

---

## 📈 Model Workflow

1. Load dataset
2. Clean & preprocess data
3. Encode categorical variables
4. Train regression model
5. Take user input
6. Predict price dynamically

---

## 📌 Key Highlights

* Real-time prediction using a trained ML model
* Lightweight and interactive UI
* Demonstrates complete ML pipeline
* Beginner-friendly project with practical use case

---

## 🧪 Future Improvements

* Add advanced models (XGBoost, LightGBM)
* Hyperparameter tuning
* Add location-based map visualization
* Deploy on cloud (Streamlit Cloud / AWS / GCP)
* Use deep learning for better accuracy

---

## 👨‍💻 Authors

* **Chakshat Bali**
* **Savi Chopra**

---

## 📜 License

This project is open-source and available under the MIT License.

---


* Add **badges (GitHub, Python, Streamlit)** for a more professional look
* Or convert this into a **top-tier resume project description (ATS optimized)**
