# 🍽️ RestaurantIQ

### AI-Powered Restaurant Demand Forecasting & Inventory Recommendation System

Predict restaurant demand using Machine Learning and optimize inventory planning with intelligent recommendations.

[![Live Demo](https://img.shields.io/badge/%F0%9F%9A%80-Live%20Demo-success?style=for-the-badge)](https://restaurant-demand-forecasting-vh4q.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)](https://github.com/vrs2k5/restaurant-demand-forecasting)
[![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black?style=for-the-badge&logo=flask)](https://github.com/vrs2k5/restaurant-demand-forecasting)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?style=for-the-badge&logo=scikitlearn)](https://github.com/vrs2k5/restaurant-demand-forecasting)
[![SQLite](https://img.shields.io/badge/SQLite-Database-blue?style=for-the-badge&logo=sqlite)](https://github.com/vrs2k5/restaurant-demand-forecasting)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-UI-purple?style=for-the-badge&logo=bootstrap)](https://github.com/vrs2k5/restaurant-demand-forecasting)

---

# 🌐 Live Demo

🚀 **Try the application here**

### https://restaurant-demand-forecasting-vh4q.onrender.com/

> **Note:** The application is hosted on Render's free plan. If inactive, it may take **30–60 seconds** to wake up.

---

# 📖 Project Overview

RestaurantIQ is a Machine Learning-based web application that predicts restaurant demand and provides inventory recommendations to help restaurant owners optimize stock planning, reduce food wastage, and improve operational efficiency.

The application compares multiple Machine Learning algorithms and presents the best-performing model through an interactive dashboard.

**A note on the data:** this project uses a *synthetic* dataset simulating 50 fictional Southeast Asian restaurants (Jan 2024 – Jan 2025, 14 menu items, with weather / promotion / event flags). It is not scraped real-world operational data, and that's stated here deliberately rather than implied otherwise. The modeling methodology below is built to hold up regardless of the data source.

---

# ✨ Features

- 📈 Restaurant Demand Forecasting
- 🍽 Smart Inventory Recommendation
- 🤖 Multiple Machine Learning Models
- 📊 Interactive Dashboard
- 📉 Sales Analytics
- 📚 Prediction History
- 💾 SQLite Database Integration
- 🌐 Live Web Application
- 📱 Responsive Bootstrap UI

---

# 🧠 Machine Learning Models

The project compares four regression algorithms, evaluated on a **chronological** train/test split (the most recent 20% of dates held out as the test set). This matters because a random split would let the model train on data that comes *after* what it's tested on — which leaks information and makes the reported accuracy look better than it would actually be in production. For a time-series forecasting problem, the split has to respect time order.

| Model             | R²     | MAE   | RMSE   |
|-------------------|--------|-------|--------|
| Linear Regression | 0.629  | 100.6 | 147.7  |
| Random Forest     | 0.712  | 87.9  | 130.1  |
| XGBoost           | 0.731  | 83.3  | 125.7  |
| **LightGBM**      | **0.733** | **83.3** | **125.2** |

LightGBM is selected as the model used for inventory recommendations. Feature engineering includes sequence-based lag features (previous 1–2 recorded transactions per restaurant + menu item, since the dataset is sparse enough that a literal "yesterday" lag isn't meaningful) plus cyclical date encoding, weather, promotion, and event flags.

The system evaluates these models and uses the best-performing one for prediction.

---

# 🛠️ Technology Stack

## Backend
- Python
- Flask
- SQLAlchemy

## Machine Learning
- Scikit-Learn
- XGBoost
- LightGBM
- Pandas
- NumPy
- Joblib

## Frontend
- HTML5
- CSS3
- Bootstrap 5
- JavaScript

## Database
- SQLite

## Deployment
- Render

---

# 📷 Application Screenshots

## 🏠 Home Page
![Home](https://github.com/vrs2k5/restaurant-demand-forecasting/raw/main/screenshots/home.png)

---

## 📈 Prediction
![Prediction](https://github.com/vrs2k5/restaurant-demand-forecasting/raw/main/screenshots/prediction.png)

---

## 🍽 Inventory Recommendation
![Recommendation](https://github.com/vrs2k5/restaurant-demand-forecasting/raw/main/screenshots/recommendation.png)

---

## 📊 Dashboard
![Dashboard](https://github.com/vrs2k5/restaurant-demand-forecasting/raw/main/screenshots/dashboard.png)

---

## 📉 Analytics
![Analytics](https://github.com/vrs2k5/restaurant-demand-forecasting/raw/main/screenshots/analytics.png)

---

## 📚 Prediction History
![History](https://github.com/vrs2k5/restaurant-demand-forecasting/raw/main/screenshots/history.png)

---

# ⚙️ System Workflow

```
            User
              │
              ▼
    Flask Web Application
              │
              ▼
     Feature Engineering
              │
              ▼
  Machine Learning Models
   ├── Linear Regression
   ├── Random Forest
   ├── XGBoost
   └── LightGBM
              │
              ▼
     Demand Prediction
              │
              ▼
Inventory Recommendation
              │
              ▼
  Dashboard & Analytics
```

---

# 📂 Project Structure

```
RestaurantIQ
│
├── dataset/
├── database/
├── saved_models/
├── screenshots/
├── static/
├── templates/
│
├── app.py
├── config.py
├── model.py
├── models.py
├── Procfile
├── requirements.txt
└── README.md
```

---

# 🚀 Installation

Clone the repository
```
git clone https://github.com/vrs2k5/restaurant-demand-forecasting.git
```

Move into the project
```
cd restaurant-demand-forecasting
```

Create a virtual environment
```
python -m venv venv
```

### Windows
```
venv\Scripts\activate
```

### Linux / macOS
```
source venv/bin/activate
```

Install dependencies
```
pip install -r requirements.txt
```

Run the application
```
python app.py
```

Open your browser and visit:
```
http://127.0.0.1:5000
```

---

# 🚀 Future Enhancements

- PostgreSQL Integration
- User Authentication & Authorization
- Docker Support
- CI/CD with GitHub Actions
- Real-time Demand Forecasting
- Cloud Storage Integration
- REST API Documentation
- Mobile Application

---

# 👨‍💻 Developer

### **Venkata Ramana Sai Nimmakanti**

- **GitHub:** [vrs2k5](https://github.com/vrs2k5)
- **LinkedIn:** [Venkata Ramana Sai Nimmakanti](https://www.linkedin.com/in/venkata-ramana-sai-nimmakanti-450718298/)

---

# ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.

Your support helps improve the project and motivates future development.

---

### 🍽️ RestaurantIQ
**Built with ❤️ using Python, Flask & Machine Learning**
