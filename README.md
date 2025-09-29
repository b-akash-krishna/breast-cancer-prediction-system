# Breast Cancer Prediction System

This project is a full-stack application for predicting breast cancer diagnosis. It features a Python backend built with Flask for the machine learning model and a React/TypeScript frontend for a user-friendly interface. The system allows users to make predictions, retrain the model with new data, and manage datasets.

---

## 🚀 Features

* **Real-time Prediction**: Get an instant diagnosis (Malignant or Benign) with a confidence score based on a patient's features.
* **Dynamic Model Training**: Retrain the machine learning model on an uploaded CSV dataset or a selected subset of the current data.
* **Comprehensive Metrics**: View detailed performance metrics of the model, including accuracy, precision, recall, F1-score, and a confusion matrix.
* **Feature Importance Analysis**: Visualize the importance of each feature used in the model to understand the factors driving predictions.
* **Data Management**: Upload new datasets via CSV files to improve the model's performance.
* **Health Checks**: API endpoints to monitor the health and status of the backend server and the loaded model.

---

## 🛠️ Technologies Used

### Backend
* **Python**: The core language for the backend.
* **Flask**: A micro web framework for building the API.
* **scikit-learn**: Used for machine learning functionalities, including model training (`LogisticRegression`), data scaling (`StandardScaler`), and feature selection (`SelectKBest`).
* **Pandas & NumPy**: Essential libraries for data manipulation and numerical operations.
* **Flask-CORS**: Handles Cross-Origin Resource Sharing for communication between the frontend and backend.

### Frontend
* **React**: A JavaScript library for building the user interface.
* **TypeScript**: A typed superset of JavaScript that adds static typing.
* **Vite**: A fast build tool for the frontend development server.
* **Recharts & Chart.js**: Libraries for creating data visualizations like charts and graphs.
* **Lucide-React**: An icon library for a clean UI.
* **Tailwind CSS**: A utility-first CSS framework for styling.

---

## ⚙️ Installation & Setup

### Prerequisites

* Python 3.9+
* Node.js and npm (or yarn)

### Backend Setup

1.  Navigate to the `backend` directory.
2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
3.  Start the Flask server:
    ```bash
    python app.py
    ```
    The API will run on `http://127.0.0.1:5000`.

### Frontend Setup

1.  Navigate to the project's root directory.
2.  Install the frontend dependencies:
    ```bash
    npm install
    ```
3.  Start the development server:
    ```bash
    npm run dev
    ```
    The frontend application will be available at `http://localhost:5173`.

---

## 💻 API Endpoints

The backend exposes several REST API endpoints for interaction with the model and data.

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | Home endpoint for health check. |
| `/health` | `GET` | Checks the health and status of the API and model. |
| `/status` | `GET` | Provides a comprehensive summary of the model's status and metrics. |
| `/predict` | `POST` | Makes a prediction based on a list of input features. |
| `/data` | `GET` | Fetches the current dataset with optional filters. |
| `/data/random-sample` | `GET` | Retrieves a random sample from the dataset. |
| `/train` | `POST` | Retrains the model on a selected subset of the data. |
| `/upload` | `POST` | Uploads a new CSV file to use for training the model. |
| `/predict/batch` | `POST` | Performs batch predictions from an uploaded CSV file. |
| `/files` | `GET` | Lists all uploaded CSV files. |
| `/files/<filename>` | `DELETE` | Deletes a specific uploaded file. |
| `/files/<filename>/analyze` | `GET` | Provides an analysis of a specific uploaded file. |
