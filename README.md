# Breast Cancer Prediction App

This is a simple web app that predicts whether a breast cancer tumor is malignant or benign using a logistic regression model.

## Project Structure

- `app.py`: Flask web application
- `model.pkl`: Trained ML model
- `cancer_data.csv`: Dataset used for training
- `templates/index.html`: User input form for predictions
- `requirements.txt`: Required Python packages

## How to Run

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000/` in your browser.