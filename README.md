# Iris Species Predictor 🌸

A web app that predicts Iris flower species (Setosa, Versicolor, Virginica) using a trained machine learning model.

## Tech Stack
- **Model**: Scikit-Learn (Random Forest / Logistic Regression)
- **Backend**: Flask (Python)
- **Deployment**: Vercel

## Local Development

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py
```

Then open: http://127.0.0.1:5000

Or just double-click **start.bat** (Windows).

## Project Structure

```
iris-project/
├── app.py                 # Flask web application
├── iris_model.pkl         # Trained ML model
├── iris_model.ipynb       # Jupyter notebook (training)
├── requirements.txt       # Python dependencies
├── vercel.json            # Vercel deployment config
├── templates/
│   └── index.html         # Web UI
└── start.bat              # Quick-start script (Windows)
```

## How It Works

1. Enter 4 flower measurements (sepal/petal length & width in cm)
2. Click **Predict Iris Species**
3. See the predicted species and confidence percentage

## Deploying to Vercel

```bash
npm i -g vercel
vercel login
vercel --prod
```
