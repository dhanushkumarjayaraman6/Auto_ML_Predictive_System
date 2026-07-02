# AutoML Predictive System

Upload a dataset (CSV, XLSX, XLS, JSON, or TSV), and this Streamlit app will:

1. **Load** the file and convert it into a clean pandas DataFrame.
2. **Clean** it — remove duplicate rows, drop columns that are mostly empty, and fill or drop missing values (your choice).
3. **Detect** whether the target column you pick is a classification or regression problem.
4. **Train multiple models** — Logistic/Linear Regression, Random Forest, Decision Tree, KNN, SVM, and XGBoost (if installed) — and evaluate each with the right metrics.
5. **Recommend the best model**, show a leaderboard, plot feature importance, and let you download the winning model as a `.pkl` file ready for reuse.

## Demo flow

Upload file → preview & clean data → pick target column → click **Train** → see leaderboard → download best model.

## Setup

'''bash
git clone <your-repo-url>
cd <repo-folder>
pip install -r requirements.txt
streamlit run app.py
'''

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Tech stack

- **Streamlit** — web UI
- **pandas / numpy** — data loading & cleaning
- **scikit-learn** — preprocessing, models, metrics
- **XGBoost** — gradient boosting model (optional; app runs fine without it)

## Models compared

| Type            | Models |
|-----------------|--------|
| Classification  | Logistic Regression, Random Forest, Decision Tree, KNN, SVM, XGBoost |
| Regression      | Linear Regression, Random Forest, Decision Tree, KNN, SVM, XGBoost |

The app auto-detects which set to use based on the target column you select.

## Project structure

'''
 app.py              # main Streamlit application
 requirements.txt    # Python dependencies
 README.md
'''

## Notes

- If XGBoost isn't installed, the app skips it silently rather than erroring out.
- Best suited to small-to-medium tabular datasets (a few thousand rows) since models are trained live in the browser session.
