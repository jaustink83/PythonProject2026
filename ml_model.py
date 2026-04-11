import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from ucimlrepo import fetch_ucirepo

# ---------------------------------------------------------------------------
# Field definitions for HTML and model.
# Each entry describes how the field should render and what values are valid.
# ---------------------------------------------------------------------------
ALL_ALLOWED_FIELDS = {
    'sex':          {'type': 'select',  'options': ['M', 'F'],              'label': 'Sex'},
    'age':          {'type': 'number',  'min': 15, 'max': 22,               'label': 'Age'},
    'studytime':    {'type': 'select',  'options': [1, 2, 3, 4],
                     'option_labels': ['< 2 hrs/week', '2–5 hrs/week',
                     '5–10 hrs/week', '> 10 hrs/week'],                     'label': 'Weekly Study Time'},
    'failures':     {'type': 'number',  'min': 0, 'max': 4,                 'label': 'Past Class Failures'},
    'absences':     {'type': 'number',  'min': 0, 'max': 93,                'label': 'Number of Absences'},
    'freetime':     {'type': 'select',  'options': [1, 2, 3, 4, 5],
                     'option_labels': ['Very low', 'Low', 'Medium', 'High', 'Very high'],
                                                                            'label': 'Free Time After School'},
    'goout':        {'type': 'select',  'options': [1, 2, 3, 4, 5],
                     'option_labels': ['Very low', 'Low', 'Medium', 'High', 'Very high'],
                                                                            'label': 'Going Out with Friends'},
    'Dalc':         {'type': 'select',  'options': [1, 2, 3, 4, 5],
                     'option_labels': ['Very low', 'Low', 'Medium', 'High', 'Very high'],
                                                                            'label': 'Weekday Alcohol Use'},
    'Walc':         {'type': 'select',  'options': [1, 2, 3, 4, 5],
                     'option_labels': ['Very low', 'Low', 'Medium', 'High', 'Very high'],
                                                                            'label': 'Weekend Alcohol Use'},
    'health':       {'type': 'select',  'options': [1, 2, 3, 4, 5],
                     'option_labels': ['Very bad', 'Bad', 'OK', 'Good', 'Very good'],
                                                                            'label': 'Current Health Status'},
    'internet':     {'type': 'select',  'options': ['yes', 'no'],           'label': 'Internet Access at Home'},
    'romantic':     {'type': 'select',  'options': ['yes', 'no'],           'label': 'In a Romantic Relationship'},
    'activities':   {'type': 'select',  'options': ['yes', 'no'],           'label': 'Extracurricular Activities'},
    'Medu':         {'type': 'select',  'options': [0, 1, 2, 3, 4],
                     'option_labels': ['None', 'Primary (4th grade)',
                                       'Middle school (5th–9th)',
                                       'Secondary school', 'Higher education'],
                                                                            'label': "Mother's Education"},
    'Fedu':         {'type': 'select',  'options': [0, 1, 2, 3, 4],
                     'option_labels': ['None', 'Primary (4th grade)',
                                       'Middle school (5th–9th)',
                                       'Secondary school', 'Higher education'],
                                                                            'label': "Father's Education"},
    'higher':       {'type': 'select',  'options': ['yes', 'no'],           'label': 'Wants Higher Education'},
    'famsup':       {'type': 'select',  'options': ['yes', 'no'],           'label': 'Family Educational Support'},
}

# Fields the user can actually change
CONTROLLABLE_FIELDS = [
    'studytime', 'absences', 'failures',
    'freetime', 'goout', 'Dalc', 'Walc',
    'health', 'activities', 'internet',
    'romantic',
]

def interpret_grade(score: int) -> dict:
    """Convert a 0–20 integer score to letter grade, percentage and colour hint."""
    score = int(round(float(score)))
    score = max(0, min(20, score))  #Score can go above 20 and under 0.  This keeps the 0-20 range intact.
    percentage = score * 5
    if score >= 16:
        letter, css_class = 'A', 'grade-a'
    elif score >= 14:
        letter, css_class = 'B', 'grade-b'
    elif score >= 12:
        letter, css_class = 'C', 'grade-c'
    elif score >= 10:
        letter, css_class = 'D', 'grade-d'
    else:
        letter, css_class = 'F', 'grade-f'
    return {
        'score':      score,
        'letter':     letter,
        'percentage': percentage,
        'css_class':  css_class,
        'out_of':     20,
    }


class GradeModel:
    """
    Loads the UCI Student Performance dataset,
    Trains a RandomForestClassifier on the subset of features exposed via ALL_ALLOWED_FIELDS
    Provides a predict() method that accepts any subset of those features from user input
    """

    def __init__(self):
        self.model        = None
        self.imputer      = SimpleImputer(strategy='mean')
        self.train_columns = None

    # ------------------------------------------------------------------
    def _load_and_train(self, user_inputs: dict):
        #Get Data from UCI ML Repo
        #dataset = fetch_ucirepo(id=320)

        #Get data loacally
        student_mat = pd.read_csv("student-mat.csv", sep=";")
        student_por = pd.read_csv("student-por.csv", sep=";")
        target_col = "G3"

        class SimpleDataset:
            class data:
                pass

        dataset = SimpleDataset()
        dataset.data = SimpleDataset.data()

        # Use Math dataset as primary (common choice); swap for student_por if needed
        df = pd.concat([student_mat, student_por], ignore_index=True)
        dataset.data.features = df.drop(columns=[target_col])
        dataset.data.targets  = df[[target_col]]

        X = dataset.data.features
        y = dataset.data.targets['G3']

        # Keep only fields we expose to users
        #allowed_cols = [c for c in ALL_ALLOWED_FIELDS.keys() if c in X.columns]
        allowed_cols = [c for c in user_inputs if c in X.columns]
        X = X[allowed_cols]

        # Drop the columns that have data type incompatible with sklearn
        X = pd.get_dummies(X, drop_first=True)
        self.train_columns = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.imputer.fit(X_train)
        X_train_imp = pd.DataFrame(
            self.imputer.transform(X_train), columns=self.train_columns
        )
        self.model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        self.model.fit(X_train_imp, y_train)

        # Quick accuracy report to console
        X_test_imp = pd.DataFrame(
            self.imputer.transform(X_test), columns=self.train_columns
        )
        acc = self.model.score(X_test_imp, y_test)
        print(f"[GradeModel] Test accuracy (exact match): {acc:.2%}")

    # ------------------------------------------------------------------
    def _build_row(self, user_inputs: dict) -> pd.DataFrame:
        """Turn a raw user_inputs dict into an imputed, aligned DataFrame row."""
        row = pd.DataFrame([user_inputs])
        row = pd.get_dummies(row, drop_first=True)
        row = row.reindex(columns=self.train_columns)
        row_imp = pd.DataFrame(
            self.imputer.transform(row), columns=self.train_columns
        )
        return row_imp

    # ------------------------------------------------------------------
    def predict(self, user_inputs: dict) -> dict:
        """
        Accept a dict of any subset of ALL_ALLOWED_FIELDS values.
        Returns a dict with grade info.  (R1 only — no what-if or importances.)
        """
        self._load_and_train(user_inputs)
        row_imp  = self._build_row(user_inputs)
        raw_pred = self.model.predict(row_imp)[0]
        result   = interpret_grade(raw_pred)
        result['fields_used']    = list(user_inputs.keys())
        #result['fields_imputed'] = [
        #    c for c in ALL_ALLOWED_FIELDS if c not in user_inputs
        #]
        return result


# ---------------------------------------------------------------------------
# Module-level singleton — created once at import time inside app context.
# ---------------------------------------------------------------------------
_model_instance: GradeModel | None = None


def get_model() -> GradeModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = GradeModel()
    return _model_instance
