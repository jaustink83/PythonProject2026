"""
Added generate_whatif() and _feature_importance.
predict() now returns the full result.
Added detailed comments.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from ucimlrepo import fetch_ucirepo

"""
ALL_ALLOWED_FIELDS is a dist that matches the field in the UCI ML Dataset
It also defines how the fields are displayed on the predict.html page
"""
ALL_ALLOWED_FIELDS = {
    'sex':        {'type': 'select',  'options': ['M', 'F'],           'label': 'Sex'},
    'age':        {'type': 'number',  'min': 15, 'max': 22,            'label': 'Age'},
    'studytime':  {'type': 'select',  'options': [1, 2, 3, 4],
                   'option_labels': ['< 2 hrs/week', '2–5 hrs/week',
                                     '5–10 hrs/week', '> 10 hrs/week'], 'label': 'Weekly Study Time'},
    'failures':   {'type': 'number',  'min': 0, 'max': 4,              'label': 'Past Class Failures'},
    'absences':   {'type': 'number',  'min': 0, 'max': 93,             'label': 'Number of Absences'},
    'freetime':   {'type': 'select',  'options': [1, 2, 3, 4, 5],
                   'option_labels': ['Very low', 'Low', 'Medium', 'High', 'Very high'],
                                                                        'label': 'Free Time After School'},
    'goout':      {'type': 'select',  'options': [1, 2, 3, 4, 5],
                   'option_labels': ['Very low', 'Low', 'Medium', 'High', 'Very high'],
                                                                        'label': 'Going Out with Friends'},
    'Dalc':       {'type': 'select',  'options': [1, 2, 3, 4, 5],
                   'option_labels': ['Very low', 'Low', 'Medium', 'High', 'Very high'],
                                                                        'label': 'Weekday Alcohol Use'},
    'Walc':       {'type': 'select',  'options': [1, 2, 3, 4, 5],
                   'option_labels': ['Very low', 'Low', 'Medium', 'High', 'Very high'],
                                                                        'label': 'Weekend Alcohol Use'},
    'health':     {'type': 'select',  'options': [1, 2, 3, 4, 5],
                   'option_labels': ['Very bad', 'Bad', 'OK', 'Good', 'Very good'],
                                                                        'label': 'Current Health Status'},
    'internet':   {'type': 'select',  'options': ['yes', 'no'],        'label': 'Internet Access at Home'},
    'romantic':   {'type': 'select',  'options': ['yes', 'no'],        'label': 'In a Romantic Relationship'},
    'activities': {'type': 'select',  'options': ['yes', 'no'],        'label': 'Extracurricular Activities'},
    'Medu':       {'type': 'select',  'options': [0, 1, 2, 3, 4],
                   'option_labels': ['None', 'Primary (4th grade)',
                                     'Middle school (5th–9th)',
                                     'Secondary school', 'Higher education'],
                                                                        'label': "Mother's Education"},
    'Fedu':       {'type': 'select',  'options': [0, 1, 2, 3, 4],
                   'option_labels': ['None', 'Primary (4th grade)',
                                     'Middle school (5th–9th)',
                                     'Secondary school', 'Higher education'],
                                                                        'label': "Father's Education"},
    'higher':     {'type': 'select',  'options': ['yes', 'no'],        'label': 'Wants Higher Education'},
    'famsup':     {'type': 'select',  'options': ['yes', 'no'],        'label': 'Family Educational Support'},
}

"""
These are the fields from the data set that the user has control to change.
This excludes things out of the user's control.
"""
CONTROLLABLE_FIELDS = [
    'studytime', 'absences', 'failures', 'freetime',
    'goout', 'Dalc', 'Walc', 'health', 'activities', 'internet', 'romantic',
]

"""
This function converts the European grading scale that the data set is built on 
into a U.S. grading scale that our users will better understand.
"""
def interpret_grade(score) -> dict:
    score = int(round(float(score)))
    score = max(0, min(20, score))
    percentage = score * 5
    if score >= 16:   letter, css_class = 'A', 'grade-a'
    elif score >= 14: letter, css_class = 'B', 'grade-b'
    elif score >= 12: letter, css_class = 'C', 'grade-c'
    elif score >= 10: letter, css_class = 'D', 'grade-d'
    else:             letter, css_class = 'F', 'grade-f'
    return {'score': score, 'letter': letter, 'percentage': percentage,
            'css_class': css_class, 'out_of': 20}

"""
This is the class that loads the data set, trains the model, and predicts the results
"""
class GradeModel:

    """
    Initializes the class with 3 fields
    model - will store the trained model from the UCI ML dataset we can use to predict user's outcome.
    imputer - how we will deal with fields not supplied by the user.  We have this turned off for now as it skews the results.
    train_columns - The list of columns that exist in the dataset and have values supplied by the user.
    """
    def __init__(self):
        self.model         = None
        self.imputer       = SimpleImputer(strategy='mean')
        self.train_columns = None

    """
    This method takes in the list of user supplied fields, loads the dataset, and trains the model
    """
    def _load_and_train(self, user_inputs: dict):

        # Load the UCI ML dataset for Student Grade prediction
        try:
            # Get Data from UCI ML Repo
            dataset = fetch_ucirepo(id=320)
        except ConnectionError:
            # If the UCI ML repo is down/offline load the data from local disk as a backup

            # Locally teh data is in 2 CSV files, load them both
            student_mat = pd.read_csv("student-mat.csv", sep=";")
            student_por = pd.read_csv("student-por.csv", sep=";")

            # Our target is to predict Final Grade, this is defined as 'G3' in the dataset
            target_col = "G3"

            # We need these 2 nested classes to mimic how fetch_ucirepo functions with our local data
            class SimpleDataset:
                class data:
                    pass

            # Use the nested class to make an instance of each
            dataset = SimpleDataset()
            dataset.data = SimpleDataset.data()

            # Here we combine the data from both CSV files into a single DataFrame
            df = pd.concat([student_mat, student_por], ignore_index=True)
            # Drop 'G3' from the dataset.  Don't use it to predict
            dataset.data.features = df.drop(columns=[target_col])
            # Set 'G3' as the filed we want to predict in the dataset object
            dataset.data.targets  = df[[target_col]]

        # Pull out all the data in the model
        X           = dataset.data.features
        # Pull out the target column from the model
        y           = dataset.data.targets['G3']
        # Determine what columns from the model for which the user also provided values
        allowed     = [c for c in user_inputs if c in X.columns]
        # Keep only the data from the model for which the user supplied values of their own
        X           = X[allowed]
        # Drop the 1st row of data, they are headers
        X           = pd.get_dummies(X, drop_first=True)

        """
        This section is supplied by the UCI ML repository for how to properly train the data model
        """
        self.train_columns = X.columns.tolist()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.imputer.fit(X_train)

        X_tr = pd.DataFrame(self.imputer.transform(X_train), columns=self.train_columns)
        self.model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        self.model.fit(X_tr, y_train)

    """
    This function takes the user_input dict object, all the user supplied data
    and converts it into a single row DataFrame, which is required for teh prediction model
    """
    def _build_row(self, user_inputs: dict) -> pd.DataFrame:
        row = pd.DataFrame([user_inputs])
        row = pd.get_dummies(row, drop_first=True)
        row = row.reindex(columns=self.train_columns)
        return pd.DataFrame(self.imputer.transform(row), columns=self.train_columns)

    """
    This function takes the user data.  For the columns the user provided data (max 5)
    the function will generate the prediction for all possible values for this these fields to
    generate a 'What If...' chart and allows the user to see how their prediction would change
    if they changed their behavior.
    """
    def _generate_whatif(self, base_row: pd.DataFrame, user_inputs: dict) -> dict:
        """
        For each controllable field the user provided, sweep its full range
        and predict the grade at each step.  Returns a dict keyed by field name.
        """
        scenarios = {}
        provided_controllable = [f for f in CONTROLLABLE_FIELDS if f in user_inputs]

        for field in provided_controllable[:5]:          # cap at 5 charts
            meta   = ALL_ALLOWED_FIELDS.get(field, {})
            if meta.get('type') == 'select':
                values = meta['options']
                x_labels = meta.get('option_labels', [str(v) for v in values])
            else:
                values   = list(range(int(meta.get('min', 0)),
                                      int(meta.get('max', 10)) + 1))
                x_labels = [str(v) for v in values]

            grades = []
            for val in values:
                test_row = base_row.copy()
                # Zero out all dummified variants of this field
                for col in self.train_columns:
                    if col == field or col.startswith(field + '_'):
                        test_row[col] = 0
                # Set the correct column
                if field in self.train_columns:
                    test_row[field] = val
                elif f"{field}_{val}" in self.train_columns:
                    test_row[f"{field}_{val}"] = 1
                grades.append(int(round(float(self.model.predict(test_row)[0]))))

            # Mark which bar is the user's current value
            user_val   = user_inputs.get(field)
            highlights = []
            for val in values:
                try:
                    match = float(val) == float(user_val)
                except (TypeError, ValueError):
                    match = str(val) == str(user_val)
                highlights.append(match)

            scenarios[field] = {
                'label':      ALL_ALLOWED_FIELDS[field]['label'],
                'x_labels':   x_labels,
                'grades':     grades,
                'highlights': highlights,
                'user_value': str(user_val),
            }
        return scenarios

    """
    This function uses the model data to determine the top 8 impactful columns
    It ranks them from most to least impactful and provide how impactful each column is
    """
    def _feature_importances(self, top_n: int = 8) -> dict:
        pairs = sorted(
            zip(self.train_columns, self.model.feature_importances_.tolist()),
            key=lambda x: x[1], reverse=True
        )[:top_n]
        return {
            'labels': [p[0] for p in pairs],
            'values': [round(p[1], 4) for p in pairs],
        }

    """
    This function is the main control in the prediction model.
    The user's data is input and the logic does the following: 
    - Load and train the UCI ML model
    - Converts the web data (dict) to DataFrame which the model requires
    - Predicts the user's 'Final Grade'
    - Retain a list of the following:
        - Columns Used/Provided by User
        - 'What If' Chart.js data to display to user
        - 'Importance' data to show which fields impact the predict the most
    """
    def predict(self, user_inputs: dict) -> dict:

        self._load_and_train(user_inputs)
        row_imp  = self._build_row(user_inputs)
        raw_pred = self.model.predict(row_imp)[0]
        result   = interpret_grade(raw_pred)
        result['fields_used']    = list(user_inputs.keys())
        #result['fields_imputed'] = [c for c in ALL_ALLOWED_FIELDS if c not in user_inputs]
        result['whatif']         = self._generate_whatif(row_imp, user_inputs)
        result['importances']    = self._feature_importances()
        return result


# This is the instance of the Grade Prediction Model.  Prevents the user from making multiple instance of the same class
_model_instance: GradeModel | None = None

"""
app.py will call this to instantiate the GradeModel.  The model will load into _model_instance and be accessible to app.py
If the GradeModel has already been created previously, the same instance will be returned so app.py can access the values 
"""
def get_model() -> GradeModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = GradeModel()
    return _model_instance