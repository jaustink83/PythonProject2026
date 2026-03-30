from sklearn.impute import SimpleImputer
from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def interpret_grade(score):
    percentage = score * 5
    if score >= 16:
        letter = 'A'
    elif score >= 14:
        letter = 'B'
    elif score >= 12:
        letter = 'C'
    elif score >= 10:
        letter = 'D'
    else:
        letter = 'F'
    return f"{score}/20 ({percentage}%) - Grade: {letter}"

imputer = SimpleImputer(strategy='mean')
my_features = ['sex', 'studytime', 'age', 'absences', 'failures']
# sex - M or F
# age - 15 through 22
# studytime - weekly study time in hours
# absences - number of missed classes
# failures - number

# Fetch dataset
student_performance = fetch_ucirepo(id=320)

# Extract features and targets
X = student_performance.data.features #[my_features]
y = student_performance.data.targets['G3'] # Using G3 (final grade) as target

# The dataset has many string/object types that sklearn cannot handle directly
X = pd.get_dummies(X, drop_first=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

imputer.fit(X_train)

# Initialize and train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prints required fields
print(model.feature_names_in_)

new_student = pd.DataFrame([{
    'studytime': 4,
    'age': 17,
    'absences': 0,
    'failures': 0,
    'sex_M': 1,
}])

new_student = new_student.reindex(columns=X_train.columns)
new_student_imputed = imputer.transform(new_student)
new_student_final = pd.DataFrame(new_student_imputed, columns=X_train.columns)

prediction = model.predict(new_student_final)
print(f"Predicted Grade: {interpret_grade(prediction[0])}")