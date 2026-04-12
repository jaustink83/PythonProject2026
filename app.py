"""
/predict now renders results.html with Chart.js charts
instead of the plain console-confirmation page.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, render_template, request, redirect, url_for, flash
from ml_model import get_model, ALL_ALLOWED_FIELDS

# Setup Flask to run the application website
app = Flask(__name__)
# Set session secret
app.secret_key = 'student-grade-secret'

# Load zip.  This lets us combine multiple objects into a single return
app.jinja_env.globals.update(zip=zip)

# Preload the data model to speed up the website for the user.
with app.app_context():
    get_model()

# When user goes to the root of the website, load predict.html, pass ALL_ALLOWED_FIELDS to the website, use the ZIP package to compress the data into a single object.
@app.route('/', methods=['GET'])
def index():
    return render_template('predict.html', fields=ALL_ALLOWED_FIELDS, zip=zip)

# When the user submits the form to /predict.html get the data, and store it into a Dict object
@app.route('/predict', methods=['POST'])
def predict():
    form_data   = request.form.to_dict()
    user_inputs = {}

    for field, meta in ALL_ALLOWED_FIELDS.items():
        val = form_data.get(field, '').strip()
        if val == '':
            continue
        try:
            if meta['type'] == 'number':
                user_inputs[field] = float(val)
            else:
                try:
                    user_inputs[field] = float(val)
                except ValueError:
                    user_inputs[field] = val
        except Exception as exc:
            print(f"[predict] Skipping {field!r}: {exc}")

    # If the user didnt enter data into ANY field, load a Flash error message and return to the main page to display it.
    if not user_inputs:
        flash('Please fill in at least one field before predicting.', 'warning')
        return redirect(url_for('index'))

    # Ensure model is instatiated and kick off the prediction logic by passing in the Dict object with all the user data
    model  = get_model()
    result = model.predict(user_inputs)

    # Print basic output to console for debugging
    print(f"\n[R2] Prediction: {result['score']}/20 ({result['letter']})")

    # Take the result of the prediction model and send it all to the /results.html page to be displayed
    return render_template(
        'results.html',
        result=result,
        inputs=user_inputs,
        fields=ALL_ALLOWED_FIELDS,
    )

# Start the Flask website.
app.run(debug=True, host='0.0.0.0', port=5000)
