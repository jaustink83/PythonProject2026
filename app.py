"""
Routes:
  GET  /          → render input form
  POST /predict   → run model, print result to console, show confirmation page
"""

from flask import Flask, render_template, request, redirect, url_for, flash
from ml_model import get_model, ALL_ALLOWED_FIELDS

app = Flask(__name__)
app.jinja_env.globals.update(zip=zip)
app.secret_key = 'release1-dev-secret'


# Preload the model when the app starts so the first request is faster.
with app.app_context():
    get_model()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/', methods=['GET'])
def index():
    """Render the prediction form."""
    return render_template('predict.html', fields=ALL_ALLOWED_FIELDS)


@app.route('/predict', methods=['POST'])
def predict():
    """Collect form data, run model, print result to console."""
    form_data = request.form.to_dict()

    # Build user_inputs from only the fields that were actually filled in.
    user_inputs = {}
    for field, meta in ALL_ALLOWED_FIELDS.items():
        val = form_data.get(field, '').strip()
        if val == '':
            continue  # Leave blank
        try:
            if meta['type'] == 'number':
                user_inputs[field] = float(val)
            else:
                try:
                    user_inputs[field] = float(val)
                except ValueError:
                    user_inputs[field] = val
        except Exception as exc:
            print(f"[predict] Skipping field {field!r}: {exc}")

    if not user_inputs:
        flash('Please fill in at least one field before predicting.', 'warning')
        return redirect(url_for('index'))

    model  = get_model()
    result = model.predict(user_inputs)

    # ── Console output ──────────────────
    print("\n" + "=" * 55)
    print("  GRADE PREDICTION RESULT")
    print("=" * 55)
    print(f"  Predicted grade : {result['score']}/20")
    print(f"  Percentage      : {result['percentage']}%")
    print(f"  Letter grade    : {result['letter']}")
    print("-" * 55)
    print(f"  Fields provided : {', '.join(result['fields_used']) or 'none'}")
    print("=" * 55 + "\n")
    # ────────────────────────────────────────────────────────────────────

    return render_template(
        'result_console.html',
        result=result,
        inputs=user_inputs,
        fields=ALL_ALLOWED_FIELDS,
    )


# ---------------------------------------------------------------------------

app.run(debug=True, host='0.0.0.0', port=5000)