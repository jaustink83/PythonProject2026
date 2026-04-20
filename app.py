import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import (LoginManager, login_user, logout_user,
                         login_required, current_user)
from flask_bcrypt import Bcrypt
from flask_moment import Moment

from db import db, User, Prediction
from ml_model import get_model, ALL_ALLOWED_FIELDS


# Setup Flask to run the application website
app = Flask(__name__)
# Set session secret
app.secret_key = 'student-grade-secret'

# Load zip.  This lets us combine multiple objects into a single return
app.jinja_env.globals.update(zip=zip)

# Database parameters — injected at runtime from ECS task environment / Secrets Manager
DB_USER = os.environ.get('DB_USER', 'gradeuser')
DB_PASS = os.environ.get('DB_PASS', 'gradepassword')
DB_HOST = os.environ.get('DB_HOST', 'localhost')
DB_PORT = os.environ.get('DB_PORT', '3306')
DB_NAME = os.environ.get('DB_NAME', 'gradepredictor')

# Set the DB URL
app.config['SQLALCHEMY_DATABASE_URI'] = (
    f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)
# We don't need teh default Tracking that Alchemy provides
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# allows the Flask app to use the DB
db.init_app(app)
# Allows the Flask app to use BCrypt to hash the passwords
bcrypt = Bcrypt(app)
# Allows the Flask app to reference the moment instance to determine local timezone
moment = Moment(app)

# Sets the loginManager for the Flask app
login_manager = LoginManager(app)
# Where to send user to login
login_manager.login_view         = 'login'
# Messge to display to users who need to login
login_manager.login_message      = 'Please log in to access that page.'
# Message type
login_manager.login_message_category = 'info'

# Defines a call-back for Flask to load a user
@login_manager.user_loader
def load_user(user_id: str):
    return db.session.get(User, int(user_id))

# Preload the data model to speed up the website for the user.
# Also loads the DB classes for use within the application
with app.app_context():
    db.create_all()
    get_model()   # warm up ML model at startup



# ====================================================================
# Auth routes
# ====================================================================

# GET - If the user is already logged in, redirect to index.html else send to register.html
# POST - Process the form data to register the user into the DB
@app.route('/register', methods=['GET', 'POST'])
def register():
    # If user is already logged in redirect them to the index page
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    # This is a POST call so parse the form data, validate it, and register the user
    if request.method == 'POST':

        # Get all the form data
        username = request.form.get('username', '').strip()
        email    = request.form.get('email',    '').strip().lower()
        password = request.form.get('password', '')
        confirm  = request.form.get('confirm',  '')

        # ----------Basic validation on the data the user entered--------------
        if not username or not email or not password:
            flash('All fields are required.', 'danger')
            return redirect(url_for('register'))

        if password != confirm:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('register'))

        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'danger')
            return redirect(url_for('register'))

        if User.query.filter_by(username=username).first():
            flash('That username is already taken.', 'danger')
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash('An account with that email already exists.', 'danger')
            return redirect(url_for('register'))
        # -------------------End validation------------------------

        # Hash the password so we don't store them in plain text
        hashed = bcrypt.generate_password_hash(password).decode('utf-8')

        # Create the User object to prepare it for the DB
        user   = User(username=username, email=email, password_hash=hashed)
        # Add the User to the DB and commit
        db.session.add(user)
        db.session.commit()

        # Display the success message
        flash('Account created! You can now log in.', 'success')

        # Redirect user to login page
        return redirect(url_for('login'))

    # Display the Register page
    return render_template('register.html')

# GET - If the user is already logged in, redirect to index.html else send to login.html
# POST - Process the form data to log the user in
@app.route('/login', methods=['GET', 'POST'])
def login():

    # If user is already logged in redirect them to the index page
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    # This is a POST call so parse the form data, validate it, and log the user in
    if request.method == 'POST':

        #Get the form data
        username   = request.form.get('username', '').strip()
        password   = request.form.get('password', '')
        remember   = bool(request.form.get('remember'))

        #Lookup the username provided in the DB
        user       = User.query.filter_by(username=username).first()

        # Test if
        #   the username exists in the DB
        #   the password provided matches what is stored in the DB
        if user and bcrypt.check_password_hash(user.password_hash, password):
            # flask_login: create a session for the User
            login_user(user, remember=remember)
            # flask_login: if the user was redirected here while trying to get to a differnet page
            # get that page so we knmow where to send them
            next_page = request.args.get('next')
            # If next_page exists, redirect the user to that page, else send them to index.html
            return redirect(next_page or url_for('index'))

        # We didn't get logged in, register an error message
        flash('Incorrect username or password.', 'danger')

    #Load the login page
    return render_template('login.html')

#Logs teh user out of hte application
@app.route('/logout')
@login_required
def logout():
    # flask_login: invalidate the session for the User
    logout_user()

    # Register the successful logout message
    flash('You have been logged out.', 'info')

    # Redirect the user to the login page
    return redirect(url_for('login'))

# ====================================================================
# Main app routes
# ====================================================================

# When user goes to the root of the website
#   load predict.html
#   pass ALL_ALLOWED_FIELDS and recent to the website,
#   use the ZIP package to compress the data into a single object.
@app.route('/', methods=['GET'])
@login_required  #User must be logged in
def index():
    # Query the 5 most recent predictions so we can load them on the sidebar
    recent = (Prediction.query
              .filter_by(user_id=current_user.id)
              .order_by(Prediction.created_at.desc())
              .limit(5)
              .all())
    # Load predict.html and
    return render_template('predict.html', fields=ALL_ALLOWED_FIELDS,
                           recent=recent, zip=zip)

# When the user submits the form to /predict.html get the data, and store it into a Dict object
@app.route('/predict', methods=['POST'])
@login_required #User must be logged in
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

    # If the user didn't enter data into ANY field, load a Flash error message and return to the main page to display it.
    if not user_inputs:
        flash('Please fill in at least one field before predicting.', 'warning')
        return redirect(url_for('index'))

    # Ensure model is instantiated and kick off the prediction logic by passing in the Dict object with all the user data
    model  = get_model()
    result = model.predict(user_inputs)

    # Save to DB and commit
    pred = Prediction(
        user_id         = current_user.id,
        inputs_json     = json.dumps(user_inputs),
        predicted_score = result['score'],
        grade_letter    = result['letter'],
        percentage      = result['percentage'],
    )
    db.session.add(pred)
    db.session.commit()

    # Print basic output to console for debugging
    print(f"Saved prediction #{pred.id} for user {current_user.username}: "
          f"{result['score']}/20 ({result['letter']})")

    # Take the result of the prediction model and send it all to the /results.html page to be displayed
    # Adding Prediction ID generated from the DB
    return render_template(
        'results.html',
        result=result,
        inputs=user_inputs,
        fields=ALL_ALLOWED_FIELDS,
        prediction_id=pred.id)

# When the user loads /history.html get the historical predictions data, and store it into a Dict object
@app.route('/history')
@login_required
def history():
    # Look in the URL for ?page=#, get that number if it exists or use the default value of 1
    page  = request.args.get('page', 1, type=int)

    # Query the Predictions DB class for historical predictions for the logged in user
    #   Order by Date Created (newest first)
    #   Display 15 results per page
    preds = (Prediction.query
             .filter_by(user_id=current_user.id)
             .order_by(Prediction.created_at.desc())
             .paginate(page=page, per_page=15, error_out=False))

    #Load the history.html page passing in the historical predictions
    return render_template('history.html', preds=preds)

# Start the Flask website.
app.run(debug=True, host='0.0.0.0', port=5000)
