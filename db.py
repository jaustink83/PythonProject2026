"""
The db object is initialised here and imported into app.py.
"""

import json
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()


class User(UserMixin, db.Model):
    """Registered user account."""
    __tablename__ = 'users'

    id            = db.Column(db.Integer,     primary_key=True)
    username      = db.Column(db.String(80),  unique=True,  nullable=False)
    email         = db.Column(db.String(120), unique=True,  nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at    = db.Column(db.DateTime,    default=datetime.utcnow)

    predictions   = db.relationship('Prediction', backref='user', lazy='dynamic',
                                    cascade='all, delete-orphan')

    def __repr__(self):
        return f'<User {self.username!r}>'


class Prediction(db.Model):
    """One saved prediction result from a user."""
    __tablename__ = 'predictions'

    id              = db.Column(db.Integer,    primary_key=True)
    user_id         = db.Column(db.Integer,    db.ForeignKey('users.id'), nullable=False)
    inputs_json     = db.Column(db.Text,       nullable=False)   # JSON-encoded user_inputs dict
    predicted_score = db.Column(db.Integer,    nullable=False)
    grade_letter    = db.Column(db.String(2),  nullable=False)
    percentage      = db.Column(db.Integer,    nullable=False)
    created_at      = db.Column(db.DateTime,   default=datetime.utcnow)

    def get_inputs(self) -> dict:
        """Convert the stored JSON inputs back to a Python dict."""
        try:
            return json.loads(self.inputs_json)
        except Exception:
            return {}

    def __repr__(self):
        return f'<Prediction user={self.user_id} score={self.predicted_score}>'
