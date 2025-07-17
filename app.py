# app.py
from flask import Flask, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
import os
from flask_cors import CORS
import joblib # Import joblib for loading models
import pandas as pd # Import pandas for data manipulation
import numpy as np # Import numpy for numerical operations
import random # Import random for dummy data generation
from nltk.sentiment.vader import SentimentIntensityAnalyzer # Ensure this is imported here if used directly
import threading # For simulating async report generation
import time # For simulated delays
from io import BytesIO # To simulate file generation
import uuid 

# Import prediction functions and necessary feature lists from generate_train_predict.py
from generate_train_predict import (
    predict_churn,
    predict_claims_submission,
    predict_segmentation,
    predict_sentiment,
    predict_risk_score,
    recommend_retention_actions,
    predict_fraud,
    features_churn,
    categorical_features_churn,
    features_claims_forecast,
    categorical_features_claims_forecast,
    features_segmentation,
    features_risk,
    categorical_features_risk,
    features_fraud,
    categorical_features_fraud
)


# --- Flask App Configuration ---
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
app.config['JWT_SECRET_KEY'] = 'super-secret'  # Change this in production
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit
CORS(app)

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Ensure models folder exists and load models
MODELS_DIR = 'models'
DATASETS_DIR = 'datasets'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True) # Ensure datasets folder also exists

# --- Global Model Variables (Load models once when app starts) ---
churn_model = None
claims_forecast_model = None
client_segmentation_model = None
scaler_segmentation = None
cluster_mapping = None
risk_score_model = None
fraud_detection_model = None
preprocessor_fraud = None
recommendation_model = None # This will now be cf_model
client_id_to_encoded = None
encoded_to_client_id = None
action_type_to_encoded = None
encoded_to_action_type = None
sparse_item_user = None
sentiment_analyzer = None # For direct use in app.py if needed, though functions now wrap it


def load_ml_models():
    global churn_model, claims_forecast_model, client_segmentation_model, \
           scaler_segmentation, cluster_mapping, risk_score_model, \
           fraud_detection_model, preprocessor_fraud, recommendation_model, \
           client_id_to_encoded, encoded_to_client_id, action_type_to_encoded, \
           encoded_to_action_type, sparse_item_user, sentiment_analyzer

    print("Attempting to load ML models...")
    try:
        churn_model = joblib.load(os.path.join(MODELS_DIR, 'churn_prediction_model.pkl'))
        print("Churn Prediction Model loaded.")
    except FileNotFoundError:
        print("Churn Prediction Model not found. Please run generate_train_predict.py.")
    
    try:
        claims_forecast_model = joblib.load(os.path.join(MODELS_DIR, 'claims_forecast_model.pkl'))
        print("Claims Forecasting Model loaded.")
    except FileNotFoundError:
        print("Claims Forecasting Model not found. Please run generate_train_predict.py.")

    try:
        client_segmentation_model = joblib.load(os.path.join(MODELS_DIR, 'client_segmentation_model.pkl'))
        print("Client Segmentation Model loaded.")
    except FileNotFoundError:
        print("Client Segmentation Model not found. Please run generate_train_predict.py.")

    try:
        scaler_segmentation = joblib.load(os.path.join(MODELS_DIR, 'scaler_segmentation.pkl'))
        print("Scaler (for Segmentation) loaded.")
    except FileNotFoundError:
        print("Scaler for Segmentation not found. Please run generate_train_predict.py.")

    try:
        cluster_mapping = joblib.load(os.path.join(MODELS_DIR, 'cluster_mapping.pkl'))
        print("Cluster Mapping loaded.")
    except FileNotFoundError:
        print("Cluster Mapping not found. Please run generate_train_predict.py.")

    try:
        risk_score_model = joblib.load(os.path.join(MODELS_DIR, 'risk_scoring_model.pkl'))
        print("Risk Scoring Model loaded.")
    except FileNotFoundError:
        print("Risk Scoring Model not found. Please run generate_train_predict.py.")

    try:
        fraud_detection_model = joblib.load(os.path.join(MODELS_DIR, 'fraud_detection_model.pkl'))
        print("Fraud Detection Model loaded.")
    except FileNotFoundError:
        print("Fraud Detection Model not found. Please run generate_train_predict.py.")

    try:
        preprocessor_fraud = joblib.load(os.path.join(MODELS_DIR, 'preprocessor_fraud.pkl'))
        print("Preprocessor (for Fraud) loaded.")
    except FileNotFoundError:
        print("Preprocessor for Fraud not found. Please run generate_train_predict.py.")

    try:
        recommendation_model = joblib.load(os.path.join(MODELS_DIR, 'collaborative_filtering_model.pkl'))
        client_id_to_encoded = joblib.load(os.path.join(MODELS_DIR, 'client_id_to_encoded.pkl'))
        encoded_to_client_id = joblib.load(os.path.join(MODELS_DIR, 'encoded_to_client_id.pkl'))
        action_type_to_encoded = joblib.load(os.path.join(MODELS_DIR, 'action_type_to_encoded.pkl'))
        encoded_to_action_type = joblib.load(os.path.join(MODELS_DIR, 'encoded_to_action_type.pkl'))
        sparse_item_user = joblib.load(os.path.join(MODELS_DIR, 'sparse_item_user.pkl'))
        print("Recommendation Models (CF, mappings, sparse matrix) loaded.")
    except FileNotFoundError as e:
        print(f"Recommendation Model or its components not found: {e}. Please run generate_train_predict.py.")


    # Initialize NLTK VADER for sentiment analysis
    try:
        sentiment_analyzer = SentimentIntensityAnalyzer()
        print("Sentiment Analyzer (NLTK VADER) initialized.")
    except Exception as e:
        print(f"Could not initialize NLTK VADER: {e}. Ensure 'vader_lexicon' is downloaded (nltk.download('vader_lexicon')).")


# --- Database Models ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String, nullable=False)
    email = db.Column(db.String, unique=True, nullable=False)
    password_hash = db.Column(db.String, nullable=False)
    role = db.Column(db.String, default='user')
    last_login = db.Column(db.String)
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'role': self.role,
            'lastLogin': self.last_login # Changed to camelCase
        }

class Client(db.Model):
    id = db.Column(db.String, primary_key=True)
    full_name = db.Column(db.String, nullable=False)
    email = db.Column(db.String)
    phone_number = db.Column(db.String)
    address = db.Column(db.String)
    age = db.Column(db.Integer)
    gender = db.Column(db.String)
    location = db.Column(db.String)
    registration_date = db.Column(db.String)
    engagement_score = db.Column(db.Float)
    churn_probability = db.Column(db.Float)
    category = db.Column(db.String)
    sentiment_score = db.Column(db.String)
    risk_score = db.Column(db.Float)
    retention_recommendation = db.Column(db.String)
    fraud_flagged = db.Column(db.Boolean, default=False)
    claims = db.relationship('Claim', backref='client', lazy=True)

    def to_dict(self):
        return {
            'id': self.id,
            'fullName': self.full_name, # Changed to camelCase
            'email': self.email,
            'phoneNumber': self.phone_number, # Changed to camelCase
            'address': self.address,
            'age': self.age,
            'gender': self.gender,
            'location': self.location,
            'registrationDate': self.registration_date, # Changed to camelCase
            'engagementScore': self.engagement_score, # Changed to camelCase
            'churnProbability': self.churn_probability, # Changed to camelCase
            'category': self.category,
            'sentimentScore': self.sentiment_score, # Changed to camelCase
            'riskScore': self.risk_score, # Changed to camelCase
            'retentionRecommendation': self.retention_recommendation, # Changed to camelCase
            'fraudFlagged': self.fraud_flagged # Changed to camelCase
        }

class Claim(db.Model):
    id = db.Column(db.String, primary_key=True)
    client_id = db.Column(db.String, db.ForeignKey('client.id'))
    claim_type = db.Column(db.String)
    claim_amount = db.Column(db.Float)
    submission_date = db.Column(db.String)
    claim_status = db.Column(db.String)
    claim_probability = db.Column(db.Float)
    fraud_likelihood_score = db.Column(db.Float)
    is_fraud_flagged = db.Column(db.Boolean, default=False)
    supporting_documents = db.Column(db.PickleType) # Stores list of filenames

    def to_dict(self):
        return {
            'id': self.id,
            'clientId': self.client_id, # Changed to camelCase
            'claimType': self.claim_type, # Changed to camelCase
            'claimAmount': self.claim_amount, # Changed to camelCase
            'submissionDate': self.submission_date, # Changed to camelCase
            'claimStatus': self.claim_status, # Changed to camelCase
            'claimProbability': self.claim_probability, # Changed to camelCase
            'fraudLikelihoodScore': self.fraud_likelihood_score, # Changed to camelCase
            'isFraudFlagged': self.is_fraud_flagged, # Changed to camelCase
            'supportingDocuments': self.supporting_documents # Changed to camelCase
        }

class Notification(db.Model):
    id = db.Column(db.String, primary_key=True) # Changed to String to match frontend UUIDs
    type = db.Column(db.String)
    message = db.Column(db.String)
    date = db.Column(db.String) # Changed to 'date' for frontend consistency
    read = db.Column(db.Boolean, default=False)
    link = db.Column(db.String) # Changed from related_id to link

    def to_dict(self):
        return {
            'id': self.id,
            'type': self.type,
            'message': self.message,
            'date': self.date,
            'read': self.read,
            'link': self.link
        }

class Message(db.Model):
    id = db.Column(db.String, primary_key=True) # Changed to String to match frontend
    sender_id = db.Column(db.String)
    receiver_id = db.Column(db.String)
    content = db.Column(db.String)
    timestamp = db.Column(db.String)
    read = db.Column(db.Boolean, default=False)

    def to_dict(self):
        return {
            'id': self.id,
            'senderId': self.sender_id, # Changed to camelCase
            'receiverId': self.receiver_id, # Changed to camelCase
            'content': self.content,
            'timestamp': self.timestamp,
            'read': self.read
        }

class Report(db.Model): # Added Report Model
    id = db.Column(db.String, primary_key=True)
    title = db.Column(db.String)
    type = db.Column(db.String)
    generated_at = db.Column(db.String)
    status = db.Column(db.String)
    file_type = db.Column(db.String)
    size_bytes = db.Column(db.Integer)

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'type': self.type,
            'generatedAt': self.generated_at, # Changed to camelCase
            'status': self.status,
            'fileType': self.file_type, # Changed to camelCase
            'sizeBytes': self.size_bytes # Changed to camelCase
        }

class SupportTicket(db.Model): # Added SupportTicket Model
    id = db.Column(db.String, primary_key=True)
    subject = db.Column(db.String)
    description = db.Column(db.String)
    priority = db.Column(db.String)
    submitted_at = db.Column(db.String)
    attachments = db.Column(db.PickleType) # Stores list of filename strings

    def to_dict(self):
        return {
            'id': self.id,
            'subject': self.subject,
            'description': self.description,
            'priority': self.priority,
            'submittedAt': self.submitted_at, # Changed to camelCase
            'attachments': self.attachments # This will be list of filenames from backend
        }


# --- Auth Routes ---
@app.route('/auth/login', methods=['POST'])
def login():
    data = request.get_json() # Use get_json for application/json POST
    email = data.get('email')
    password = data.get('password')

    user = User.query.filter_by(email=email).first()
    if user and bcrypt.check_password_hash(user.password_hash, password):
        access_token = create_access_token(identity=user.id, expires_delta=timedelta(days=1))
        user.last_login = datetime.now()
        db.session.commit()
        return jsonify({
            'success': True,
            'data': {
                'id': user.id,
                'name': user.name,
                'email': user.email,
                'role': user.role,
                'token': access_token
            }
        })
    return jsonify({'success': False, 'message': 'Invalid credentials'}), 401

# --- Dashboard Route ---
@app.route('/dashboard', methods=['GET'])
# @jwt_required()
def dashboard():
    # Simulate loading data
    time.sleep(0.5)

    total_clients = Client.query.count()
    total_claims = Claim.query.count()
    pending_claims = Claim.query.filter_by(claim_status='Pending').count()
    approved_claims = Claim.query.filter_by(claim_status='Approved').count()
    
    # Calculate churn rate based on clients flagged with high churn probability
    # Using a threshold of > 70% as per ChurnRiskPage.tsx classification
    churn_clients_count = Client.query.filter(Client.churn_probability > 70).count()
    churn_rate = (churn_clients_count / total_clients) * 100 if total_clients else 0

    fraud_alerts = Claim.query.filter_by(is_fraud_flagged=True).count()

    # Aggregate clients by category for dashboard (PieChartComponent)
    clients_by_category_raw = db.session.query(Client.category, db.func.count(Client.id)).group_by(Client.category).all()
    clients_by_category_data = [{'name': cat, 'value': count} for cat, count in clients_by_category_raw]

    # Claims trend (LineChartComponent) - dummy data for now
    claims_trend_data = []
    # Generate data for the last 8 months to match frontend example's x-axis labels
    for i in range(8):
        # Calculate month and year for current iteration (e.g., Aug, Jul, ..., Jan)
        current_date = datetime.now() - timedelta(days=30 * (7 - i)) # Go back from current month to generate
        month_abbr = current_date.strftime("%b") # e.g., Jan, Feb
        claims_trend_data.append({
            'month': month_abbr, # Matches xAxisDataKey="month"
            'claims': random.randint(15, 40), # Dummy claims count
            'fraud': random.randint(1, 8) # Dummy fraud alerts
        })

    return jsonify({'success': True, 'data': {
        'totalClients': total_clients,
        'totalClaims': total_claims,
        'pendingClaims': pending_claims,
        'approvedClaims': approved_claims,
        'churnRate': round(churn_rate, 2),
        'clientsByCategory': clients_by_category_data, # For PieChartComponent
        'claimsTrend': claims_trend_data, # For LineChartComponent
        'fraudAlerts': fraud_alerts,
    }})

# --- Clients Routes ---
@app.route('/clients', methods=['GET'])
# @jwt_required()
def get_clients():
    time.sleep(0.8) # Simulate network delay
    clients = Client.query.all()
    return jsonify({'success': True, 'data': [c.to_dict() for c in clients]})

@app.route('/clients/<client_id>', methods=['GET'])
# @jwt_required()
def get_client_details(client_id):
    time.sleep(0.5) # Simulate network delay
    client = Client.query.get(client_id)
    if not client:
        return jsonify({'success': False, 'message': 'Client not found'}), 404
    return jsonify({'success': True, 'data': client.to_dict()})

# --- Claims Routes ---
@app.route('/claims', methods=['GET'])
# @jwt_required()
def get_claims():
    time.sleep(0.8) # Simulate network delay
    claims = Claim.query.all()
    return jsonify({'success': True, 'data': [c.to_dict() for c in claims]})

@app.route('/claims/<claim_id>', methods=['GET'])
# @jwt_required()
def get_claim_details(claim_id):
    time.sleep(0.5) # Simulate network delay
    claim = Claim.query.get(claim_id)
    if not claim:
        return jsonify({'success': False, 'message': 'Claim not found'}), 404
    return jsonify({'success': True, 'data': claim.to_dict()})

@app.route('/claims/new', methods=['POST'])
# @jwt_required()
def create_new_claim():
    time.sleep(1.5) # Simulate processing delay
    
    client_id = request.form.get('clientId')
    claim_type = request.form.get('claimType')
    claim_amount = request.form.get('claimAmount')

    if not client_id or not claim_type or not claim_amount:
        return jsonify({'success': False, 'message': 'Missing required claim data (Client ID, Type, Amount).'}), 400

    try:
        claim_amount = float(claim_amount)
        if claim_amount <= 0:
            return jsonify({'success': False, 'message': 'Claim amount must be positive.'}), 400
    except ValueError:
        return jsonify({'success': False, 'message': 'Invalid claim amount format.'}), 400

    client_exists = Client.query.get(client_id)
    if not client_exists:
        return jsonify({'success': False, 'message': f'Client with ID {client_id} not found.'}), 404

    supporting_documents_filenames = []
    if 'supportingDocuments' in request.files:
        files = request.files.getlist('supportingDocuments')
        for file in files:
            if file.filename:
                allowed_extensions = {'png', 'jpg', 'jpeg', 'pdf', 'doc', 'docx'}
                if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    supporting_documents_filenames.append(filename)
                else:
                    print(f"Skipping disallowed file type: {file.filename}")

    new_claim_id = f"CLM-{uuid.uuid4().hex[:8].upper()}"
    
    # --- Integrate Fraud and Claims Probability Prediction ---
    if not fraud_detection_model or not preprocessor_fraud:
        return jsonify({'success': False, 'message': 'Fraud detection model not loaded.'}), 500

    # Prepare claim data for fraud prediction function
    claim_data_for_fraud = {
        'claimAmount': claim_amount,
        'claimType': claim_type,
        # Default/placeholder values for other features if not directly from input
        'claimProbability': 0.0, # Will be set by predict_claims_submission
        'fraudLikelihoodScore': 0.0, # Will be set by predict_fraud
    }
    
    fraud_prediction_result = predict_fraud(
        claim_data_for_fraud, 
        MODELS_DIR, 
        preprocessor_fraud, 
        features_fraud, 
        categorical_features_fraud
    )
    is_fraud_flagged = fraud_prediction_result['is_fraud_flagged']
    fraud_likelihood = fraud_prediction_result['fraud_likelihood_score']

    # Prepare client data for claims submission probability
    # You need to fetch all necessary client features for this prediction.
    # This might require querying the Client model and aggregating claims/interactions.
    # For simplicity, here we'll use a subset of client_exists attributes.
    # In a real app, you'd load/prepare full client data for the prediction function.
    client_data_for_claims_pred = {
        'age': client_exists.age,
        'engagementScore': client_exists.engagement_score,
        'churnProbability': client_exists.churn_probability,
        'riskScore': client_exists.risk_score,
        'gender': client_exists.gender,
        'location': client_exists.location,
        'category': client_exists.category,
        'sentimentScore': client_exists.sentiment_score,
        'registrationDate': client_exists.registration_date,
        'total_interactions': len(client_exists.claims) # This is a placeholder; real total_interactions needs logs
    }
    # Fetch additional client-specific data needed for predict_claims_submission if not already in client_exists
    # Example: totalClaims, averageClaimAmount, recency_last_claim, recency_registration
    # The `predict_claims_submission` function handles some of this internally if `registrationDate` is passed.
    
    claims_prob_result = predict_claims_submission(
        client_data_for_claims_pred, 
        MODELS_DIR, 
        features_claims_forecast, 
        categorical_features_claims_forecast
    )
    claim_probability = claims_prob_result['claim_probability']

    new_claim = Claim(
        id=new_claim_id,
        client_id=client_id,
        claim_type=claim_type,
        claim_amount=claim_amount,
        submission_date=datetime.now(),
        claim_status='Pending',
        claim_probability=claim_probability,
        fraud_likelihood_score=fraud_likelihood,
        is_fraud_flagged=is_fraud_flagged,
        supporting_documents=supporting_documents_filenames
    )
    db.session.add(new_claim)
    db.session.commit()

    return jsonify({'success': True, 'data': new_claim.to_dict(), 'message': 'Claim submitted successfully!'})

# --- Churn Risk Routes ---
@app.route('/churn-risk', methods=['GET'])
# @jwt_required()
def get_churn_risk_clients():
    time.sleep(0.8)
    clients = Client.query.all()
    
    # Recalculate churn probability for all clients using the model
    updated_clients = []
    for client in clients:
        # Prepare client data for churn prediction
        client_data_for_churn = {
            'age': client.age,
            'engagementScore': client.engagement_score,
            'riskScore': client.risk_score,
            'gender': client.gender,
            'location': client.location,
            'category': client.category,
            'sentimentScore': client.sentiment_score,
            'registrationDate': client.registration_date,
            # Placeholder for other features that the model might expect:
            'totalClaims': len(client.claims), # From client relationship
            'averageClaimAmount': sum(c.claim_amount for c in client.claims) / len(client.claims) if client.claims else 0.0,
            'lastClaimDate': max([c.submission_date for c in client.claims]) if client.claims else None,
            # These values will be calculated within predict_churn
            'total_interactions': 0, # Assuming this comes from logs, not directly on Client model
            'last_interaction_date': None
        }

        # Use the predict_churn function
        if churn_model: # Ensure model is loaded
            churn_prediction_result = predict_churn(
                client_data_for_churn, 
                MODELS_DIR, 
                features_churn, 
                categorical_features_churn
            )
            client.churn_probability = churn_prediction_result['churn_probability']
        else:
            print("Churn model not loaded, using existing churn_probability.")
            # Fallback to existing if model not available
        
        updated_clients.append(client)
    
    db.session.commit() # Commit any updates to churn_probability

    # Sort by churn probability descending
    clients_sorted = sorted(updated_clients, key=lambda c: c.churn_probability, reverse=True)
    return jsonify({'success': True, 'data': [c.to_dict() for c in clients_sorted]})

@app.route('/churn-risk/actions/<client_id>', methods=['POST'])
# @jwt_required()
def take_retention_action(client_id):
    time.sleep(1.0)
    client = Client.query.get(client_id)
    if not client:
        return jsonify({'success': False, 'message': 'Client not found'}), 404

    action_data = request.get_json()
    action_type = action_data.get('actionType')
    notes = action_data.get('notes')
    scheduled_date = action_data.get('scheduledDate')

    print(f"Retention action recorded for client {client_id}: Type='{action_type}', Notes='{notes}', Scheduled='{scheduled_date}'")

    # Re-predict retention recommendations after an action
    if recommendation_model:
        recommendation_result = recommend_retention_actions(client_id, MODELS_DIR, N=3)
        client.retention_recommendation = ", ".join(recommendation_result.get('recommendations', []))
        db.session.commit()
    else:
        print("Recommendation model not loaded, cannot update retention recommendation.")

    return jsonify({'success': True, 'message': f'Retention action "{action_type}" taken for client {client_id}.'})

# --- Reports Routes ---
@app.route('/reports', methods=['GET'])
# @jwt_required()
def get_reports_list():
    time.sleep(0.8)
    reports = Report.query.order_by(Report.generated_at.desc()).all()
    return jsonify({'success': True, 'data': [r.to_dict() for r in reports]})

@app.route('/reports/generate', methods=['POST'])
# @jwt_required()
def generate_custom_report_endpoint():
    time.sleep(0.5)
    data = request.get_json()
    
    report_type = data.get('reportType')
    file_type = data.get('fileType', 'pdf')
    start_date_str = data.get('startDate')
    end_date_str = data.get('endDate')
    filters = data.get('filters', {})

    if not report_type:
        return jsonify({'success': False, 'message': 'Report type is required.'}), 400

    report_id = f"RPT-{uuid.uuid4().hex[:8].upper()}"
    generated_at = datetime.now()
    
    title = f"{report_type.replace('-', ' ').title()} Report ({datetime.now().strftime('%Y-%m-%d %H:%M')})"
    if start_date_str and end_date_str:
        title += f" from {start_date_str.split('T')[0]} to {end_date_str.split('T')[0]}"

    new_report = Report(
        id=report_id,
        title=title,
        type=report_type,
        generated_at=generated_at,
        status="generating",
        file_type=file_type,
        size_bytes=None
    )
    db.session.add(new_report)
    db.session.commit()

    def _simulate_report_generation(report_id_to_update, rpt_type, rpt_filters):
        time.sleep(random.uniform(3, 7))
        with app.app_context():
            report_in_db = Report.query.get(report_id_to_update)
            if report_in_db:
                # In a real scenario, you would fetch relevant data and apply ML models
                # based on rpt_type and rpt_filters here to generate actual report content.
                # For this example, we'll just simulate success/failure.
                if random.random() < 0.85:
                    report_in_db.status = 'ready'
                    report_in_db.size_bytes = random.randint(50 * 1024, 5 * 1024 * 1024)
                    print(f"Report {report_id_to_update} is ready.")
                else:
                    report_in_db.status = 'failed'
                    print(f"Report {report_id_to_update} failed.")
                db.session.commit()

    threading.Thread(target=_simulate_report_generation, args=(new_report.id, report_type, filters)).start()

    return jsonify({'success': True, 'data': new_report.to_dict(), 'message': 'Report generation initiated.'}), 202

@app.route('/reports/download/<report_id>', methods=['GET'])
# @jwt_required()
def download_report(report_id):
    time.sleep(0.5)
    report = Report.query.get(report_id)

    if not report:
        return jsonify({'success': False, 'message': 'Report not found.'}), 404
    if report.status != 'ready':
        return jsonify({'success': False, 'message': 'Report is not ready for download or has failed.'}), 409

    file_type = report.file_type
    content = f"This is a dummy {file_type.upper()} report for '{report.title}' generated on {report.generated_at}. Report ID: {report.id}\n\n"
    content += "This content simulates complex report data based on your filters."
    content += "\n\nFor actual data, implement real report generation logic with your ML models and data sources."

    mimetype_map = {
        'pdf': 'application/pdf',
        'csv': 'text/csv',
        'excel': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    }
    mimetype = mimetype_map.get(file_type, 'application/octet-stream')

    file_buffer = BytesIO(content.encode('utf-8'))
    file_buffer.seek(0)

    return send_file(
        file_buffer,
        mimetype=mimetype,
        as_attachment=True,
        download_name=f"{report.title.replace(' ', '_').replace(':', '')}.{file_type}"
    )

# --- Notifications Routes (Already good, minor adjustments for consistency) ---
@app.route('/notifications', methods=['GET'])
# @jwt_required()
def get_notifications():
    time.sleep(0.8)
    status_filter = request.args.get('status', 'all').lower()
    
    notifications_query = Notification.query

    if status_filter == 'unread':
        notifications_query = notifications_query.filter_by(read=False)
    
    notifications = notifications_query.order_by(Notification.date.desc()).all()
    
    return jsonify({'success': True, 'data': [n.to_dict() for n in notifications], "message": "Notifications fetched successfully."})

@app.route('/notifications/<notification_id>/read', methods=['PATCH'])
# @jwt_required()
def mark_notification_read(notification_id):
    time.sleep(0.3)
    notification = Notification.query.get(notification_id)
    if not notification:
        return jsonify({'success': False, 'message': 'Notification not found'}), 404
    
    notification.read = True
    db.session.commit()
    return jsonify({'success': True, 'message': 'Notification marked as read.', 'data': notification.to_dict()})

@app.route('/notifications/mark-all-read', methods=['PATCH'])
# @jwt_required()
def mark_all_notifications_read():
    time.sleep(0.5)
    updated_count = 0
    notifications_to_mark = Notification.query.filter_by(read=False).all()
    for notification in notifications_to_mark:
        notification.read = True
        updated_count += 1
    db.session.commit()
    return jsonify({'success': True, 'message': f'Marked {updated_count} notifications as read.', 'data': {'count': updated_count}})

@app.route('/notifications/<notification_id>', methods=['DELETE'])
# @jwt_required()
def delete_notification(notification_id):
    time.sleep(0.3)
    notification = Notification.query.get(notification_id)
    if not notification:
        return jsonify({'success': False, 'message': 'Notification not found'}), 404
    
    db.session.delete(notification)
    db.session.commit()
    return jsonify({'success': True, 'message': 'Notification dismissed successfully.'})


# --- Messages Routes ---
@app.route('/messages', methods=['GET'])
# @jwt_required()
def get_messages():
    time.sleep(0.8)
    messages = Message.query.order_by(Message.timestamp.asc()).all()
    return jsonify({'success': True, 'data': [m.to_dict() for m in messages]})

@app.route('/messages', methods=['POST'])
# @jwt_required()
def send_message():
    time.sleep(1.0)
    data = request.get_json()
    receiver_id = data.get('receiverId')
    content = data.get('content')

    if not receiver_id or not content:
        return jsonify({'success': False, 'message': 'Receiver ID and content are required.'}), 400

    sender_id = get_jwt_identity() if get_jwt_identity() else "user-001"

    new_message_id = str(uuid.uuid4())
    new_message = Message(
        id=new_message_id,
        sender_id=sender_id,
        receiver_id=receiver_id,
        content=content,
        timestamp=datetime.now(),
        read=False
    )
    db.session.add(new_message)
    db.session.commit()

    return jsonify({'success': True, 'data': new_message.to_dict(), 'message': 'Message sent successfully.'})

# --- Support Routes ---
@app.route('/support/tickets', methods=['POST'])
# @jwt_required()
def submit_support_ticket():
    time.sleep(1.5)
    
    subject = request.form.get('subject')
    description = request.form.get('description')
    priority = request.form.get('priority')

    if not subject or not description or not priority:
        return jsonify({'success': False, 'message': 'Subject, description, and priority are required.'}), 400

    attachments_info = []
    if 'attachments' in request.files:
        files = request.files.getlist('attachments')
        for file in files:
            if file.filename:
                allowed_extensions = {'png', 'jpg', 'jpeg', 'pdf', 'doc', 'docx'}
                if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    attachments_info.append({
                        'name': filename,
                        'size': os.path.getsize(filepath),
                        'type': file.mimetype
                    })
                else:
                    print(f"Skipping disallowed attachment: {filename}")

    new_ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
    new_ticket = SupportTicket(
        id=new_ticket_id,
        subject=subject,
        description=description,
        priority=priority,
        submitted_at=datetime.now(),
        attachments=attachments_info
    )
    db.session.add(new_ticket)
    db.session.commit()

    return jsonify({'success': True, 'data': new_ticket.to_dict(), 'message': 'Your support ticket has been submitted successfully!'}), 201

# --- Help Routes ---

@app.route('/help/categories', methods=['GET'])
# @jwt_required()
def get_help_categories():
    time.sleep(0.5)
    mock_categories = [
        {
            "id": "getting-started",
            "title": "Getting Started",
            "iconName": "Book",
            "description": "Learn the basics of navigating and utilizing the system.",
            "articles": [
                {"id": "art-gs-1", "name": "System Overview", "slug": "system-overview"},
                {"id": "art-gs-2", "name": "Navigation Guide", "slug": "navigation-guide"},
            ],
        },
        {
            "id": "claims-management",
            "title": "Claims Management",
            "iconName": "FileText",
            "description": "Everything you need to know about handling insurance claims efficiently.",
            "articles": [
                {"id": "art-cm-1", "name": "Creating a New Claim", "slug": "creating-a-new-claim"},
                {"id": "art-cm-2", "name": "Processing Claims Workflows", "slug": "processing-claims"},
            ],
        },
    ]
    return jsonify({'success': True, 'data': mock_categories, "message": "Help categories fetched."})

@app.route('/help/faqs', methods=['GET'])
# @jwt_required()
def get_help_faqs():
    time.sleep(0.5)
    search_query = request.args.get('search', '').lower()

    mock_faqs = [
        {
            "id": "faq-1",
            "question": "How do I reset my password?",
            "answer": "To reset your password, navigate to the **login page** and click on the 'Forgot Password' link. Enter your registered email address, and follow the instructions sent to your inbox to set a new password.",
        },
        {
            "id": "faq-2",
            "question": "What do the risk scores mean for clients?",
            "answer": "Risk scores provide an analytical assessment of a client's potential churn or fraud risk. Scores are generated based on a comprehensive evaluation of historical data, engagement patterns, and other behavioral metrics. A higher score indicates a higher predicted risk.",
        },
        {
            "id": "faq-3",
            "question": "How can I export reports from the system?",
            "answer": "You can export various reports from the **Reports** section. First, select the desired report type and specify the date range. Then, click the 'Export' button and choose your preferred format, such as **PDF, CSV, or Excel**.",
        },
    ]

    filtered_faqs = [
        faq for faq in mock_faqs
        if search_query in faq['question'].lower() or search_query in faq['answer'].lower()
    ]
    return jsonify({'success': True, 'data': filtered_faqs, "message": "FAQs fetched."})

@app.route('/help/articles/<slug>', methods=['GET'])
# @jwt_required()
def get_help_article(slug):
    time.sleep(0.5)
    article_content = db["help_articles"].get(slug)
    if article_content:
        return jsonify({'success': True, 'data': article_content, "message": "Article fetched."})
    return jsonify({'success': False, 'message': 'Article not found.'}), 404

# --- Machine Learning Endpoints ---

@app.route('/predict/churn', methods=['POST'])
# @jwt_required()
def predict_churn_endpoint(): # Renamed to avoid conflict with imported function
    if not churn_model:
        return jsonify({'success': False, 'message': 'Churn prediction model not loaded.'}), 500

    data = request.json
    client_id = data.get('clientId')

    if not client_id:
        return jsonify({'success': False, 'message': 'Client ID is required.'}), 400

    client = Client.query.get(client_id)
    if not client:
        return jsonify({'success': False, 'message': 'Client not found.'}), 404

    # Prepare client data for churn prediction function
    client_data_for_churn = {
        'age': client.age,
        'engagementScore': client.engagement_score,
        'riskScore': client.risk_score,
        'gender': client.gender,
        'location': client.location,
        'category': client.category,
        'sentimentScore': client.sentiment_score,
        'registrationDate': client.registration_date,
        'totalClaims': len(client.claims),
        'averageClaimAmount': sum(c.claim_amount for c in client.claims) / len(client.claims) if client.claims else 0.0,
        'lastClaimDate': max([c.submission_date for c in client.claims]) if client.claims else None,
        'total_interactions': 0, # Placeholder, will be updated based on real interaction logs
        'last_interaction_date': None # Placeholder, will be updated based on real interaction logs
    }
    # In a real system, you would query interaction logs to get `total_interactions` and `last_interaction_date`

    try:
        churn_prediction_result = predict_churn(
            client_data_for_churn, 
            MODELS_DIR, 
            features_churn, 
            categorical_features_churn
        )
        churn_probability = churn_prediction_result['churn_probability']
        churn_prediction_label = "Likely to Churn" if churn_probability > 50 else "Not Likely to Churn"

        client.churn_probability = round(churn_probability, 2)
        db.session.commit()

        return jsonify({
            'success': True,
            'clientId': client_id,
            'churnProbability': round(churn_probability, 2),
            'prediction': churn_prediction_label
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error during churn prediction: {str(e)}'}), 500

@app.route('/predict/claims_forecast', methods=['POST'])
# @jwt_required()
def predict_claims_forecast_endpoint(): # Renamed
    if not claims_forecast_model:
        return jsonify({'success': False, 'message': 'Claims forecasting model not loaded.'}), 500

    data = request.json
    client_id = data.get('clientId')
    forecast_period_months = data.get('periodMonths', 3)

    client = Client.query.get(client_id)
    if not client:
        return jsonify({'success': False, 'message': 'Client not found.'}), 404

    # Prepare client data for claims submission prediction
    client_data_for_claims_pred = {
        'age': client.age,
        'engagementScore': client.engagement_score,
        'churnProbability': client.churn_probability,
        'riskScore': client.risk_score,
        'gender': client.gender,
        'location': client.location,
        'category': client.category,
        'sentimentScore': client.sentiment_score,
        'registrationDate': client.registration_date,
        'total_interactions': 0 # Placeholder for actual total interactions from logs
    }

    try:
        claims_prob_result = predict_claims_submission(
            client_data_for_claims_pred, 
            MODELS_DIR, 
            features_claims_forecast, 
            categorical_features_claims_forecast
        )
        # The claims_forecast_model in generate_train_predict.py predicts has_filed_claim,
        # not a count. We'll map this to a dummy count for demonstration.
        # In a real app, claims_forecast_model would be a time series model predicting counts.
        
        # Based on claims_prob_result['claim_probability'] which is a percentage
        if claims_prob_result['claim_probability'] > 50:
            forecasted_claims = random.randint(1, 3) * (forecast_period_months / 3) # Scale roughly by period
        else:
            forecasted_claims = random.randint(0, 1) # Low probability, so fewer claims

        return jsonify({
            'success': True,
            'clientId': client_id,
            'forecastPeriodMonths': forecast_period_months,
            'forecastedClaimsCount': int(forecasted_claims) # Cast to int for consistency
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error during claims forecasting: {str(e)}'}), 500


@app.route('/predict/client_segmentation', methods=['POST'])
# @jwt_required()
def predict_client_segmentation_endpoint(): # Renamed
    global client_segmentation_model, scaler_segmentation, cluster_mapping

    if not client_segmentation_model or not scaler_segmentation or not cluster_mapping:
        return jsonify({'success': False, 'message': 'Client segmentation model or its components not loaded.'}), 500

    data = request.json
    client_ids = data.get('clientIds')

    if not client_ids or not isinstance(client_ids, list):
        return jsonify({'success': False, 'message': 'A list of client IDs is required.'}), 400

    segmentation_results = []
    for client_id in client_ids:
        client = Client.query.get(client_id)
        if not client:
            segmentation_results.append({'clientId': client_id, 'segment': 'Not Found', 'message': 'Client not found'})
            continue

        # Prepare client data for segmentation prediction
        client_data_for_segmentation = {
            'age': client.age,
            'engagementScore': client.engagement_score,
            'churnProbability': client.churn_probability,
            'riskScore': client.risk_score,
            'registrationDate': client.registration_date,
            'totalClaims': len(client.claims),
            'averageClaimAmount': sum(c.claim_amount for c in client.claims) / len(client.claims) if client.claims else 0.0,
            'lastClaimDate': max([c.submission_date for c in client.claims]) if client.claims else None,
            'total_interactions': 0, # Placeholder
            'last_interaction_date': None # Placeholder
        }

        try:
            segment_pred_result = predict_segmentation(
                client_data_for_segmentation, 
                MODELS_DIR, 
                MODELS_DIR, # For scaler_segmentation
                MODELS_DIR, # For cluster_mapping
                features_segmentation
            )
            segment_name = segment_pred_result['predicted_category']
            
            segmentation_results.append({'clientId': client_id, 'segment': segment_name})

            client.category = segment_name # Update client's category in DB
            db.session.commit()

        except Exception as e:
            segmentation_results.append({'clientId': client_id, 'segment': 'Error', 'message': str(e)})

    return jsonify({'success': True, 'data': segmentation_results})

@app.route('/predict/sentiment', methods=['POST'])
# @jwt_required()
def analyze_sentiment_endpoint(): # Renamed
    if not sentiment_analyzer: # This refers to the global VADER instance
        return jsonify({'success': False, 'message': 'Sentiment analyzer not loaded.'}), 500

    data = request.json
    text_to_analyze = data.get('text')

    if not text_to_analyze:
        return jsonify({'success': False, 'message': 'Text to analyze is required.'}), 400

    try:
        sentiment_result = predict_sentiment(text_to_analyze) # Use the imported function
        sentiment_label = sentiment_result['sentiment_label']
        scores = {'compound': sentiment_result['compound_score']} # Reconstruct scores if needed

        return jsonify({
            'success': True,
            'text': text_to_analyze,
            'sentiment': sentiment_label,
            'scores': scores
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error during sentiment analysis: {str(e)}'}), 500

@app.route('/predict/risk_score', methods=['POST'])
# @jwt_required()
def predict_risk_score_endpoint(): # Renamed
    if not risk_score_model:
        return jsonify({'success': False, 'message': 'Risk scoring model not loaded.'}), 500

    data = request.json
    client_id = data.get('clientId')
    claim_id = data.get('claimId') # Claim ID is optional, for richer context

    features_for_risk_pred = {}
    client_obj = None
    claim_obj = None

    if client_id:
        client_obj = Client.query.get(client_id)
        if not client_obj:
            return jsonify({'success': False, 'message': 'Client not found.'}), 404
    
    if claim_id:
        claim_obj = Claim.query.get(claim_id)
        if not claim_obj:
            return jsonify({'success': False, 'message': 'Claim not found.'}), 404
        if not client_obj and claim_obj.client: # If client_id not provided, derive from claim
            client_obj = claim_obj.client

    if not client_obj: # A client is always needed for risk score calculation
        return jsonify({'success': False, 'message': 'Client data is essential for risk scoring.'}), 400

    # Prepare client_data for predict_risk_score function
    # It expects comprehensive client data, including derived features or raw data to calculate them
    client_data_for_risk = {
        'age': client_obj.age,
        'engagementScore': client_obj.engagement_score,
        'churnProbability': client_obj.churn_probability,
        'riskScore': client_obj.risk_score, # Existing risk score
        'gender': client_obj.gender,
        'location': client_obj.location,
        'category': client_obj.category,
        'sentimentScore': client_obj.sentiment_score,
        'registrationDate': client_obj.registration_date,
        'totalClaims': len(client_obj.claims),
        'averageClaimAmount': sum(c.claim_amount for c in client_obj.claims) / len(client_obj.claims) if client_obj.claims else 0.0,
        'lastClaimDate': max([c.submission_date for c in client_obj.claims]) if client_obj.claims else None,
        'total_interactions': 0, # Placeholder for actual interactions data
        'last_interaction_date': None, # Placeholder for actual interactions data
        'feedbackText': "" # Placeholder for sentiment analysis if available (e.g., from recent messages)
    }

    if claim_obj:
        # Add claim-specific data that might influence risk score if model uses it
        client_data_for_risk['claimAmount'] = claim_obj.claim_amount
        client_data_for_risk['claimType'] = claim_obj.claim_type
        # Add other claim features if your risk model utilizes them directly (e.g., submission date related features)
        # For simplicity, if claim_obj is present, we might override client.risk_score based on this claim's fraud likelihood
        # This will be handled by the predict_risk_score function, which uses the internal ML models.

    try:
        risk_prediction_result = predict_risk_score(
            client_data_for_risk, 
            MODELS_DIR,
            features_churn, categorical_features_churn,
            features_segmentation,
            features_risk, categorical_features_risk
        )
        predicted_risk_score_value = risk_prediction_result['predicted_risk_score']
        retention_recommendation_text = risk_prediction_result['retention_recommendation']
        
        # Update client's risk score and retention recommendation in DB
        client_obj.risk_score = float(predicted_risk_score_value)
        client_obj.retention_recommendation = retention_recommendation_text # Update client's recommendation
        
        if claim_obj: # If a claim was part of the input, update its fraud score/flag too
            claim_obj.fraud_likelihood_score = float(predicted_risk_score_value) # Using risk score as proxy for fraud likelihood
            claim_obj.is_fraud_flagged = predicted_risk_score_value > 70 # Example threshold for flagging fraud
        
        db.session.commit()

        return jsonify({
            'success': True,
            'clientId': client_id,
            'claimId': claim_id,
            'riskScore': round(float(predicted_risk_score_value), 2),
            'retentionRecommendation': retention_recommendation_text
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error during risk scoring: {str(e)}'}), 500

@app.route('/predict/fraud_detection', methods=['POST'])
# @jwt_required()
def predict_fraud_detection_endpoint(): # Renamed
    global fraud_detection_model, preprocessor_fraud

    if not fraud_detection_model or not preprocessor_fraud:
        return jsonify({'success': False, 'message': 'Fraud detection model or preprocessor not loaded.'}), 500

    data = request.json
    claim_id = data.get('claimId')

    if not claim_id:
        return jsonify({'success': False, 'message': 'Claim ID is required.'}), 400

    claim = Claim.query.get(claim_id)
    if not claim:
        return jsonify({'success': False, 'message': 'Claim not found.'}), 404

    # Prepare claim data for fraud prediction function
    claim_data_for_fraud = {
        'claimAmount': claim.claim_amount,
        'claimType': claim.claim_type,
        'claimProbability': claim.claim_probability, # Assuming this is known or mocked
        'fraudLikelihoodScore': claim.fraud_likelihood_score # Assuming this is known or mocked
    }

    try:
        fraud_prediction_result = predict_fraud(
            claim_data_for_fraud, 
            MODELS_DIR, 
            preprocessor_fraud, 
            features_fraud, 
            categorical_features_fraud
        )
        is_fraud_flagged = fraud_prediction_result['is_fraud_flagged']
        fraud_likelihood_score = fraud_prediction_result['fraud_likelihood_score']
        
        claim.is_fraud_flagged = is_fraud_flagged
        claim.fraud_likelihood_score = fraud_likelihood_score
        db.session.commit()

        return jsonify({
            'success': True,
            'claimId': claim_id,
            'isFraudFlagged': is_fraud_flagged,
            'fraudLikelihoodScore': fraud_likelihood_score
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error during fraud detection: {str(e)}'}), 500

@app.route('/recommendations/retention/<client_id>', methods=['GET'])
# @jwt_required()
def get_retention_recommendations_endpoint(client_id): # Renamed
    global recommendation_model, client_id_to_encoded, encoded_to_client_id, \
           action_type_to_encoded, encoded_to_action_type, sparse_item_user

    if not recommendation_model or client_id_to_encoded is None or encoded_to_client_id is None \
       or action_type_to_encoded is None or encoded_to_action_type is None or sparse_item_user is None:
        return jsonify({'success': False, 'message': 'Recommendation model or its components not loaded. Please run generate_train_predict.py.'}), 500

    client = Client.query.get(client_id)
    if not client:
        return jsonify({'success': False, 'message': 'Client not found.'}), 404

    try:
        rec_actions_result = recommend_retention_actions(client_id, MODELS_DIR, N=3)
        recommended_actions_list = rec_actions_result.get('recommendations', [])
        
        # Update client's retention_recommendation field in DB
        client.retention_recommendation = ", ".join(recommended_actions_list)
        db.session.commit()

        return jsonify({
            'success': True,
            'clientId': client_id,
            'recommendedRetentionActions': recommended_actions_list
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error during retention recommendation: {str(e)}'}), 500


# --- Admin Routes ---
@app.route('/admin/users', methods=['GET'])
# @jwt_required()
def get_users():
    users = User.query.all()
    return jsonify({'success': True, 'data': [u.to_dict() for u in users]})

@app.route('/admin/users/new', methods=['POST'])
# @jwt_required()
def create_user():
    current_user_id = get_jwt_identity()
    current_user = db.session.get(User, current_user_id)
    if not current_user or current_user.role != 'admin':
        return jsonify({'success': False, 'message': 'Unauthorized: Only admins can create users.'}), 403

    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    role = data.get('role', 'user')

    if not name or not email or not password:
        return jsonify({'success': False, 'message': 'Name, email, and password are required.'}), 400
    if not isinstance(email, str) or '@' not in email:
        return jsonify({'success': False, 'message': 'Invalid email format.'}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({'success': False, 'message': 'User with this email already exists.'}), 409
    
    allowed_roles = ['user', 'manager', 'admin']
    if role not in allowed_roles:
        return jsonify({'success': False, 'message': f"Invalid role. Allowed roles are: {', '.join(allowed_roles)}."}), 400
    
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    
    new_user = User(
        name=name,
        email=email,
        password_hash=hashed_password,
        role=role,
        last_login=""
    )
    
    db.session.add(new_user)
    db.session.commit()

    return jsonify({'success': True, 'data': new_user.to_dict(), 'message': 'User created successfully!'}), 201

@app.route('/admin/users/<int:user_id>', methods=['GET', 'PUT'])
# @jwt_required()
def manage_user(user_id):
    current_user_id = get_jwt_identity()
    current_user = db.session.get(User, current_user_id)

    if not current_user or current_user.role != 'admin':
        if current_user and current_user.id == user_id:
            pass
        else:
            return jsonify({'success': False, 'message': 'Unauthorized: Access denied.'}), 403

    user_to_manage = db.session.get(User, user_id)
    if not user_to_manage:
        return jsonify({'success': False, 'message': 'User not found.'}), 404

    if request.method == 'GET':
        return jsonify({'success': True, 'data': user_to_manage.to_dict()})

    elif request.method == 'PUT':
        data = request.get_json()
        name = data.get('name')
        email = data.get('email')
        password = data.get('password')
        role = data.get('role')

        if name:
            user_to_manage.name = name
        if email:
            existing_user_with_email = User.query.filter_by(email=email).first()
            if existing_user_with_email and existing_user_with_email.id != user_id:
                return jsonify({'success': False, 'message': 'User with this email already exists.'}), 409
            user_to_manage.email = email
        if password:
            user_to_manage.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
        if role:
            allowed_roles = ['user', 'manager', 'admin']
            if role not in allowed_roles:
                return jsonify({'success': False, 'message': f"Invalid role. Allowed roles are: {', '.join(allowed_roles)}."}), 400
            user_to_manage.role = role
        
        db.session.commit()
        return jsonify({'success': True, 'data': user_to_manage.to_dict(), 'message': 'User updated successfully!'}), 200


@app.route('/admin/users/<int:user_id>/role', methods=['PUT'])
# @jwt_required()
def update_user_role(user_id):
    current_user_id = get_jwt_identity()
    current_user = db.session.get(User, current_user_id)
    if not current_user or current_user.role != 'admin':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403

    user = db.session.get(User, user_id)
    if not user:
        return jsonify({'success': False, 'message': 'User not found'}), 404

    data = request.get_json()
    new_role = data.get('role')
    if not new_role:
        return jsonify({'success': False, 'message': 'Role not provided'}), 400
    
    allowed_roles = ['user', 'manager', 'admin']
    if new_role not in allowed_roles:
        return jsonify({'success': False, 'message': f"Invalid role. Allowed roles are: {', '.join(allowed_roles)}."}), 400

    user.role = new_role
    db.session.commit()
    return jsonify({'success': True, 'data': user.to_dict(), 'message': f"User {user.email} role updated to {new_role}"})

@app.route('/admin/settings', methods=['PUT'])
# @jwt_required()
def update_settings():
    current_user_id = get_jwt_identity()
    current_user = db.session.get(User, current_user_id)
    if not current_user or current_user.role != 'admin':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403

    settings = request.get_json()
    print(f"Admin settings updated: {settings}")
    return jsonify({'success': True, 'data': settings, 'message': 'Settings updated successfully'})


# --- Main entry point ---
def create_default_accounts():
    default_users = [
        {
            'name': 'Master Admin',
            'email': 'admin@example.com',
            'role': 'admin',
            'password': 'AdminPass123'
        },
        {
            'name': 'Operations Manager',
            'email': 'manager@example.com',
            'role': 'manager',
            'password': 'ManagerPass123'
        },
        {
            'name': 'Default User',
            'email': 'user@example.com',
            'role': 'user',
            'password': 'UserPass123'
        }
    ]
    for user_data in default_users:
        existing_user = User.query.filter_by(email=user_data['email']).first()
        if not existing_user:
            user = User(
                name=user_data['name'],
                email=user_data['email'],
                role=user_data['role'],
                password_hash=bcrypt.generate_password_hash(user_data['password']).decode('utf-8'),
                last_login=""
            )
            db.session.add(user)
            print(f"Created default {user_data['role']} account: {user_data['email']} / {user_data['password']}")
    db.session.commit()

def populate_client_data():
    try:
        clients_csv_path = os.path.join(DATASETS_DIR, "clients.csv")
        claims_csv_path = os.path.join(DATASETS_DIR, "claims.csv")

        if not os.path.exists(clients_csv_path):
            print(f"'{clients_csv_path}' not found. Skipping client data population.")
            print("Please ensure you run 'generate_train_predict.py' to create the datasets.")
            return
        if not os.path.exists(claims_csv_path):
            print(f"'{claims_csv_path}' not found. Skipping claims data population.")
            print("Please ensure you run 'generate_train_predict.py' to create the datasets.")
            return

        df_clients = pd.read_csv(clients_csv_path)
        df_claims = pd.read_csv(claims_csv_path)
        
        # Ensure IDs are strings
        df_clients['id'] = df_clients['id'].astype(str)
        df_claims['clientId'] = df_claims['clientId'].astype(str)
        df_claims['id'] = df_claims['id'].astype(str)

        print("Populating database with generated data...")

        for _, row in df_clients.iterrows():
            if not Client.query.get(row['id']):
                client = Client(
                    id=row['id'],
                    full_name=row['fullName'],
                    email=row['email'],
                    phone_number=row['phoneNumber'],
                    address=row['address'],
                    age=row['age'],
                    gender=row['gender'],
                    location=row['location'],
                    registration_date=row['registrationDate'],
                    engagement_score=row['engagementScore'],
                    churn_probability=row['churnProbability'],
                    category=row['category'],
                    sentiment_score=row['sentimentScore'],
                    risk_score=row['riskScore'],
                    retention_recommendation=row['retentionRecommendation'],
                    fraud_flagged=row['fraudFlagged']
                )
                db.session.add(client)
        db.session.commit()
        print(f"Added {Client.query.count()} clients.")

        for _, row in df_claims.iterrows():
            if not Claim.query.get(row['id']):
                docs = row['supportingDocuments']
                if isinstance(docs, str) and docs.startswith('[') and docs.endswith(']'):
                    try:
                        docs = eval(docs)
                    except:
                        docs = []
                else:
                    docs = []

                claim = Claim(
                    id=row['id'],
                    client_id=row['clientId'],
                    claim_type=row['claimType'],
                    claim_amount=row['claimAmount'],
                    submission_date=row['submissionDate'],
                    claim_status=row['claimStatus'],
                    claim_probability=row['claimProbability'],
                    fraud_likelihood_score=row['fraudLikelihoodScore'],
                    is_fraud_flagged=row['isFraudFlagged'],
                    supporting_documents=docs
                )
                db.session.add(claim)
        db.session.commit()
        print(f"Added {Claim.query.count()} claims.")

        # Populate initial Notifications
        for notif_data in [
            {
                "id": str(uuid.uuid4()), "type": "fraud",
                "message": "High potential fraud alert detected for client **Tinashe Marufu** (ID: 8). Immediate review recommended.",
                "date": (datetime.now() - timedelta(hours=2)), "read": False, "link": "/clients/8"
            },
            {
                "id": str(uuid.uuid4()), "type": "churn",
                "message": "Client **Tapiwa Mutare** (ID: 2) shows an increased churn probability. Consider proactive outreach.",
                "date": (datetime.now() - timedelta(days=1)), "read": False, "link": "/clients/2"
            },
            {
                "id": str(uuid.uuid4()), "type": "claim",
                "message": "New claim **#CLM-2591** submitted by **Sekai Makoni** (ID: 5) for review.",
                "date": (datetime.now() - timedelta(days=2)), "read": False, "link": "/claims/CLM-2591"
            },
        ]:
            if not Notification.query.get(notif_data['id']):
                db.session.add(Notification(**notif_data))
        db.session.commit()
        print(f"Added {Notification.query.count()} notifications.")

        # Populate initial Reports
        for report_data in [
            {
                "id": "rpt-001", "title": "Q2 2025 Claims Summary", "type": "claims",
                "generated_at": (datetime.now() - timedelta(days=5)),
                "status": "ready", "file_type": "pdf", "size_bytes": 123456
            },
            {
                "id": "rpt-002", "title": "May 2025 Fraud Analysis", "type": "fraud",
                "generated_at": (datetime.now() - timedelta(days=7)),
                "status": "ready", "file_type": "excel", "size_bytes": 250000
            },
            {
                "id": "rpt-003", "title": "Client Churn Prediction (Upcoming Month)", "type": "churn",
                "generated_at": (datetime.now() - timedelta(minutes=2)),
                "status": "generating", "file_type": "csv", "size_bytes": None
            },
            {
                "id": "rpt-004", "title": "April 2025 Claims Details", "type": "claims",
                "generated_at": (datetime.now() - timedelta(days=35)),
                "status": "ready", "file_type": "csv", "size_bytes": 89000
            },
        ]:
            if not Report.query.get(report_data['id']):
                db.session.add(Report(**report_data))
        db.session.commit()
        print(f"Added {Report.query.count()} reports.")

        # Populate initial Messages
        # Adding mock messages for client-001 and client-002 conversations
        mock_messages_data = [
            {"id": str(uuid.uuid4()), "sender_id": "client-001", "receiver_id": "user-001", "content": "Hi, I have a question about my recent claim.", "timestamp": (datetime.now() - timedelta(minutes=60)), "read": True},
            {"id": str(uuid.uuid4()), "sender_id": "user-001", "receiver_id": "client-001", "content": "Certainly! How can I assist you today?", "timestamp": (datetime.now() - timedelta(minutes=58)), "read": True},
            {"id": str(uuid.uuid4()), "sender_id": "client-001", "receiver_id": "user-001", "content": "I submitted a claim for property damage last week, ID CLM-PRO-789. It seems to be stuck in \"Pending\" status.", "timestamp": (datetime.now() - timedelta(minutes=55)), "read": False},
            {"id": str(uuid.uuid4()), "sender_id": "user-001", "receiver_id": "client-001", "content": "Let me check on that for you. Please hold a moment.", "timestamp": (datetime.now() - timedelta(minutes=54)), "read": False},
            {"id": str(uuid.uuid4()), "sender_id": "client-002", "receiver_id": "user-001", "content": "Good morning! I need to update my contact information.", "timestamp": (datetime.now() - timedelta(days=1, minutes=30)), "read": True},
            {"id": str(uuid.uuid4()), "sender_id": "user-001", "receiver_id": "client-002", "content": "Of course! Could you please provide your new details?", "timestamp": (datetime.now() - timedelta(days=1, minutes=28)), "read": True},
        ]
        for msg_data in mock_messages_data:
            if not Message.query.get(msg_data['id']):
                db.session.add(Message(**msg_data))
        db.session.commit()
        print(f"Added {Message.query.count()} messages.")

    except FileNotFoundError as e:
        print(f"Required dataset file not found: {e}. Please ensure you run 'generate_train_predict.py' to create the datasets.")
    except Exception as e:
        print(f"Error populating database: {e}")


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        create_default_accounts()  # Create default user accounts
        populate_client_data() # Populate with generated data
        load_ml_models() # Load ML models after DB is ready
    app.run(debug=True)
