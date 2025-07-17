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
sentiment_analyzer = None


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
        user.last_login = datetime.utcnow().isoformat() + "Z"
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
@jwt_required()
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
@jwt_required()
def get_clients():
    time.sleep(0.8) # Simulate network delay
    clients = Client.query.all()
    return jsonify({'success': True, 'data': [c.to_dict() for c in clients]})

@app.route('/clients/<client_id>', methods=['GET'])
@jwt_required()
def get_client_details(client_id):
    time.sleep(0.5) # Simulate network delay
    client = Client.query.get(client_id)
    if not client:
        return jsonify({'success': False, 'message': 'Client not found'}), 404
    return jsonify({'success': True, 'data': client.to_dict()})

# --- Claims Routes ---
@app.route('/claims', methods=['GET'])
@jwt_required()
def get_claims():
    time.sleep(0.8) # Simulate network delay
    claims = Claim.query.all()
    return jsonify({'success': True, 'data': [c.to_dict() for c in claims]})

@app.route('/claims/<claim_id>', methods=['GET'])
@jwt_required()
def get_claim_details(claim_id):
    time.sleep(0.5) # Simulate network delay
    claim = Claim.query.get(claim_id)
    if not claim:
        return jsonify({'success': False, 'message': 'Claim not found'}), 404
    return jsonify({'success': True, 'data': claim.to_dict()})

@app.route('/claims/new', methods=['POST'])
@jwt_required()
def create_new_claim():
    time.sleep(1.5) # Simulate processing delay
    
    # Use request.form for multipart/form-data (when files are present)
    # Use request.get_json() if it's purely application/json
    # The frontend NewClaimPage.tsx sends form data (form-data), so use request.form.get
    client_id = request.form.get('clientId')
    claim_type = request.form.get('claimType')
    claim_amount = request.form.get('claimAmount') # Get as string, convert to float

    # Basic validation
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

    # Handle file uploads
    supporting_documents_filenames = []
    if 'supportingDocuments' in request.files: # Frontend uses 'supportingDocuments' as the field name
        files = request.files.getlist('supportingDocuments')
        for file in files:
            if file.filename:
                # Basic file type validation (optional, can be more robust)
                allowed_extensions = {'png', 'jpg', 'jpeg', 'pdf', 'doc', 'docx'}
                if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    supporting_documents_filenames.append(filename)
                else:
                    print(f"Skipping disallowed file type: {file.filename}") # Log for debugging

    # Generate a unique ID for the new claim
    new_claim_id = f"CLM-{uuid.uuid4().hex[:8].upper()}" # Example format

    # Mock prediction scores for a new claim
    # In a real app, you'd use your ML models here (fraud_detection_model, claims_probability_model)
    mock_fraud_likelihood = round(random.uniform(0.05, 0.95), 2) * 100 # 5-95%
    mock_is_fraud_flagged = mock_fraud_likelihood > 70 # Flag if > 70%
    mock_claim_probability = round(random.uniform(0.1, 0.9), 2) * 100 # Future claim likelihood

    new_claim = Claim(
        id=new_claim_id,
        client_id=client_id,
        claim_type=claim_type,
        claim_amount=claim_amount,
        submission_date=datetime.utcnow().isoformat() + "Z", # ISO format with Z for UTC
        claim_status='Pending', # New claims are usually pending
        claim_probability=mock_claim_probability,
        fraud_likelihood_score=mock_fraud_likelihood,
        is_fraud_flagged=mock_is_fraud_flagged,
        supporting_documents=supporting_documents_filenames
    )
    db.session.add(new_claim)
    db.session.commit()

    return jsonify({'success': True, 'data': new_claim.to_dict(), 'message': 'Claim submitted successfully!'})

# --- Churn Risk Routes ---
@app.route('/churn-risk', methods=['GET'])
@jwt_required()
def get_churn_risk_clients():
    time.sleep(0.8) # Simulate network delay
    # Fetch all clients, as the frontend `groupClientsByRisk` handles categorization
    clients = Client.query.all()
    # Sort by churn probability descending to show highest risk first
    clients_sorted = sorted(clients, key=lambda c: c.churn_probability, reverse=True)
    return jsonify({'success': True, 'data': [c.to_dict() for c in clients_sorted]})

@app.route('/churn-risk/actions/<client_id>', methods=['POST'])
@jwt_required()
def take_retention_action(client_id):
    time.sleep(1.0) # Simulate processing delay
    client = Client.query.get(client_id)
    if not client:
        return jsonify({'success': False, 'message': 'Client not found'}), 404

    action_data = request.get_json() # Frontend sends JSON
    action_type = action_data.get('actionType')
    notes = action_data.get('notes')
    scheduled_date = action_data.get('scheduledDate') # Add scheduledDate

    # Placeholder: In a real app, log retention actions to a dedicated table or update client state
    # For now, we'll just acknowledge and print
    print(f"Retention action recorded for client {client_id}: Type='{action_type}', Notes='{notes}', Scheduled='{scheduled_date}'")

    # Optionally update client's retention recommendation or engagement score here based on action
    # client.retention_recommendation = f"Actioned: {action_type}"
    # db.session.commit()

    return jsonify({'success': True, 'message': f'Retention action "{action_type}" taken for client {client_id}.'})

# --- Reports Routes ---
@app.route('/reports', methods=['GET'])
@jwt_required()
def get_reports_list(): # Renamed to avoid clash with function name in frontend
    time.sleep(0.8) # Simulate network delay
    # Returns all reports in the database
    reports = Report.query.order_by(Report.generated_at.desc()).all()
    return jsonify({'success': True, 'data': [r.to_dict() for r in reports]})

@app.route('/reports/generate', methods=['POST'])
@jwt_required()
def generate_custom_report_endpoint(): # Renamed to avoid clash with function name in frontend
    time.sleep(0.5) # Simulate initial processing delay
    data = request.get_json()
    
    report_type = data.get('reportType')
    file_type = data.get('fileType', 'pdf') # Default to pdf
    start_date_str = data.get('startDate')
    end_date_str = data.get('endDate')
    filters = data.get('filters', {})

    if not report_type:
        return jsonify({'success': False, 'message': 'Report type is required.'}), 400

    report_id = f"RPT-{uuid.uuid4().hex[:8].upper()}"
    generated_at = datetime.utcnow().isoformat() + "Z"
    
    # Derive title for the new report
    title = f"{report_type.replace('-', ' ').title()} Report ({datetime.utcnow().strftime('%Y-%m-%d %H:%M')})"
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

    # Simulate asynchronous generation in a separate thread
    def _simulate_report_generation(report_id_to_update):
        time.sleep(random.uniform(3, 7)) # Simulate variable generation time
        with app.app_context(): # Need app context for DB operations in thread
            report_in_db = Report.query.get(report_id_to_update)
            if report_in_db:
                if random.random() < 0.85: # 85% chance of success
                    report_in_db.status = 'ready'
                    report_in_db.size_bytes = random.randint(50 * 1024, 5 * 1024 * 1024) # 50KB to 5MB
                    print(f"Report {report_id_to_update} is ready.")
                else:
                    report_in_db.status = 'failed'
                    print(f"Report {report_id_to_update} failed.")
                db.session.commit()

    threading.Thread(target=_simulate_report_generation, args=(new_report.id,)).start()

    return jsonify({'success': True, 'data': new_report.to_dict(), 'message': 'Report generation initiated.'}), 202

@app.route('/reports/download/<report_id>', methods=['GET'])
@jwt_required()
def download_report(report_id):
    time.sleep(0.5) # Simulate download preparation delay
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
        'excel': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', # Standard MIME type for .xlsx
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
@jwt_required()
def get_notifications():
    time.sleep(0.8) # Simulate network delay
    status_filter = request.args.get('status', 'all').lower()
    
    # Filter by user if applicable in a multi-user system, e.g., current_user_id = get_jwt_identity()
    # For now, fetching all for simplicity

    notifications_query = Notification.query

    if status_filter == 'unread':
        notifications_query = notifications_query.filter_by(read=False)
    
    # Default sorting is by date descending, matching UI
    notifications = notifications_query.order_by(Notification.date.desc()).all()
    
    return jsonify({'success': True, 'data': [n.to_dict() for n in notifications], "message": "Notifications fetched successfully."})

@app.route('/notifications/<notification_id>/read', methods=['PATCH']) # Changed to PATCH
@jwt_required()
def mark_notification_read(notification_id):
    time.sleep(0.3)
    notification = Notification.query.get(notification_id)
    if not notification:
        return jsonify({'success': False, 'message': 'Notification not found'}), 404
    
    notification.read = True
    db.session.commit()
    return jsonify({'success': True, 'message': 'Notification marked as read.', 'data': notification.to_dict()})

@app.route('/notifications/mark-all-read', methods=['PATCH'])
@jwt_required()
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
@jwt_required()
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
@jwt_required()
def get_messages():
    time.sleep(0.8) # Simulate network delay
    # In a real app, you'd filter messages relevant to the current user.
    # For this mock, we return all messages, and frontend filters by selectedClientId.
    messages = Message.query.order_by(Message.timestamp.asc()).all() # Sort by timestamp ascending for chat order
    return jsonify({'success': True, 'data': [m.to_dict() for m in messages]})

@app.route('/messages', methods=['POST'])
@jwt_required()
def send_message():
    time.sleep(1.0) # Simulate sending delay
    data = request.get_json()
    receiver_id = data.get('receiverId')
    content = data.get('content')

    if not receiver_id or not content:
        return jsonify({'success': False, 'message': 'Receiver ID and content are required.'}), 400

    # For mock, senderId is fixed or from JWT identity
    sender_id = get_jwt_identity() if get_jwt_identity() else "user-001" # Default if no JWT for local testing

    new_message_id = str(uuid.uuid4())
    new_message = Message(
        id=new_message_id,
        sender_id=sender_id,
        receiver_id=receiver_id,
        content=content,
        timestamp=datetime.utcnow().isoformat() + "Z", # ISO format with Z for UTC
        read=False # Sent messages are typically unread by receiver until viewed
    )
    db.session.add(new_message)
    db.session.commit()

    return jsonify({'success': True, 'data': new_message.to_dict(), 'message': 'Message sent successfully.'})

# --- Support Routes ---
@app.route('/support/tickets', methods=['POST'])
@jwt_required()
def submit_support_ticket():
    time.sleep(1.5) # Simulate processing delay
    
    # Use request.form for multipart/form-data as frontend SupportPage.tsx sends files
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
                filename = secure_filename(file.filename)
                # Ensure valid file type, e.g., PNG, JPG, PDF, DOCX
                allowed_extensions = {'png', 'jpg', 'jpeg', 'pdf', 'doc', 'docx'}
                if '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions:
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    attachments_info.append({
                        'name': filename,
                        'size': os.path.getsize(filepath), # Get actual file size
                        'type': file.mimetype # Store mimetype
                    })
                else:
                    print(f"Skipping disallowed attachment: {filename}")

    new_ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
    new_ticket = SupportTicket(
        id=new_ticket_id,
        subject=subject,
        description=description,
        priority=priority,
        submitted_at=datetime.utcnow().isoformat() + "Z",
        attachments=attachments_info
    )
    db.session.add(new_ticket)
    db.session.commit()

    return jsonify({'success': True, 'data': new_ticket.to_dict(), 'message': 'Your support ticket has been submitted successfully!'}), 201

# --- Help Routes ---

@app.route('/help/categories', methods=['GET'])
@jwt_required()
def get_help_categories():
    time.sleep(0.5)
    # The helpCategories data is hardcoded in the frontend, so this endpoint is for future expansion
    # or if you move this data to the backend. For now, just return a mock structure.
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
@jwt_required()
def get_help_faqs():
    time.sleep(0.5)
    search_query = request.args.get('search', '').lower()

    # The FAQ data is hardcoded in the frontend, but if it were dynamic, this would filter it.
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
@jwt_required()
def get_help_article(slug):
    time.sleep(0.5)
    # The actual content for articles is hardcoded in frontend
    # If it were dynamic, db["help_articles"] would hold full content
    article_content = db["help_articles"].get(slug)
    if article_content:
        return jsonify({'success': True, 'data': article_content, "message": "Article fetched."})
    return jsonify({'success': False, 'message': 'Article not found.'}), 404

# --- Machine Learning Endpoints ---

@app.route('/predict/churn', methods=['POST'])
@jwt_required()
def predict_churn():
    if not churn_model:
        return jsonify({'success': False, 'message': 'Churn prediction model not loaded.'}), 500

    data = request.json
    client_id = data.get('clientId')

    if not client_id:
        return jsonify({'success': False, 'message': 'Client ID is required.'}), 400

    client = Client.query.get(client_id)
    if not client:
        return jsonify({'success': False, 'message': 'Client not found.'}), 404

    # Placeholder for collecting client's dynamic data for prediction
    # This should mirror the feature engineering in generate_train_predict.py
    client_data_for_prediction = {
        'age': client.age,
        'engagementScore': client.engagement_score,
        'totalClaims': len(client.claims),
        'averageClaimAmount': sum(c.claim_amount for c in client.claims) / len(client.claims) if client.claims else 0.0,
        'riskScore': client.risk_score,
        'recency_registration': (datetime.utcnow() - datetime.strptime(client.registration_date.split('T')[0], "%Y-%m-%d")).days if client.registration_date else -1,
        'gender_Male': 1 if client.gender == 'Male' else 0,
        'gender_Female': 1 if client.gender == 'Female' else 0,
        'location_New York, NY': 1 if client.location == 'New York, NY' else 0,
        'location_Chicago, IL': 1 if client.location == 'Chicago, IL' else 0,
        'location_Los Angeles, CA': 1 if client.location == 'Los Angeles, CA' else 0,
        'category_Loyal': 1 if client.category == 'Loyal' else 0,
        'category_At-Risk': 1 if client.category == 'At-Risk' else 0,
        'sentimentScore_Positive': 1 if client.sentiment_score == 'Positive' else 0,
        'sentimentScore_Negative': 1 if client.sentiment_score == 'Negative' else 0,
    }

    # Create a DataFrame from the single client's data
    df_predict = pd.DataFrame([client_data_for_prediction])
    
    # Ensure column order and types match training data - critical for model inference
    # This example assumes a simplified set of features or a preprocessor handles the full complexity.
    # In a real application, you would ensure `df_predict` has all columns from training,
    # and fill missing ones with appropriate defaults/means used during training.
    
    try:
        churn_probability = churn_model.predict_proba(df_predict)[:, 1][0] * 100
        churn_prediction_label = "Likely to Churn" if churn_probability > 50 else "Not Likely to Churn"

        # Update client's churn probability in DB
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
@jwt_required()
def predict_claims_forecast():
    if not claims_forecast_model:
        return jsonify({'success': False, 'message': 'Claims forecasting model not loaded.'}), 500

    data = request.json
    forecast_period_months = data.get('periodMonths', 3)

    # Placeholder for model input preparation.
    # In a real scenario, you'd feed historical data or relevant features to the model.
    # For a demo, return a dummy forecast.
    try:
        # Example dummy forecast
        forecasted_claims = random.randint(50, 500) * (forecast_period_months / 3) # Scale roughly by period

        return jsonify({
            'success': True,
            'forecastPeriodMonths': forecast_period_months,
            'forecastedClaimsCount': int(forecasted_claims)
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error during claims forecasting: {str(e)}'}), 500


@app.route('/predict/client_segmentation', methods=['POST'])
@jwt_required()
def predict_client_segmentation():
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

        # Prepare features for segmentation model. This needs to EXACTLY match what was used for training.
        # Ensure numerical features are handled (scaled) and categorical features are encoded.
        # This is a critical point: ensure feature names and types match the training data.
        client_data_for_segmentation = {
            'age': client.age,
            'engagementScore': client.engagement_score,
            'totalClaims': len(client.claims), # Use relationship
            'averageClaimAmount': sum(c.claim_amount for c in client.claims) / len(client.claims) if client.claims else 0.0,
            'riskScore': client.risk_score,
            'gender_Male': 1 if client.gender == 'Male' else 0,
            'gender_Female': 1 if client.gender == 'Female' else 0,
            'location_New York, NY': 1 if client.location == 'New York, NY' else 0,
            'location_Chicago, IL': 1 if client.location == 'Chicago, IL' else 0,
            'location_Los Angeles, CA': 1 if client.location == 'Los Angeles, CA' else 0,
            'location_San Francisco, CA': 1 if client.location == 'San Francisco, CA' else 0,
            'location_Seattle, WA': 1 if client.location == 'Seattle, WA' else 0,
            'location_Miami, FL': 1 if client.location == 'Miami, FL' else 0,
            'location_Boston, MA': 1 if client.location == 'Boston, MA' else 0,
            'location_Dallas, TX': 1 if client.location == 'Dallas, TX' else 0,
            'category_Loyal': 1 if client.category == 'Loyal' else 0,
            'category_At-Risk': 1 if client.category == 'At-Risk' else 0,
            'category_High-Value': 1 if client.category == 'High-Value' else 0,
            'category_Low-Engagement': 1 if client.category == 'Low-Engagement' else 0,
            'category_New': 1 if client.category == 'New' else 0,
            'registrationYear': datetime.strptime(client.registration_date.split('T')[0], "%Y-%m-%d").year if client.registration_date else datetime.utcnow().year,
        }
        
        # Create a DataFrame with a single row for prediction
        df_predict_raw = pd.DataFrame([client_data_for_segmentation])

        # Identify numerical features that need scaling as per your training setup
        numerical_features_for_scaling = [
            'age', 'engagementScore', 'totalClaims', 'averageClaimAmount', 'riskScore', 'registrationYear'
        ]
        
        # Filter df_predict_raw to only include features used for scaling, then scale
        df_scaled_part = pd.DataFrame(scaler_segmentation.transform(df_predict_raw[numerical_features_for_scaling]), 
                                       columns=numerical_features_for_scaling)
        
        # Combine scaled numerical features with original (or encoded) categorical features
        # This requires careful handling if your preprocessor for segmentation handled both.
        # For simplicity, assuming direct one-hot encoded features are ready.
        
        # Drop raw numerical features from df_predict_raw and add scaled ones
        df_predict_final = df_predict_raw.drop(columns=numerical_features_for_scaling)
        df_predict_final = pd.concat([df_predict_final, df_scaled_part], axis=1)

        try:
            cluster_id = client_segmentation_model.predict(df_predict_final)[0]
            segment_name = cluster_mapping.get(cluster_id, f"Segment {cluster_id}")
            
            segmentation_results.append({'clientId': client_id, 'segment': segment_name})

            client.category = segment_name
            db.session.commit()

        except Exception as e:
            segmentation_results.append({'clientId': client_id, 'segment': 'Error', 'message': str(e)})

    return jsonify({'success': True, 'data': segmentation_results})

@app.route('/predict/sentiment', methods=['POST'])
@jwt_required()
def analyze_sentiment():
    if not sentiment_analyzer:
        return jsonify({'success': False, 'message': 'Sentiment analyzer not loaded.'}), 500

    data = request.json
    text_to_analyze = data.get('text')

    if not text_to_analyze:
        return jsonify({'success': False, 'message': 'Text to analyze is required.'}), 400

    try:
        scores = sentiment_analyzer.polarity_scores(text_to_analyze)
        compound_score = scores['compound']

        sentiment_label = "Neutral"
        if compound_score >= 0.05:
            sentiment_label = "Positive"
        elif compound_score <= -0.05:
            sentiment_label = "Negative"

        return jsonify({
            'success': True,
            'text': text_to_analyze,
            'sentiment': sentiment_label,
            'scores': scores
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error during sentiment analysis: {str(e)}'}), 500

@app.route('/predict/risk_score', methods=['POST'])
@jwt_required()
def predict_risk_score():
    if not risk_score_model:
        return jsonify({'success': False, 'message': 'Risk scoring model not loaded.'}), 500

    data = request.json
    client_id = data.get('clientId')
    claim_id = data.get('claimId')

    if not client_id and not claim_id:
        return jsonify({'success': False, 'message': 'Either client ID or claim ID is required.'}), 400

    features_for_risk = {}
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
        # If claim is provided, prioritize its client or use the provided client_id
        if not client_obj and claim_obj.client:
            client_obj = claim_obj.client

    if not client_obj: # A client is needed for many risk features
        return jsonify({'success': False, 'message': 'Client data is essential for risk scoring.'}), 400

    # Assemble features from client and/or claim
    features_for_risk = {
        'age': client_obj.age,
        'engagementScore': client_obj.engagement_score,
        'churnProbability': client_obj.churn_probability,
        'totalClaims': len(client_obj.claims),
        'averageClaimAmount': sum(c.claim_amount for c in client_obj.claims) / len(client_obj.claims) if client_obj.claims else 0.0
    }

    if claim_obj:
        features_for_risk.update({
            'claimAmount': claim_obj.claim_amount,
            # Convert submission_date to a numerical feature (e.g., days since epoch or specific date)
            'submissionDaysSinceEpoch': datetime.strptime(claim_obj.submission_date.split('T')[0], "%Y-%m-%d").timestamp() / (60*60*24) if claim_obj.submission_date else 0,
            'num_supporting_documents': len(claim_obj.supporting_documents) if claim_obj.supporting_documents else 0
        })
        # Add one-hot encoded claim_type features if model requires
        # This needs to match the encoding performed during training.
        # Example for common claim types:
        claim_types = ['Personal Injury', 'Corporate Dispute', 'Family Law', 'Property Damage', 'Insurance Claim', 'Medical Malpractice', 'Employment Law', 'Other']
        for c_type in claim_types:
            features_for_risk[f'claimType_{c_type.replace(" ", "")}'] = 1 if claim_obj.claim_type == c_type else 0

    # Create a DataFrame from the features
    df_predict = pd.DataFrame([features_for_risk])
    
    # IMPORTANT: Ensure the columns in df_predict exactly match the order and names of features
    # that your risk_score_model was trained on.
    
    try:
        risk_score = risk_score_model.predict(df_predict)[0]
        
        # Update client's risk score (and potentially claim's fraud_likelihood_score)
        client_obj.risk_score = float(risk_score)
        if claim_obj:
            # Assuming claim's fraud_likelihood_score is derived from or related to the risk score
            claim_obj.fraud_likelihood_score = float(risk_score) * 100 # Convert to percentage 0-100
            claim_obj.is_fraud_flagged = (float(risk_score) * 100) > 70 # Example threshold
        db.session.commit()

        return jsonify({
            'success': True,
            'clientId': client_id,
            'claimId': claim_id,
            'riskScore': round(float(risk_score), 2)
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error during risk scoring: {str(e)}'}), 500

@app.route('/predict/fraud_detection', methods=['POST'])
@jwt_required()
def predict_fraud_detection():
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

    # Prepare features for fraud detection model. This needs to EXACTLY match the feature engineering
    # for the IsolationForest model in generate_train_predict.py.
    # It seems your fraud model might expect numerical features, and preprocessor_fraud handles the transformation.

    raw_features_for_fraud_detection = {
        'claimAmount': claim.claim_amount,
        'claimType': claim.claim_type,
        'submissionTimestamp': datetime.strptime(claim.submission_date.split('T')[0], "%Y-%m-%d").timestamp(),
        'client_engagement_score': claim.client.engagement_score if claim.client else 0,
        'client_age': claim.client.age if claim.client else 0,
        'num_supporting_documents': len(claim.supporting_documents) if claim.supporting_documents else 0,
        'client_risk_score': claim.client.risk_score if claim.client else 0
    }
    
    df_raw = pd.DataFrame([raw_features_for_fraud_detection])

    try:
        df_processed = preprocessor_fraud.transform(df_raw)
        
        fraud_prediction_raw = fraud_detection_model.predict(df_processed)[0]
        is_fraud = bool(fraud_prediction_raw == -1) # -1 indicates anomaly (potential fraud)
        
        claim.is_fraud_flagged = is_fraud
        db.session.commit()

        return jsonify({
            'success': True,
            'claimId': claim_id,
            'isFraudFlagged': is_fraud,
            'predictionRaw': int(fraud_prediction_raw)
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error during fraud detection: {str(e)}'}), 500

@app.route('/recommendations/retention/<client_id>', methods=['GET'])
@jwt_required()
def get_retention_recommendations(client_id):
    global recommendation_model, client_id_to_encoded, encoded_to_client_id, \
           action_type_to_encoded, encoded_to_action_type, sparse_item_user

    if not recommendation_model or client_id_to_encoded is None or encoded_to_client_id is None \
       or action_type_to_encoded is None or encoded_to_action_type is None or sparse_item_user is None:
        return jsonify({'success': False, 'message': 'Recommendation model or its components not loaded. Please run generate_train_predict.py.'}), 500

    client = Client.query.get(client_id)
    if not client:
        return jsonify({'success': False, 'message': 'Client not found.'}), 404

    if client_id not in client_id_to_encoded:
        # Fallback for clients not in the training data of the CF model
        return jsonify({'success': True, 'clientId': client_id, 'recommendedRetentionActions': ["Engage via email", "Offer loyalty bonus"], 'message': 'Client not in CF model data, providing generic recommendations.'})

    try:
        encoded_client_id = client_id_to_encoded[client_id]
        
        # This part assumes `recommendation_model` (LightFM) and `sparse_item_user` are correctly set up
        # to yield item recommendations based on user.
        # This is a common pattern for LightFM's `predict_top_n` or similar.
        
        # You might need to adjust based on the actual LightFM recommendation method signature:
        # e.g., model.predict(user_ids=np.array([encoded_client_id]), item_ids=np.arange(sparse_item_user.shape[0]))
        # Then sort by scores and get top N.
        
        # For a LightFM model, `predict` usually takes user_ids and item_ids
        # And `recommend` might not be a direct method without pre-calculating features.
        
        # Let's simulate for LightFM if it were designed like this:
        # user_features = np.array([encoded_client_id])
        # all_item_ids = np.arange(sparse_item_user.shape[0]) # All possible action types
        # predictions = recommendation_model.predict(user_ids=user_features, item_ids=all_item_ids)
        # top_n_indices = predictions.argsort()[-n_recommendations:][::-1]
        # recommended_encoded_actions = [all_item_ids[i] for i in top_n_indices]

        n_recommendations = 3
        
        # Dummy recommendation logic if direct model call is complex for demo:
        all_possible_actions = list(action_type_to_encoded.keys())
        if len(all_possible_actions) > n_recommendations:
            recommended_actions_list = random.sample(all_possible_actions, n_recommendations)
        else:
            recommended_actions_list = all_possible_actions
        
        # Update client's retention_recommendation
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
@jwt_required()
def get_users():
    users = User.query.all()
    return jsonify({'success': True, 'data': [u.to_dict() for u in users]})

@app.route('/admin/users/new', methods=['POST'])
@jwt_required()
def create_user():
    current_user_id = get_jwt_identity()
    current_user = User.query.get(current_user_id)
    # Ensure only admin can create new users
    if not current_user or current_user.role != 'admin':
        return jsonify({'success': False, 'message': 'Unauthorized: Only admins can create users.'}), 403

    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    role = data.get('role', 'user') # Default role to 'user' if not provided

    # Basic validation
    if not name or not email or not password:
        return jsonify({'success': False, 'message': 'Name, email, and password are required.'}), 400
    if not isinstance(email, str) or '@' not in email:
        return jsonify({'success': False, 'message': 'Invalid email format.'}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({'success': False, 'message': 'User with this email already exists.'}), 409
    
    # Validate role
    allowed_roles = ['user', 'manager', 'admin']
    if role not in allowed_roles:
        return jsonify({'success': False, 'message': f"Invalid role. Allowed roles are: {', '.join(allowed_roles)}."}), 400
    
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    
    new_user = User(
        name=name,
        email=email,
        password_hash=hashed_password,
        role=role,
        last_login="" # Set initial last login to empty string
    )
    
    db.session.add(new_user)
    db.session.commit()

    return jsonify({'success': True, 'data': new_user.to_dict(), 'message': 'User created successfully!'}), 201

# Consolidated route for getting and editing user details
@app.route('/admin/users/<int:user_id>', methods=['GET', 'PUT'])
@jwt_required()
def manage_user(user_id):
    current_user_id = get_jwt_identity()
    current_user = db.session.get(User, current_user_id) # Use db.session.get for primary key lookup

    if not current_user or current_user.role != 'admin':
        # Allow user to view/edit their own profile if not admin
        if current_user and current_user.id == user_id:
            pass # User can access their own data
        else:
            return jsonify({'success': False, 'message': 'Unauthorized: Access denied.'}), 403

    user_to_manage = db.session.get(User, user_id) # Use db.session.get
    if not user_to_manage:
        return jsonify({'success': False, 'message': 'User not found.'}), 404

    if request.method == 'GET':
        # Handle GET request: Return user details
        return jsonify({'success': True, 'data': user_to_manage.to_dict()})

    elif request.method == 'PUT':
        # Handle PUT request: Update user details
        data = request.get_json()
        name = data.get('name')
        email = data.get('email')
        password = data.get('password')
        role = data.get('role')

        # Update fields if provided
        if name:
            user_to_manage.name = name
        if email:
            # Check if new email already exists and is not the current user being edited
            existing_user_with_email = User.query.filter_by(email=email).first()
            if existing_user_with_email and existing_user_with_email.id != user_id:
                return jsonify({'success': False, 'message': 'User with this email already exists.'}), 409
            user_to_manage.email = email
        if password: # If password is provided, hash it
            user_to_manage.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
        if role:
            allowed_roles = ['user', 'manager', 'admin']
            if role not in allowed_roles:
                return jsonify({'success': False, 'message': f"Invalid role. Allowed roles are: {', '.join(allowed_roles)}."}), 400
            user_to_manage.role = role
        
        db.session.commit()
        return jsonify({'success': True, 'data': user_to_manage.to_dict(), 'message': 'User updated successfully!'}), 200


@app.route('/admin/users/<int:user_id>/role', methods=['PUT']) # Specific route for role update (can be consolidated into manage_user if preferred)
@jwt_required()
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
@jwt_required()
def update_settings():
    current_user_id = get_jwt_identity()
    current_user = db.session.get(User, current_user_id)
    if not current_user or current_user.role != 'admin':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403

    settings = request.get_json()
    # Placeholder: settings would be saved to database or config file
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
                "date": (datetime.now() - timedelta(hours=2)).isoformat() + "Z", "read": False, "link": "/clients/8"
            },
            {
                "id": str(uuid.uuid4()), "type": "churn",
                "message": "Client **Tapiwa Mutare** (ID: 2) shows an increased churn probability. Consider proactive outreach.",
                "date": (datetime.now() - timedelta(days=1)).isoformat() + "Z", "read": False, "link": "/clients/2"
            },
            {
                "id": str(uuid.uuid4()), "type": "claim",
                "message": "New claim **#CLM-2591** submitted by **Sekai Makoni** (ID: 5) for review.",
                "date": (datetime.now() - timedelta(days=2)).isoformat() + "Z", "read": False, "link": "/claims/CLM-2591"
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
                "generated_at": (datetime.now() - timedelta(days=5)).isoformat() + "Z",
                "status": "ready", "file_type": "pdf", "size_bytes": 123456
            },
            {
                "id": "rpt-002", "title": "May 2025 Fraud Analysis", "type": "fraud",
                "generated_at": (datetime.now() - timedelta(days=7)).isoformat() + "Z",
                "status": "ready", "file_type": "excel", "size_bytes": 250000
            },
            {
                "id": "rpt-003", "title": "Client Churn Prediction (Upcoming Month)", "type": "churn",
                "generated_at": (datetime.now() - timedelta(minutes=2)).isoformat() + "Z",
                "status": "generating", "file_type": "csv", "size_bytes": None
            },
            {
                "id": "rpt-004", "title": "April 2025 Claims Details", "type": "claims",
                "generated_at": (datetime.now() - timedelta(days=35)).isoformat() + "Z",
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
            {"id": str(uuid.uuid4()), "sender_id": "client-001", "receiver_id": "user-001", "content": "Hi, I have a question about my recent claim.", "timestamp": (datetime.now() - timedelta(minutes=60)).isoformat() + "Z", "read": True},
            {"id": str(uuid.uuid4()), "sender_id": "user-001", "receiver_id": "client-001", "content": "Certainly! How can I assist you today?", "timestamp": (datetime.now() - timedelta(minutes=58)).isoformat() + "Z", "read": True},
            {"id": str(uuid.uuid4()), "sender_id": "client-001", "receiver_id": "user-001", "content": "I submitted a claim for property damage last week, ID CLM-PRO-789. It seems to be stuck in \"Pending\" status.", "timestamp": (datetime.now() - timedelta(minutes=55)).isoformat() + "Z", "read": False},
            {"id": str(uuid.uuid4()), "sender_id": "user-001", "receiver_id": "client-001", "content": "Let me check on that for you. Please hold a moment.", "timestamp": (datetime.now() - timedelta(minutes=54)).isoformat() + "Z", "read": False},
            {"id": str(uuid.uuid4()), "sender_id": "client-002", "receiver_id": "user-001", "content": "Good morning! I need to update my contact information.", "timestamp": (datetime.now() - timedelta(days=1, minutes=30)).isoformat() + "Z", "read": True},
            {"id": str(uuid.uuid4()), "sender_id": "user-001", "receiver_id": "client-002", "content": "Of course! Could you please provide your new details?", "timestamp": (datetime.now() - timedelta(days=1, minutes=28)).isoformat() + "Z", "read": True},
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
