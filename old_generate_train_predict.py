import json
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, silhouette_score
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares
import os
import joblib

# --- Download NLTK VADER lexicon if not already downloaded ---
nltk.download('vader_lexicon', quiet=True) # Use quiet=True to suppress repeated messages

# --- Create directories if they don't exist ---
DATASETS_DIR = 'datasets'
MODELS_DIR = 'models'
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Helper Functions and Data for Zimbabwean Context ---

def generate_zim_phone_number():
    """Generates a realistic Zimbabwean mobile phone number."""
    prefixes = ['71', '77', '78', '73', '74'] # Common Zimbabwean mobile prefixes
    return f"+263 {random.choice(prefixes)}{random.randint(0, 9999999):07d}"

def generate_zim_address(city):
    """Generates a simple Zimbabwean-like street address."""
    street_names = [
        "Samora Machel Ave", "Robert Mugabe Rd", "Nelson Mandela Ave",
        "Leopold Takawira St", "Jason Moyo Ave", "Union Ave",
        "Borrowdale Rd", "Enterprise Rd", "Simon Mazorodze Rd",
        "Third St", "Fifth St", "Fife Ave", "George Silundika Ave"
    ]
    return f"{random.randint(1, 200)} {random.choice(street_names)}, {city}"

zim_first_names_male = ["Tendai", "Tapiwa", "Takudzwa", "Kudakwashe", "Tafadzwa", "Simba", "Munashe", "Anesu", "Tawanda", "Blessing", "Admire", "Pride", "Tanaka", "Farai", "Tapiwa", "Tinashe", "Innocent", "Prosper", "Liberty", "Walter"]
zim_first_names_female = ["Rudo", "Nyarai", "Chido", "Tatenda", "Fadzai", "Rufaro", "Tariro", "Tendai", "Lisa", "Memory", "Faith", "Shamiso", "Sekai", "Nompilo", "Chiedza", "Nomusa", "Precious", "Mercy", "Tsitsi", "Brenda"]
zim_last_names = [
    "Ncube", "Moyo", "Sibanda", "Dube", "Ndlovu", "Nkomo", "Mhlanga", "Chikwanda", "Musara",
    "Gumbo", "Makoni", "Mutare", "Chirwa", "Moyo", "Sibanda", "Nkomo", "Siziba", "Mabhena",
    "Huni", "Mashingaidze", "Nyathi", "Zwane", "Dlamini", "Khumalo", "Mhlanga", "Shumba",
    "Mathe", "Mavhunga", "Chirwa", "Mandaza", "Zinyama", "Marufu", "Mudenda", "Chigwedere",
    "Mupanduki", "Muchinguri", "Gatawa", "Masuku", "Mguni", "Muchena", "Chitima", "Chigudu"
]

zim_cities = ["Harare", "Bulawayo", "Mutare", "Gweru", "Kwekwe", "Masvingo", "Chinhoyi", "Kadoma", "Victoria Falls", "Bindura", "Gwanda", "Rusape"]

claim_types = ["Personal Injury", "Family Law", "Property Damage", "Motor Vehicle Accident", "Employment Dispute", "Medical Malpractice", "Commercial Dispute", "Debt Collection"]
claim_statuses = ["Pending", "Approved", "Rejected"]
interaction_types = ["Login", "Viewed Dashboard", "Opened Email", "Submitted Form", "Contacted Support", "Accessed Help Article", "Updated Profile", "Viewed Policy"]
retention_action_types = ["Offer Discount", "Schedule Call", "Send Personalized Email", "Provide Educational Content", "Send Follow-up Survey", "Invite to Webinar"]
retention_outcomes = ["Stayed", "Churned"]
sentiment_options = ["Positive", "Neutral", "Negative"]
client_categories = ["Loyal", "At-Risk", "High-Value", "Low-Engagement"]
retention_recommendations = [
    "Offer Discount", "Schedule Call", "Send Personalized Email",
    "Provide Educational Content", "Monitor Activity", "Proactive Check-in",
    "Address Specific Feedback", "VIP Treatment", "No Action Required"
]

# --- Data Generation Function ---
def generate_data(num_clients=450, max_claims_per_client=50, max_interactions_per_client=100):
    clients_data = []
    claims_data = []
    interaction_logs_data = []
    feedback_data = []
    retention_actions_data = []
    fraud_labels_data = []

    start_date_clients = datetime(2023, 5, 31) - timedelta(days=730) # 2 years ago from current time
    start_date_claims_interactions = datetime(2024, 5, 31) - timedelta(days=365) # 1 year ago from current time

    # Generate Clients
    client_ids = []
    for i in range(1, num_clients + 1):
        client_id = f"C{i:03d}"
        client_ids.append(client_id)

        gender = random.choice(["Male", "Female"])
        first_name = random.choice(zim_first_names_male) if gender == "Male" else random.choice(zim_first_names_female)
        last_name = random.choice(zim_last_names)
        full_name = f"{first_name} {last_name}"
        location = random.choice(zim_cities)

        age = random.randint(18, 65)
        registration_date = start_date_clients + timedelta(days=random.randint(0, 730))
        engagement_score = random.randint(0, 100)
        churn_probability = round(random.uniform(0, 100), 2)
        sentiment_score = random.choice(sentiment_options)
        risk_score = random.randint(0, 100)
        fraud_flagged = random.random() < 0.05 # 5% chance of being flagged

        if engagement_score >= 80 and churn_probability < 20:
            category = "Loyal"
        elif engagement_score <= 30 and churn_probability >= 60:
            category = "Low-Engagement"
        elif churn_probability >= 40 and engagement_score < 70:
            category = "At-Risk"
        else:
            category = random.choice(["Loyal", "High-Value"])

        if category == "At-Risk":
            retention_rec = random.choice(["Offer Discount", "Schedule Call", "Send Personalized Email"])
        elif category == "Low-Engagement":
            retention_rec = random.choice(["Provide Educational Content", "Send Follow-up Survey"])
        elif category == "Loyal" or category == "High-Value":
            retention_rec = "No Action Required"
        else:
            retention_rec = random.choice(retention_recommendations)

        clients_data.append({
            "id": client_id,
            "fullName": full_name,
            "email": f"{first_name.lower()}.{last_name.lower()}{random.randint(1,99)}@example.co.zw".replace(' ', ''),
            "phoneNumber": generate_zim_phone_number(),
            "address": generate_zim_address(location),
            "age": age,
            "gender": gender,
            "location": location,
            "registrationDate": registration_date.strftime("%Y-%m-%d"),
            "engagementScore": engagement_score,
            "churnProbability": churn_probability,
            "category": category,
            "sentimentScore": sentiment_score,
            "riskScore": risk_score,
            "retentionRecommendation": retention_rec,
            "fraudFlagged": fraud_flagged
        })

    claim_counter = 0
    client_claims_map = {client['id']: [] for client in clients_data}

    for client in clients_data:
        num_claims = random.randint(0, max_claims_per_client)
        for _ in range(num_claims):
            claim_counter += 1
            claim_id = f"CL{claim_counter:04d}"
            claim_type = random.choice(claim_types)
            claim_amount = round(random.uniform(100, 50000), 2)
            submission_date = start_date_claims_interactions + timedelta(days=random.randint(0, 365))
            
            client_reg_date = datetime.strptime(client['registrationDate'], "%Y-%m-%d")
            if submission_date < client_reg_date:
                submission_date = client_reg_date + timedelta(days=random.randint(1, 30))

            claim_status = random.choice(claim_statuses)
            claim_probability = random.randint(0, 100)
            fraud_likelihood_score = random.randint(0, 100)
            is_fraud_flagged = random.random() < 0.08 # 8% chance of being flagged for fraud

            supporting_documents = []
            if random.random() < 0.7:
                num_docs = random.randint(1, 3)
                for doc_idx in range(num_docs):
                    supporting_documents.append(f"{claim_id}_{doc_idx+1}_doc.pdf")

            claims_data.append({
                "id": claim_id,
                "clientId": client['id'],
                "claimType": claim_type,
                "claimAmount": claim_amount,
                "submissionDate": submission_date.strftime("%Y-%m-%d"),
                "claimStatus": claim_status,
                "claimProbability": claim_probability,
                "fraudLikelihoodScore": fraud_likelihood_score,
                "isFraudFlagged": is_fraud_flagged,
                "supportingDocuments": supporting_documents
            })
            client_claims_map[client['id']].append({
                "claimAmount": claim_amount,
                "submissionDate": submission_date
            })

            if is_fraud_flagged:
                flagged_date = submission_date + timedelta(days=random.randint(1, 10))
                confirmed_fraud = (random.random() < 0.6)
                fraud_labels_data.append({
                    "claimId": claim_id,
                    "flaggedDate": flagged_date.strftime("%Y-%m-%d"),
                    "confirmedFraud": confirmed_fraud
                })

    for client in clients_data:
        num_interactions = random.randint(5, max_interactions_per_client)
        for _ in range(num_interactions):
            interaction_date = start_date_claims_interactions + timedelta(days=random.randint(0, 365))
            client_reg_date = datetime.strptime(client['registrationDate'], "%Y-%m-%d")
            if interaction_date < client_reg_date:
                interaction_date = client_reg_date + timedelta(days=random.randint(1, 30))

            interaction_logs_data.append({
                "clientId": client['id'],
                "interactionType": random.choice(interaction_types),
                "timestamp": interaction_date.strftime("%Y-%m-%d %H:%M:%S")
            })

    for client in clients_data:
        if random.random() < 0.3:
            num_feedback = random.randint(1, 2)
            for _ in range(num_feedback):
                feedback_submitted_at = start_date_claims_interactions + timedelta(days=random.randint(0, 365))
                client_reg_date = datetime.strptime(client['registrationDate'], "%Y-%m-%d")
                if feedback_submitted_at < client_reg_date:
                    feedback_submitted_at = client_reg_date + timedelta(days=random.randint(1, 30))

                sentiment = random.choice(sentiment_options)
                if sentiment == "Positive":
                    feedback_text = random.choice([
                        "Excellent service, very happy!", "Quick and efficient, thank you.",
                        "Highly recommend, great experience.", "Smooth process, no issues."
                    ])
                elif sentiment == "Neutral":
                    feedback_text = random.choice([
                        "Service was as expected.", "No strong feelings either way.",
                        "It was okay, nothing special.", "Standard interaction."
                    ])
                else: # Negative
                    feedback_text = random.choice([
                        "Very disappointed, service was slow.", "Had issues, needs improvement.",
                        "Not satisfied with the outcome.", "Problems with the app/website."
                    ])

                feedback_data.append({
                    "clientId": client['id'],
                    "submittedAt": feedback_submitted_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "feedbackText": feedback_text,
                    "labeledSentiment": sentiment
                })

    for client in clients_data:
        if client['category'] in ["At-Risk", "Low-Engagement"] or random.random() < 0.1:
            num_actions = random.randint(1, 2)
            for _ in range(num_actions):
                action_type = random.choice(retention_action_types)
                action_date = start_date_claims_interactions + timedelta(days=random.randint(0, 365))
                client_reg_date = datetime.strptime(client['registrationDate'], "%Y-%m-%d")
                if action_date < client_reg_date:
                    action_date = client_reg_date + timedelta(days=random.randint(1, 30))

                if action_type == "Offer Discount" and client['churnProbability'] > 50 and random.random() < 0.7:
                    outcome = "Stayed"
                elif action_type == "Schedule Call" and client['churnProbability'] > 40 and random.random() < 0.6:
                    outcome = "Stayed"
                elif client['churnProbability'] > 70 and random.random() < 0.8:
                    outcome = "Churned"
                else:
                    outcome = random.choice(retention_outcomes)

                retention_actions_data.append({
                    "clientId": client['id'],
                    "actionType": action_type,
                    "actionDate": action_date.strftime("%Y-%m-%d"),
                    "outcome": outcome
                })

    # Generate Client Claim Summary
    client_claim_summary_data = []
    for client in clients_data:
        client_id = client['id']
        claims_for_client = client_claims_map.get(client_id, [])

        total_claims = len(claims_for_client)
        average_claim_amount = 0
        last_claim_date = None

        if total_claims > 0:
            total_amount = sum(c['claimAmount'] for c in claims_for_client)
            average_claim_amount = round(total_amount / total_claims, 2)
            last_claim_date = max(c['submissionDate'] for c in claims_for_client).strftime("%Y-%m-%d")

        client_claim_summary_data.append({
            "clientId": client_id,
            "totalClaims": total_claims,
            "averageClaimAmount": average_claim_amount,
            "lastClaimDate": last_claim_date
        })

    return {
        "clients": clients_data,
        "claims": claims_data,
        "interaction_logs": interaction_logs_data,
        "feedback": feedback_data,
        "retention_actions": retention_actions_data,
        "fraud_labels": fraud_labels_data,
        "client_claim_summary": client_claim_summary_data
    }

# --- Generate the Data ---
all_datasets = generate_data(num_clients=500, max_claims_per_client=4, max_interactions_per_client=60)

# Convert to Pandas DataFrames
df_clients = pd.DataFrame(all_datasets["clients"])
df_claims = pd.DataFrame(all_datasets["claims"])
df_interaction_logs = pd.DataFrame(all_datasets["interaction_logs"])
df_feedback = pd.DataFrame(all_datasets["feedback"])
df_retention_actions = pd.DataFrame(all_datasets["retention_actions"])
df_fraud_labels = pd.DataFrame(all_datasets["fraud_labels"])
df_client_claim_summary = pd.DataFrame(all_datasets["client_claim_summary"])


# --- Save DataFrames to CSV ---
df_clients.to_csv(f"{DATASETS_DIR}/clients.csv", index=False)
df_claims.to_csv(f"{DATASETS_DIR}/claims.csv", index=False)
df_interaction_logs.to_csv(f"{DATASETS_DIR}/interaction_logs.csv", index=False)
df_feedback.to_csv(f"{DATASETS_DIR}/feedback.csv", index=False)
df_retention_actions.to_csv(f"{DATASETS_DIR}/retention_actions.csv", index=False)
df_fraud_labels.to_csv(f"{DATASETS_DIR}/fraud_labels.csv", index=False)
df_client_claim_summary.to_csv(f"{DATASETS_DIR}/client_claim_summary.csv", index=False)

print(f"Generated datasets saved to '{DATASETS_DIR}' folder.")

# --- Print Heads of DataFrames ---
print("\n--- DataFrame Heads (First 10 rows) ---")
print("\nClients Data:")
print(df_clients.head(10))
print("\nClaims Data:")
print(df_claims.head(10))
print("\nInteraction Logs Data:")
print(df_interaction_logs.head(10))
print("\nFeedback Data:")
print(df_feedback.head(10))
print("\nRetention Actions Data:")
print(df_retention_actions.head(10))
print("\nFraud Labels Data:")
print(df_fraud_labels.head(10))
print("\nClient Claim Summary Data:")
print(df_client_claim_summary.head(10))


# --- Data Preprocessing for Models ---

# Merge client_claim_summary into clients for richer features
df_clients_enriched = df_clients.merge(df_client_claim_summary, left_on='id', right_on='clientId', how='left')
df_clients_enriched['totalClaims'] = df_clients_enriched['totalClaims'].fillna(0).astype(int)
df_clients_enriched['averageClaimAmount'] = df_clients_enriched['averageClaimAmount'].fillna(0)
df_clients_enriched['lastClaimDate'] = pd.to_datetime(df_clients_enriched['lastClaimDate'], errors='coerce')


# Feature Engineering for Client Churn/Risk Models
df_clients_enriched['registrationYear'] = pd.to_datetime(df_clients_enriched['registrationDate']).dt.year
df_clients_enriched['registrationMonth'] = pd.to_datetime(df_clients_enriched['registrationDate']).dt.month
df_clients_enriched['recency_last_claim'] = (datetime.now() - df_clients_enriched['lastClaimDate']).dt.days.fillna(-1)
df_clients_enriched['recency_registration'] = (datetime.now() - pd.to_datetime(df_clients_enriched['registrationDate'])).dt.days

# For interaction logs, aggregate features per client
df_interaction_logs['timestamp'] = pd.to_datetime(df_interaction_logs['timestamp'])
interaction_counts = df_interaction_logs.groupby('clientId')['interactionType'].count().rename('total_interactions')
last_interaction = df_interaction_logs.groupby('clientId')['timestamp'].max().rename('last_interaction_date')
df_clients_enriched = df_clients_enriched.merge(interaction_counts, left_on='id', right_index=True, how='left')
df_clients_enriched = df_clients_enriched.merge(last_interaction, left_on='id', right_index=True, how='left')
df_clients_enriched['total_interactions'] = df_clients_enriched['total_interactions'].fillna(0) 
df_clients_enriched['recency_last_interaction'] = (datetime.now() - df_clients_enriched['last_interaction_date']).dt.days.fillna(-1)

# Ensure 'outcome' is a numerical target for retention_actions
df_retention_actions['outcome_encoded'] = df_retention_actions['outcome'].apply(lambda x: 1 if x == 'Stayed' else 0)


# --- 1. Client Churn Prediction Model (XGBoost) ---
print("\n--- Training Client Churn Prediction Model ---")
# Features for churn prediction
features_churn = [
    'age', 'engagementScore', 'totalClaims', 'averageClaimAmount',
    'riskScore', 'total_interactions', 'recency_last_claim', 'recency_last_interaction',
    'recency_registration'
]
categorical_features_churn = ['gender', 'location', 'category', 'sentimentScore']

df_clients_churn_data = df_clients_enriched.copy()
df_clients_churn_data['churned_actual'] = (df_clients_churn_data['churnProbability'] > 70).astype(int)

churned_via_retention = df_retention_actions[df_retention_actions['outcome'] == 'Churned']['clientId'].unique()
df_clients_churn_data['churned_actual'] = df_clients_churn_data.apply(
    lambda row: 1 if row['id'] in churned_via_retention else row['churned_actual'], axis=1
)

preprocessor_churn = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), [f for f in features_churn if f in df_clients_churn_data.columns]),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_churn)
    ],
    remainder='passthrough'
)

df_clients_churn_data_filtered = df_clients_churn_data.dropna(subset=features_churn + categorical_features_churn).copy()

X = df_clients_churn_data_filtered[features_churn + categorical_features_churn]
y = df_clients_churn_data_filtered['churned_actual']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model_churn_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_churn),
    ('classifier', xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42))
])

model_churn_pipeline.fit(X_train, y_train)

# Save model
joblib.dump(model_churn_pipeline, f"{MODELS_DIR}/churn_prediction_model.pkl")

y_pred_churn = model_churn_pipeline.predict(X_test)
y_prob_churn = model_churn_pipeline.predict_proba(X_test)[:, 1]

print(f"Churn Prediction Accuracy: {accuracy_score(y_test, y_pred_churn):.4f}")
print("Churn Prediction Classification Report:\n", classification_report(y_test, y_pred_churn))

def predict_churn(client_data: dict, model_path, features_churn, categorical_features_churn):
    model_pipeline = joblib.load(model_path)
    input_df = pd.DataFrame([client_data])
    
    for col in (features_churn + categorical_features_churn):
        if col not in input_df.columns:
            if col in features_churn:
                input_df[col] = 0.0
            else:
                input_df[col] = "Unknown"

    input_df['registrationDate'] = pd.to_datetime(input_df['registrationDate'], errors='coerce')
    input_df['lastClaimDate'] = pd.to_datetime(input_df['lastClaimDate'], errors='coerce')
    input_df['last_interaction_date'] = pd.to_datetime(input_df['last_interaction_date'], errors='coerce')
        
    input_df['registrationYear'] = input_df['registrationDate'].dt.year.fillna(0).astype(int)
    input_df['registrationMonth'] = input_df['registrationDate'].dt.month.fillna(0).astype(int)

    input_df['recency_last_claim'] = (datetime.now() - input_df['lastClaimDate'].fillna(pd.NaT)).dt.days.fillna(-1) 
    input_df['recency_registration'] = (datetime.now() - input_df['registrationDate'].fillna(pd.NaT)).dt.days
    input_df['total_interactions'] = input_df['total_interactions'].fillna(0)
    input_df['averageClaimAmount'] = input_df['averageClaimAmount'].fillna(0)
    input_df['recency_last_interaction'] = (datetime.now() - input_df['last_interaction_date'].fillna(pd.NaT)).dt.days.fillna(-1)

    X_predict = input_df[features_churn + categorical_features_churn]
    
    churn_prob = model_pipeline.predict_proba(X_predict)[:, 1][0]
    risk_category = "High Risk" if churn_prob > 0.6 else ("Medium Risk" if churn_prob > 0.3 else "Low Risk")
    return {"churn_probability": round(churn_prob * 100, 2), "risk_category": risk_category}

example_client_data_churn = {
    'age': 35, 'engagementScore': 20, 'totalClaims': 1, 'averageClaimAmount': 500,
    'riskScore': 80, 'total_interactions': 5,
    'gender': 'Female', 'location': 'Harare', 'category': 'At-Risk', 'sentimentScore': 'Negative',
    'registrationDate': '2023-01-15', 'lastClaimDate': '2024-03-20', 'last_interaction_date': '2025-04-01'
}
churn_pred = predict_churn(example_client_data_churn, f"{MODELS_DIR}/churn_prediction_model.pkl", features_churn, categorical_features_churn)
print(f"\nExample Churn Prediction for a new client: {churn_pred}")


# --- 2. Claims Submission Forecasting Model (XGBoost) ---
print("\n--- Training Claims Submission Forecasting Model ---")

df_claims_forecast_data = df_clients_enriched.copy()
df_claims_forecast_data['has_filed_claim'] = (df_claims_forecast_data['totalClaims'] > 0).astype(int)

features_claims_forecast = [
    'age', 'engagementScore', 'total_interactions', 'recency_registration',
    'churnProbability', 'riskScore'
]
categorical_features_claims_forecast = ['gender', 'location', 'category', 'sentimentScore']

preprocessor_claims = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), [f for f in features_claims_forecast if f in df_claims_forecast_data.columns]),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_claims_forecast)
    ],
    remainder='passthrough'
)

df_claims_forecast_data_filtered = df_claims_forecast_data.dropna(subset=features_claims_forecast + categorical_features_claims_forecast).copy()

X_claims = df_claims_forecast_data_filtered[features_claims_forecast + categorical_features_claims_forecast]
y_claims = df_claims_forecast_data_filtered['has_filed_claim']

X_train_claims, X_test_claims, y_train_claims, y_test_claims = train_test_split(X_claims, y_claims, test_size=0.2, random_state=42, stratify=y_claims)

model_claims_forecast_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_claims),
    ('classifier', xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42))
])

model_claims_forecast_pipeline.fit(X_train_claims, y_train_claims)

# Save model
joblib.dump(model_claims_forecast_pipeline, f"{MODELS_DIR}/claims_forecast_model.pkl")

y_pred_claims = model_claims_forecast_pipeline.predict(X_test_claims)
y_prob_claims = model_claims_forecast_pipeline.predict_proba(X_test_claims)[:, 1]

print(f"Claims Submission Prediction Accuracy: {accuracy_score(y_test_claims, y_pred_claims):.4f}")
print("Claims Submission Prediction Classification Report:\n", classification_report(y_test_claims, y_pred_claims))

def predict_claims_submission(client_data: dict, model_path, features_claims_forecast, categorical_features_claims_forecast):
    model_pipeline = joblib.load(model_path)
    input_df = pd.DataFrame([client_data])
    for col in (features_claims_forecast + categorical_features_claims_forecast):
        if col not in input_df.columns:
            if col in features_claims_forecast:
                input_df[col] = 0.0
            else:
                input_df[col] = "Unknown"
            
    input_df['registrationDate'] = pd.to_datetime(input_df['registrationDate'], errors='coerce')

    input_df['recency_registration'] = (datetime.now() - input_df['registrationDate'].fillna(pd.NaT)).dt.days.fillna(-1)

    X_predict = input_df[features_claims_forecast + categorical_features_claims_forecast]
    
    claim_prob = model_pipeline.predict_proba(X_predict)[:, 1][0]
    
    if claim_prob > 0.5:
        predicted_claim_type = random.choice(claim_types)
    else:
        predicted_claim_type = "No Claim Expected"
        
    return {"claim_probability": round(claim_prob * 100, 2), "predicted_claim_type": predicted_claim_type}

example_client_data_claims = {
    'age': 40, 'engagementScore': 75, 'total_interactions': 45,
    'churnProbability': 15, 'riskScore': 20,
    'gender': 'Male', 'location': 'Bulawayo', 'category': 'Loyal', 'sentimentScore': 'Positive',
    'registrationDate': '2024-05-01'
}
claims_pred = predict_claims_submission(example_client_data_claims, f"{MODELS_DIR}/claims_forecast_model.pkl", features_claims_forecast, categorical_features_claims_forecast)
print(f"\nExample Claims Submission Prediction for a new client: {claims_pred}")


# --- 3. Client Segmentation Model (K-Means) ---
print("\n--- Training Client Segmentation Model ---")
features_segmentation = [
    'age', 'engagementScore', 'totalClaims', 'averageClaimAmount',
    'churnProbability', 'riskScore', 'total_interactions',
    'recency_last_claim', 'recency_last_interaction', 'recency_registration'
]
df_segmentation_data = df_clients_enriched.dropna(subset=features_segmentation).copy()
X_segmentation = df_segmentation_data[features_segmentation]

scaler_segmentation = StandardScaler()
X_segmentation_scaled = scaler_segmentation.fit_transform(X_segmentation)

num_clusters = 4

model_segmentation = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
df_segmentation_data['cluster'] = model_segmentation.fit_predict(X_segmentation_scaled)

cluster_mapping = {}
for i in range(num_clusters):
    cluster_mean = df_segmentation_data[df_segmentation_data['cluster'] == i][
        ['engagementScore', 'churnProbability', 'totalClaims', 'averageClaimAmount', 'recency_last_interaction']
    ].mean()
    
    if cluster_mean['churnProbability'] < 30 and cluster_mean['engagementScore'] > 70:
        cluster_mapping[i] = "Loyal"
    elif cluster_mean['churnProbability'] > 60 and cluster_mean['engagementScore'] < 40:
        cluster_mapping[i] = "At-Risk"
    elif cluster_mean['totalClaims'] > 1 and cluster_mean['averageClaimAmount'] > 1000:
        cluster_mapping[i] = "High-Value"
    else:
        cluster_mapping[i] = "Low-Engagement"

df_segmentation_data['predicted_category'] = df_segmentation_data['cluster'].map(cluster_mapping)
print("\nExample of predicted categories vs. original categories:")
print(df_segmentation_data[['category', 'predicted_category']].head())

# Save model and scaler
joblib.dump(model_segmentation, f"{MODELS_DIR}/client_segmentation_model.pkl")
joblib.dump(scaler_segmentation, f"{MODELS_DIR}/scaler_segmentation.pkl")
joblib.dump(cluster_mapping, f"{MODELS_DIR}/cluster_mapping.pkl")


def predict_segmentation(client_data: dict, model_path, scaler_path, cluster_mapping_path, features_list):
    model_kmeans = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    cluster_mapping = joblib.load(cluster_mapping_path)

    input_df = pd.DataFrame([client_data])
    for col in features_list:
        if col not in input_df.columns:
            input_df[col] = 0.0
            
    input_df['registrationDate'] = pd.to_datetime(input_df['registrationDate'], errors='coerce')
    input_df['lastClaimDate'] = pd.to_datetime(input_df['lastClaimDate'], errors='coerce')
    input_df['last_interaction_date'] = pd.to_datetime(input_df['last_interaction_date'], errors='coerce')
        
    input_df['registrationYear'] = input_df['registrationDate'].dt.year.fillna(0).astype(int)
    input_df['registrationMonth'] = input_df['registrationDate'].dt.month.fillna(0).astype(int)
    
    input_df['recency_last_claim'] = (datetime.now() - input_df['lastClaimDate'].fillna(pd.NaT)).dt.days.fillna(-1)
    input_df['recency_registration'] = (datetime.now() - input_df['registrationDate'].fillna(pd.NaT)).dt.days
    input_df['total_interactions'] = input_df['total_interactions'].fillna(0)
    input_df['averageClaimAmount'] = input_df['averageClaimAmount'].fillna(0)
    input_df['recency_last_interaction'] = (datetime.now() - input_df['last_interaction_date'].fillna(pd.NaT)).dt.days.fillna(-1)

    X_predict = input_df[features_list].values.reshape(1, -1)
    X_predict_scaled = scaler.transform(X_predict)
    cluster = model_kmeans.predict(X_predict_scaled)[0]
    predicted_category = cluster_mapping.get(cluster, "Unknown")
    return {"segment_cluster": int(cluster), "predicted_category": predicted_category}

example_client_data_segmentation = {
    'age': 28, 'engagementScore': 90, 'totalClaims': 0, 'averageClaimAmount': 0,
    'churnProbability': 5, 'riskScore': 10, 'total_interactions': 50,
    'registrationDate': '2025-01-15', 'lastClaimDate': np.nan, 'last_interaction_date': '2025-05-25'
}
segment_pred = predict_segmentation(example_client_data_segmentation, 
                                    f"{MODELS_DIR}/client_segmentation_model.pkl", 
                                    f"{MODELS_DIR}/scaler_segmentation.pkl", 
                                    f"{MODELS_DIR}/cluster_mapping.pkl", 
                                    features_segmentation)
print(f"\nExample Client Segmentation for a new client: {segment_pred}")


# --- 4. Sentiment Analysis Model (VADER) ---
print("\n--- Training Sentiment Analysis Model (VADER) ---")
analyzer = SentimentIntensityAnalyzer()

df_feedback['vader_sentiment_scores'] = df_feedback['feedbackText'].apply(analyzer.polarity_scores)
df_feedback['compound_score'] = df_feedback['vader_sentiment_scores'].apply(lambda x: x['compound'])

def vader_label(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df_feedback['vader_labeled_sentiment'] = df_feedback['compound_score'].apply(vader_label)

accuracy_vader = accuracy_score(df_feedback['labeledSentiment'], df_feedback['vader_labeled_sentiment'])
print(f"VADER Sentiment Analysis Accuracy (vs dummy labels): {accuracy_vader:.4f}")
print("VADER Sentiment Analysis Classification Report (vs dummy labels):\n", classification_report(df_feedback['labeledSentiment'], df_feedback['vader_labeled_sentiment']))

# VADER doesn't need to be saved/loaded like sklearn models, it's a rule-based system.
# The analyzer object is sufficient.

def predict_sentiment(feedback_text: str):
    analyzer = SentimentIntensityAnalyzer() # Initialize VADER for each call or pass it
    scores = analyzer.polarity_scores(feedback_text)
    compound_score = scores['compound']
    sentiment_label = vader_label(compound_score)
    return {"compound_score": compound_score, "sentiment_label": sentiment_label}

example_feedback_text_positive = "The service was absolutely fantastic! Very satisfied."
example_feedback_text_negative = "I am extremely frustrated with the slow response times."
example_feedback_text_neutral = "The product is functional, nothing special."

sentiment_pos = predict_sentiment(example_feedback_text_positive)
sentiment_neg = predict_sentiment(example_feedback_text_negative)
sentiment_neu = predict_sentiment(example_feedback_text_neutral)

print(f"\nExample Sentiment Analysis for '{example_feedback_text_positive}': {sentiment_pos}")
print(f"Example Sentiment Analysis for '{example_feedback_text_negative}': {sentiment_neg}")
print(f"Example Sentiment Analysis for '{example_feedback_text_neutral}': {sentiment_neu}")


# --- 5. Risk Scoring Model (Ensemble Model: XGBoost + Clustering + NLP) ---
print("\n--- Training Risk Scoring Model ---")

df_risk_data = df_clients_enriched.copy()

# Ensure these are the pipelines that were just trained/saved
churn_model_pipeline_loaded = joblib.load(f"{MODELS_DIR}/churn_prediction_model.pkl")
segmentation_model_loaded = joblib.load(f"{MODELS_DIR}/client_segmentation_model.pkl")
scaler_segmentation_loaded = joblib.load(f"{MODELS_DIR}/scaler_segmentation.pkl")

df_risk_data['predicted_churn_prob'] = churn_model_pipeline_loaded.predict_proba(
    df_risk_data[features_churn + categorical_features_churn]
)[:, 1]

df_risk_data['predicted_segment_cluster'] = segmentation_model_loaded.predict(
    scaler_segmentation_loaded.transform(df_risk_data[features_segmentation].fillna(0))
)

client_avg_sentiment = df_feedback.groupby('clientId')['compound_score'].mean().rename('avg_sentiment_compound')
df_risk_data = df_risk_data.merge(client_avg_sentiment, left_on='id', right_index=True, how='left')
df_risk_data['avg_sentiment_compound'] = df_risk_data['avg_sentiment_compound'].fillna(0)

features_risk = [
    'age', 'engagementScore', 'totalClaims', 'averageClaimAmount',
    'churnProbability', 'total_interactions', 'recency_last_claim', 'recency_last_interaction',
    'recency_registration', 'predicted_churn_prob', 'predicted_segment_cluster', 'avg_sentiment_compound'
]
categorical_features_risk = ['gender', 'location']

preprocessor_risk = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), [f for f in features_risk if f in df_risk_data.columns]),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_risk)
    ],
    remainder='passthrough'
)

df_risk_data_filtered = df_risk_data.dropna(subset=features_risk + categorical_features_risk).copy()

X_risk = df_risk_data_filtered[features_risk + categorical_features_risk]
y_risk = df_risk_data_filtered['riskScore']

X_train_risk, X_test_risk, y_train_risk, y_test_risk = train_test_split(X_risk, y_risk, test_size=0.2, random_state=42)

model_risk_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_risk),
    ('regressor', xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42))
])

model_risk_pipeline.fit(X_train_risk, y_train_risk)

# Save model
joblib.dump(model_risk_pipeline, f"{MODELS_DIR}/risk_scoring_model.pkl")

y_pred_risk = model_risk_pipeline.predict(X_test_risk)

print(f"Risk Scoring Model MAE: {mean_absolute_error(y_test_risk, y_pred_risk):.4f}")
print(f"Risk Scoring Model RMSE: {np.sqrt(mean_squared_error(y_test_risk, y_pred_risk)):.4f}")

def predict_risk_score(client_data: dict, models_dir,
                       features_churn, categorical_features_churn, features_segmentation, features_risk, categorical_features_risk):
    
    # Load models from the specified directory
    churn_model_pipeline = joblib.load(f"{models_dir}/churn_prediction_model.pkl")
    segmentation_model = joblib.load(f"{models_dir}/client_segmentation_model.pkl")
    scaler_segmentation = joblib.load(f"{models_dir}/scaler_segmentation.pkl")
    risk_model_pipeline = joblib.load(f"{models_dir}/risk_scoring_model.pkl")
    sentiment_analyzer = SentimentIntensityAnalyzer() # VADER is rule-based, can be initialized or passed

    input_df_single = pd.DataFrame([client_data])
    
    all_expected_cols = list(set(features_churn + categorical_features_churn + features_segmentation + features_risk + categorical_features_risk + 
                                 ['registrationDate', 'lastClaimDate', 'last_interaction_date', 'totalClaims', 'averageClaimAmount']))
    for col in all_expected_cols:
        if col not in input_df_single.columns:
            if col in ['registrationDate', 'lastClaimDate', 'last_interaction_date']:
                input_df_single[col] = np.nan
            elif col in ['totalClaims', 'total_interactions', 'age', 'engagementScore', 'churnProbability', 'riskScore', 'averageClaimAmount']:
                input_df_single[col] = 0.0
            else:
                input_df_single[col] = "Unknown"

    input_df_single['registrationDate'] = pd.to_datetime(input_df_single['registrationDate'], errors='coerce')
    input_df_single['lastClaimDate'] = pd.to_datetime(input_df_single['lastClaimDate'], errors='coerce')
    input_df_single['last_interaction_date'] = pd.to_datetime(input_df_single['last_interaction_date'], errors='coerce')
        
    input_df_single['registrationYear'] = input_df_single['registrationDate'].dt.year.fillna(0).astype(int)
    input_df_single['registrationMonth'] = input_df_single['registrationDate'].dt.month.fillna(0).astype(int)
    
    input_df_single['recency_last_claim'] = (datetime.now() - input_df_single['lastClaimDate'].fillna(pd.NaT)).dt.days.fillna(-1)
    input_df_single['recency_registration'] = (datetime.now() - input_df_single['registrationDate'].fillna(pd.NaT)).dt.days
    input_df_single['total_interactions'] = input_df_single['total_interactions'].fillna(0)
    input_df_single['averageClaimAmount'] = input_df_single['averageClaimAmount'].fillna(0)
    input_df_single['recency_last_interaction'] = (datetime.now() - input_df_single['last_interaction_date'].fillna(pd.NaT)).dt.days.fillna(-1)

    churn_prob_data = input_df_single[features_churn + categorical_features_churn]
    predicted_churn_prob = churn_model_pipeline.predict_proba(churn_prob_data)[:, 1][0]
    input_df_single['predicted_churn_prob'] = predicted_churn_prob

    segment_data = input_df_single[features_segmentation]
    segment_data = segment_data.fillna(0)
    predicted_segment_cluster = segmentation_model.predict(scaler_segmentation.transform(segment_data))[0]
    input_df_single['predicted_segment_cluster'] = predicted_segment_cluster

    if 'feedbackText' in client_data and client_data['feedbackText']:
        sentiment_result = sentiment_analyzer.polarity_scores(client_data['feedbackText'])
        avg_sentiment_compound = sentiment_result['compound']
    else:
        avg_sentiment_compound = 0
    input_df_single['avg_sentiment_compound'] = avg_sentiment_compound;

    X_predict_risk = input_df_single[features_risk + categorical_features_risk]
    
    predicted_risk_score = risk_model_pipeline.predict(X_predict_risk)[0]
    predicted_risk_score = max(0, min(100, round(predicted_risk_score)))

    if predicted_risk_score > 70:
        recommendation = "Offer Discount / Urgent Call"
    elif predicted_risk_score > 40:
        recommendation = "Personalized Email / Proactive Check-in"
    else:
        recommendation = "Monitor Activity / No Action Required"

    return {"predicted_risk_score": predicted_risk_score, "retention_recommendation": recommendation}

example_client_data_risk = {
    'age': 50, 'engagementScore': 30, 'totalClaims': 2, 'averageClaimAmount': 1200,
    'churnProbability': 65, 'total_interactions': 10,
    'gender': 'Male', 'location': 'Gweru', 'category': 'At-Risk', 'sentimentScore': 'Negative',
    'registrationDate': '2023-08-01', 'lastClaimDate': '2025-03-01', 'last_interaction_date': '2025-04-15',
    'feedbackText': 'I am very unhappy with the recent changes. My claims are taking too long!'
}
risk_pred = predict_risk_score(example_client_data_risk, MODELS_DIR,
                               features_churn, categorical_features_churn, features_segmentation, features_risk, categorical_features_risk)
print(f"\nExample Risk Score and Recommendation for a new client: {risk_pred}")


# --- 6. Fraud Detection Model (Isolation Forest) ---
print("\n--- Training Fraud Detection Model ---")

df_fraud_data = df_claims.merge(df_fraud_labels, left_on='id', right_on='claimId', how='left', suffixes=('_claim', '_label'))
df_fraud_data['confirmedFraud'] = df_fraud_data['confirmedFraud'].fillna(False).astype(bool)
df_fraud_data['isFraudFlagged'] = df_fraud_data['isFraudFlagged'].astype(bool)

features_fraud = [
    'claimAmount', 'claimProbability', 'fraudLikelihoodScore'
]
categorical_features_fraud = ['claimType', 'claimStatus']

preprocessor_fraud = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features_fraud),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_fraud)
    ],
    remainder='passthrough'
)

df_fraud_data_filtered = df_fraud_data.dropna(subset=features_fraud + categorical_features_fraud).copy()

X_fraud = df_fraud_data_filtered[features_fraud + categorical_features_fraud]
preprocessor_fraud.fit(X_fraud)
X_fraud_processed = preprocessor_fraud.transform(X_fraud)


model_fraud = IsolationForest(random_state=42, contamination=0.1)
model_fraud.fit(X_fraud_processed)

# Save model and preprocessor
joblib.dump(model_fraud, f"{MODELS_DIR}/fraud_detection_model.pkl")
joblib.dump(preprocessor_fraud, f"{MODELS_DIR}/preprocessor_fraud.pkl")


df_fraud_data_filtered['anomaly_prediction'] = model_fraud.predict(X_fraud_processed)
df_fraud_data_filtered['anomaly_score'] = model_fraud.decision_function(X_fraud_processed)

fraud_match_count = df_fraud_data_filtered[
    (df_fraud_data_filtered['isFraudFlagged'] == True) & (df_fraud_data_filtered['anomaly_prediction'] == -1)
].shape[0]
total_flagged_claims = df_fraud_data_filtered['isFraudFlagged'].sum()
total_predicted_outliers = (df_fraud_data_filtered['anomaly_prediction'] == -1).sum()

print(f"Total claims flagged for fraud (dummy): {total_flagged_claims}")
print(f"Total claims predicted as outliers by Isolation Forest: {total_predicted_outliers}")
print(f"Flagged claims correctly identified as outliers: {fraud_match_count}")

def predict_fraud(claim_data: dict, model_path, preprocessor_path, features_list, categorical_features_list):
    model_fraud = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    input_df_single = pd.DataFrame([claim_data])
    
    for col in (features_list + categorical_features_list):
        if col not in input_df_single.columns:
            if col in features_list:
                input_df_single[col] = 0.0
            elif col in categorical_features_list:
                input_df_single[col] = "Unknown"

    X_predict = input_df_single[features_list + categorical_features_list]
    X_predict_processed = preprocessor.transform(X_predict)
    
    anomaly_score = model_fraud.decision_function(X_predict_processed)[0]
    
    # To get min/max for scaling, we need a reference. In a real system, you'd save these from training.
    # For this example, we'll quickly re-process the training data to get reference scores.
    # In production, you'd save preprocessor_fraud and a min/max score from fitting.
    # X_fraud_processed_train = preprocessor.transform(df_fraud_data_filtered[features_list + categorical_features_list])
    # min_score = model_fraud.decision_function(X_fraud_processed_train).min()
    # max_score = model_fraud.decision_function(X_fraud_processed_train).max()
    
    # For consistent scoring, let's just use a fixed min/max or pass them as arguments
    # (more robust would be to save these with the model)
    # For now, let's make dummy scores for the example to work without a full re-train/re-process for min/max
    # A robust solution would be to save min_score and max_score from the training data along with the model.
    # For demonstration, we'll use a hardcoded range based on typical Isolation Forest output for this data.
    min_score_ref = -0.2 # Example reference min score
    max_score_ref = 0.2  # Example reference max score


    if (max_score_ref - min_score_ref) == 0:
        fraud_likelihood_percent = 50
    else:
        fraud_likelihood_percent = 100 - ((anomaly_score - min_score_ref) * 100 / (max_score_ref - min_score_ref))
    fraud_likelihood_percent = max(0, min(100, round(fraud_likelihood_percent)))

    is_flagged = model_fraud.predict(X_predict_processed)[0] == -1

    return {"fraud_likelihood_score": fraud_likelihood_percent, "is_fraud_flagged": bool(is_flagged)}

example_claim_data_fraud = {
    'claimAmount': 50000, 'claimProbability': 95, 'fraudLikelihoodScore': 90,
    'claimType': 'Personal Injury', 'claimStatus': 'Pending'
}
example_claim_data_normal = {
    'claimAmount': 500, 'claimProbability': 20, 'fraudLikelihoodScore': 10,
    'claimType': 'Property Damage', 'claimStatus': 'Approved'
}

fraud_pred_high = predict_fraud(example_claim_data_fraud, f"{MODELS_DIR}/fraud_detection_model.pkl", f"{MODELS_DIR}/preprocessor_fraud.pkl", features_fraud, categorical_features_fraud)
fraud_pred_low = predict_fraud(example_claim_data_normal, f"{MODELS_DIR}/fraud_detection_model.pkl", f"{MODELS_DIR}/preprocessor_fraud.pkl", features_fraud, categorical_features_fraud)

print(f"\nExample Fraud Detection for high-risk claim: {fraud_pred_high}")
print(f"Example Fraud Detection for low-risk claim: {fraud_pred_low}")


# --- 7. Real-Time Recommendation Model (Collaborative Filtering) ---
print("\n--- Training Real-Time Recommendation Model ---")

# Define all unique client IDs and action types explicitly
all_client_ids_for_mapping = pd.concat([df_clients['id'], df_retention_actions['clientId']]).unique()
all_action_types_for_mapping = df_retention_actions['actionType'].unique()

# Create consistent mappings
client_id_to_encoded = {client_id: i for i, client_id in enumerate(all_client_ids_for_mapping)}
encoded_to_client_id = {i: client_id for client_id, i in client_id_to_encoded.items()}

action_type_to_encoded = {action_type: i for i, action_type in enumerate(all_action_types_for_mapping)}
encoded_to_action_type = {i: action_type for action_type, i in action_type_to_encoded.items()}

df_interactions_cf = df_retention_actions.copy()
df_interactions_cf['interaction_value'] = df_interactions_cf['outcome'].apply(lambda x: 1 if x == 'Stayed' else 0)

# Apply the consistent mappings
df_interactions_cf['client_id_encoded'] = df_interactions_cf['clientId'].map(client_id_to_encoded)
df_interactions_cf['action_type_encoded'] = df_interactions_cf['actionType'].map(action_type_to_encoded)

# Filter out any rows where mapping failed (e.g., if a client_id appeared in df_retention_actions
# but wasn't in all_client_ids_for_mapping, which shouldn't happen with the concat)
df_interactions_cf.dropna(subset=['client_id_encoded', 'action_type_encoded'], inplace=True)

# Convert encoded IDs to integer type
df_interactions_cf['client_id_encoded'] = df_interactions_cf['client_id_encoded'].astype(int)
df_interactions_cf['action_type_encoded'] = df_interactions_cf['action_type_encoded'].astype(int)

num_users = len(client_id_to_encoded) # Use the length of the mapping
num_items = len(action_type_to_encoded) # Use the length of the mapping

row_ind = df_interactions_cf['action_type_encoded'].values
col_ind = df_interactions_cf['client_id_encoded'].values
data = df_interactions_cf['interaction_value'].values

# Check for empty data after mapping, which could cause issues
if not np.any(data):
    print("Warning: No interaction data found after encoding. Collaborative Filtering model may not train effectively.")

sparse_item_user = sparse.csc_matrix((data, (row_ind, col_ind)), shape=(num_items, num_users))

# Ensure the sparse matrix dimensions match the num_users and num_items derived from mappings
# This is a critical check for consistency
if sparse_item_user.shape != (num_items, num_users):
    raise ValueError(f"Sparse matrix shape mismatch! Expected ({num_items}, {num_users}), got {sparse_item_user.shape}")

model_cf = AlternatingLeastSquares(factors=100, regularization=0.01, alpha=2.0, iterations=50, random_state=42)
model_cf.fit(sparse_item_user)

# Save CF model and mappings
joblib.dump(model_cf, f"{MODELS_DIR}/collaborative_filtering_model.pkl")
joblib.dump(client_id_to_encoded, f"{MODELS_DIR}/client_id_to_encoded.pkl")
joblib.dump(encoded_to_client_id, f"{MODELS_DIR}/encoded_to_client_id.pkl")
joblib.dump(action_type_to_encoded, f"{MODELS_DIR}/action_type_to_encoded.pkl")
joblib.dump(encoded_to_action_type, f"{MODELS_DIR}/encoded_to_action_type.pkl")
joblib.dump(sparse_item_user, f"{MODELS_DIR}/sparse_item_user.pkl")


print("\nCollaborative Filtering Model trained.")

def recommend_retention_actions(client_id: str, models_dir, N=3):
    """
    Recommends top N retention actions for a given client, loading models from specified directory.
    `client_id`: ID of the client (e.g., 'C001').
    `models_dir`: Path to the directory containing saved models and mappings.
    `N`: Number of recommendations to return.
    """
    # Load models and mappings
    model_cf = joblib.load(f"{models_dir}/collaborative_filtering_model.pkl")
    client_id_to_encoded = joblib.load(f"{models_dir}/client_id_to_encoded.pkl")
    encoded_to_client_id = joblib.load(f"{models_dir}/encoded_to_client_id.pkl")
    action_type_to_encoded = joblib.load(f"{models_dir}/action_type_to_encoded.pkl")
    encoded_to_action_type = joblib.load(f"{models_dir}/encoded_to_action_type.pkl")
    sparse_item_user = joblib.load(f"{models_dir}/sparse_item_user.pkl")

    if client_id not in client_id_to_encoded:
        print(f"Client {client_id} not found in interaction history. Providing popular recommendations.")
        item_popularity = sparse_item_user.sum(axis=1).A1
        popular_item_indices = np.argsort(item_popularity)[::-1][:N]
        popular_actions = [encoded_to_action_type[idx] for idx in popular_item_indices if idx in encoded_to_action_type]
        return {"recommendations": popular_actions, "message": "Recommendations based on overall popularity (new client)."}
    
    client_encoded_id = client_id_to_encoded[client_id]
    
    user_interactions = sparse_item_user.T.tocsr()[client_encoded_id:client_encoded_id+1]
    
    print(f"DEBUG: client_encoded_id: {client_encoded_id}")
    print(f"DEBUG: user_interactions shape: {user_interactions.shape}")
    
    recommendations_encoded, scores = model_cf.recommend(
        0, 
        user_interactions, 
        N=N, 
        filter_already_liked_items=True
    )
    
    valid_recommendations = [(rec, score) for rec, score in zip(recommendations_encoded, scores) if rec in encoded_to_action_type]
    
    recommendations = [encoded_to_action_type[rec] for rec, _ in valid_recommendations]
    valid_scores = [score for _, score in valid_recommendations]

    return {"recommendations": recommendations, "scores": valid_scores}

example_client_id_cf = df_retention_actions['clientId'].sample(1, random_state=42).iloc[0]
print(f"\nRecommending for client: {example_client_id_cf}")

rec_actions = recommend_retention_actions(
    example_client_id_cf, MODELS_DIR, N=3
)
print(f"Recommended actions for client {example_client_id_cf}: {rec_actions}")

new_client_id_cf = "C999" 
rec_actions_new_client = recommend_retention_actions(
    new_client_id_cf, MODELS_DIR, N=3
)
print(f"Recommended actions for new client {new_client_id_cf}: {rec_actions_new_client}")