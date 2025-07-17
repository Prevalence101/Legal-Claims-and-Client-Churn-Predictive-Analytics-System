import json
import random
from datetime import datetime, timedelta

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

# --- Data Generation ---

def generate_data(num_clients=150, max_claims_per_client=5, max_interactions_per_client=50):
    clients_data = []
    claims_data = []
    interaction_logs_data = []
    feedback_data = []
    retention_actions_data = []
    fraud_labels_data = []
    client_claim_summary_data = [] # This will be derived later

    start_date_clients = datetime.now() - timedelta(days=730) # 2 years ago
    start_date_claims_interactions = datetime.now() - timedelta(days=365) # 1 year ago

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

        # Correlate category with engagement/churn
        if engagement_score >= 80 and churn_probability < 20:
            category = "Loyal"
        elif engagement_score <= 30 and churn_probability >= 60:
            category = "Low-Engagement"
        elif churn_probability >= 40 and engagement_score < 70:
            category = "At-Risk"
        else:
            category = random.choice(["Loyal", "High-Value"]) # Default to these if no clear pattern

        # Correlate retention recommendation
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

    # Generate Claims
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
            
            # Ensure claim date is after client registration date
            client_reg_date = datetime.strptime(client['registrationDate'], "%Y-%m-%d")
            if submission_date < client_reg_date:
                submission_date = client_reg_date + timedelta(days=random.randint(1, 30))

            claim_status = random.choice(claim_statuses)
            claim_probability = random.randint(0, 100)
            fraud_likelihood_score = random.randint(0, 100)
            is_fraud_flagged = random.random() < 0.08 # 8% chance of being flagged for fraud

            supporting_documents = []
            if random.random() < 0.7: # 70% chance of having documents
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

            # If claim is flagged for fraud, add to fraud_labels
            if is_fraud_flagged:
                flagged_date = submission_date + timedelta(days=random.randint(1, 10))
                confirmed_fraud = (random.random() < 0.6) # 60% of flagged claims are confirmed fraud
                fraud_labels_data.append({
                    "claimId": claim_id,
                    "flaggedDate": flagged_date.strftime("%Y-%m-%d"),
                    "confirmedFraud": confirmed_fraud
                })


    # Generate Interaction Logs
    for client in clients_data:
        num_interactions = random.randint(5, max_interactions_per_client)
        for _ in range(num_interactions):
            interaction_date = start_date_claims_interactions + timedelta(days=random.randint(0, 365))
            # Ensure interaction date is after client registration date
            client_reg_date = datetime.strptime(client['registrationDate'], "%Y-%m-%d")
            if interaction_date < client_reg_date:
                interaction_date = client_reg_date + timedelta(days=random.randint(1, 30))

            interaction_logs_data.append({
                "clientId": client['id'],
                "interactionType": random.choice(interaction_types),
                "timestamp": interaction_date.strftime("%Y-%m-%d %H:%M:%S")
            })

    # Generate Feedback
    for client in clients_data:
        if random.random() < 0.3: # 30% of clients provide feedback
            num_feedback = random.randint(1, 2)
            for _ in range(num_feedback):
                feedback_submitted_at = start_date_claims_interactions + timedelta(days=random.randint(0, 365))
                # Ensure feedback date is after client registration date
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

    # Generate Retention Actions
    for client in clients_data:
        if client['category'] in ["At-Risk", "Low-Engagement"] or random.random() < 0.1: # More likely for at-risk/low-engagement, small chance for others
            num_actions = random.randint(1, 2)
            for _ in range(num_actions):
                action_type = random.choice(retention_action_types)
                action_date = start_date_claims_interactions + timedelta(days=random.randint(0, 365))
                # Ensure action date is after client registration date
                client_reg_date = datetime.strptime(client['registrationDate'], "%Y-%m-%d")
                if action_date < client_reg_date:
                    action_date = client_reg_date + timedelta(days=random.randint(1, 30))

                # Outcome can be influenced by action type and client's churn prob
                if action_type == "Offer Discount" and client['churnProbability'] > 50 and random.random() < 0.7:
                    outcome = "Stayed" # Discount often works for high churn risk
                elif action_type == "Schedule Call" and client['churnProbability'] > 40 and random.random() < 0.6:
                    outcome = "Stayed"
                elif client['churnProbability'] > 70 and random.random() < 0.8:
                    outcome = "Churned" # High churn probability client likely churns
                else:
                    outcome = random.choice(retention_outcomes)

                retention_actions_data.append({
                    "clientId": client['id'],
                    "actionType": action_type,
                    "actionDate": action_date.strftime("%Y-%m-%d"),
                    "outcome": outcome
                })

    # Generate Client Claim Summary
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

# Generate the data
all_datasets = generate_data(num_clients=150, max_claims_per_client=4, max_interactions_per_client=60)

# Print or save the JSON data
# For brevity, I'll just print a snippet of each and indicate how to save.

print("--- Clients Data (Snippet) ---")
print(json.dumps(all_datasets["clients"][:3], indent=2))
print("\n--- Claims Data (Snippet) ---")
print(json.dumps(all_datasets["claims"][:3], indent=2))
print("\n--- Interaction Logs Data (Snippet) ---")
print(json.dumps(all_datasets["interaction_logs"][:3], indent=2))
print("\n--- Feedback Data (Snippet) ---")
print(json.dumps(all_datasets["feedback"][:3], indent=2))
print("\n--- Retention Actions Data (Snippet) ---")
print(json.dumps(all_datasets["retention_actions"][:3], indent=2))
print("\n--- Fraud Labels Data (Snippet) ---")
print(json.dumps(all_datasets["fraud_labels"][:3], indent=2))
print("\n--- Client Claim Summary Data (Snippet) ---")
print(json.dumps(all_datasets["client_claim_summary"][:3], indent=2))

# --- How to Save to JSON Files ---
# To save each dataset to its own JSON file, you can uncomment the following:

# with open('clients.json', 'w') as f:
#     json.dump(all_datasets["clients"], f, indent=2)
# with open('claims.json', 'w') as f:
#     json.dump(all_datasets["claims"], f, indent=2)
# with open('interaction_logs.json', 'w') as f:
#     json.dump(all_datasets["interaction_logs"], f, indent=2)
# with open('feedback.json', 'w') as f:
#     json.dump(all_datasets["feedback"], f, indent=2)
# with open('retention_actions.json', 'w') as f:
#     json.dump(all_datasets["retention_actions"], f, indent=2)
# with open('fraud_labels.json', 'w') as f:
#     json.dump(all_datasets["fraud_labels"], f, indent=2)
# with open('client_claim_summary.json', 'w') as f:
#     json.dump(all_datasets["client_claim_summary"], f, indent=2)

# print("\nAll datasets generated and ready to be saved as JSON files if uncommented.")