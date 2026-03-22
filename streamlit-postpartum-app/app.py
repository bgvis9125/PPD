import os
import csv
import streamlit as st
import pandas as pd
import joblib
import re
import plotly.express as px

# --- Session State Initialization ---
for key, value in {
    "logged_in": False,
    "user": "",
    "screening_ready": False
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

USERS_FILE = "users.csv"

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, newline="") as f:
            reader = csv.reader(f)
            return {rows[0]: rows[1] for rows in reader}
    else:
        # Create file with default users
        with open(USERS_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["testuser", "testpass"])
            writer.writerow(["student", "learn123"])
        return {"testuser": "testpass", "student": "learn123"}

def save_user(username, password):
    with open(USERS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([username, password])

users_db = load_users()

def login(username, password):
    if username in users_db and users_db[username] == password:
        st.session_state["logged_in"] = True
        st.session_state["user"] = username
        st.success(f"Logged in as {username}")
        st.rerun()
    else:
        st.error("Invalid username or password")

def register(username, password):
    if username in users_db:
        st.error("Username already exists. Please choose a different username.")
    else:
        save_user(username, password)
        users_db[username] = password
        st.success("Registration successful! Please login.")

def logout():
    st.session_state["logged_in"] = False
    st.session_state["user"] = ""
    st.session_state["screening_ready"] = False
    st.success("Logged out")

# --- Authentication UI ---
st.title("PsyPredict")

mode = st.radio("Select Mode:", ["Login", "Register"])

if mode == "Login":
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        login(username, password)
else:
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    if st.button("Register"):
        if new_password != confirm_password:
            st.error("Passwords do not match.")
        elif len(new_password) < 6:
            st.error("Password must be at least 6 characters.")
        else:
            register(new_username, new_password)

if st.session_state["logged_in"]:
    st.write(f"Welcome, **{st.session_state['user']}**!")
    if not st.session_state["screening_ready"]:
        if st.button("Continue to Screening"):
            st.session_state["screening_ready"] = True
            st.rerun()
        st.button("Logout", on_click=logout)
    else:
        st.button("Logout", on_click=logout)

        # --- Screening Section ---
        st.title("Answer a few questions to understand your postpartum depression risk and receive personalized guidance.")

        feature_columns = [
            'Age',
            'Feeling sad or Tearful',
            'Irritable towards baby & partner',
            'Trouble sleeping at night',
            'Problems concentrating or making decision',
            'Overeating or loss of appetite',
            'Feeling anxious',
            'Feeling of guilt',
            'Problems of bonding with baby',
            'Suicide attempt'
        ]

        def age_to_numeric(age_val):
            if isinstance(age_val, str) and '-' in age_val:
                try:
                    start, end = map(float, re.findall(r"[\d\.]+", age_val))
                    return (start + end) / 2
                except Exception:
                    return None
            try:
                return float(age_val)
            except Exception:
                return None

        form_inputs = {}
        for feature in feature_columns:
            if feature == 'Age':
                form_inputs[feature] = st.number_input(feature, min_value=16, max_value=55, step=1)
            else:
                form_inputs[feature] = st.selectbox(
                    feature,
                    ["yes", "no", "sometimes", "two or more days a week", "not interested to say", "maybe"]
                )

        if st.button("Predict Risk"):
            model = joblib.load('models/Ensemble.pkl')
            input_df = pd.DataFrame([form_inputs], columns=feature_columns)

            response_mapping = {
                'yes': 2, 'sometimes': 1, 'no': 0, 'maybe': 0.5,
                'two or more days a week': 2, 'not interested to say': -1
            }

            for col in feature_columns:
                if col != 'Age':
                    input_df[col] = input_df[col].map(response_mapping).fillna(0)
            input_df["Age"] = input_df["Age"].apply(age_to_numeric).astype(float)

            prediction = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]

            st.markdown(f"### Predicted Risk: {'High' if prediction == 1 else 'Low'}")
            st.markdown(f"### Probability Score: {prob:.2f}")
            st.success("Prediction complete!")

            # Advice Section
            if prediction == 1:
                st.markdown("### Personalized Mental Health Advice")
                st.warning("You have been identified as **high risk** for postpartum depression.")
                st.write("""
                Postpartum depression is **common and treatable**. You're not alone—many mothers experience this.
                - Reach out to a **mental health professional** or counselor as soon as possible.
                - Stay connected with family and friends; isolation worsens symptoms.
                - Focus on **self-care**—small daily walks, hydration, and rest matter.
                - You are not at fault; recovery is possible with the right help.
                """)
                st.markdown("### Recommended Resources")
                st.markdown("- [National Mental Health Helpline (India)](https://telemanas.mohfw.gov.in/)")
                st.markdown("- [CDC: Postpartum Depression Guide](https://www.ncbi.nlm.nih.gov/books/NBK519070/)")
                st.markdown("- [Find a Mental Health Clinic](https://www.mentalhealth.gov/get-help)")
                st.info("If you ever experience intrusive thoughts or suicidal ideation, seek immediate help.")
            else:
                st.markdown("### Emotional Well-being Guidance")
                st.info("Your risk appears **low**, which is wonderful.")
                st.write("""
                Continue focusing on maintaining a supportive environment and healthy routine:
                - **Talk openly** about how you feel—it reduces emotional stress.
                - Maintain **balanced sleep and nutrition**.
                - Do small activities that bring joy—music, reading, or sharing time with loved ones.
                - Remember, it’s okay to ask for help if feelings change in the future.
                """)
                st.markdown("### Helpful Wellness Resources")
                st.markdown("- [WHO Maternal Mental Health Info](https://www.who.int/news-room/fact-sheets/detail/maternal-mental-health)")
                st.markdown("- [Postpartum Support International](https://www.postpartum.net/get-help/)")
                st.markdown("- [UNICEF - Mental Health for Parents](https://www.unicef.org/parenting/health/mental-health-for-parents)")


            # Personal Symptom Visualization
            st.subheader("Your Symptom Influence Visualization")
            viz_df = input_df.copy()
            viz_df = viz_df.rename(columns=lambda x: x.replace('_', ' '))
            response_summary = {k: form_inputs[k] for k in feature_columns if k != 'Age'}

            viz_df_display = pd.DataFrame({
                "Symptom": list(response_summary.keys()),
                "Response": list(response_summary.values()),
                "Severity": [viz_df.iloc[0][c] if c in viz_df.columns else 0 for c in response_summary.keys()]
            })

            fig = px.bar(
                viz_df_display.sort_values("Severity", ascending=False),
                x="Severity",
                y="Symptom",
                color="Severity",
                text="Response",
                orientation="h",
                color_continuous_scale=["#5CB85C", "#FFD700", "#FF5349"],
                title="Your Symptom Contribution to Depression Risk"
            )
            fig.update_layout(title_font_size=18, xaxis_title="Risk Contribution Level", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

            st.info("This chart shows how your symptom responses contribute to your depression risk score.")

        st.markdown("Running away from your problems is a race you’ll never win. Instead, reach out for help and try to confront them.")
