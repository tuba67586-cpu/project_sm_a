import pickle
import pandas as pd
import streamlit as st

# -------------------------
# Load model + columns safely
# -------------------------
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('columns.pkl', 'rb') as f:
        model_columns = pickle.load(f)

except FileNotFoundError:
    st.error("❌ model.pkl or columns.pkl not found. Please train and save the model first.")
    st.stop()

# -------------------------
# UI
# -------------------------
st.title("📊 Social Media Performance Predictor")

follower_count = st.number_input("Follower Count", value=10000)
reach = st.number_input("Reach", value=5000)
impressions = st.number_input("Impressions", value=8000)
post_hour = st.slider("Post Hour", 0, 23, 12)
hashtags_count = st.number_input("Hashtags Count", value=5)
caption_length = st.number_input("Caption Length", value=100)

account_type = st.selectbox("Account Type", ["brand", "creator"])
media_type = st.selectbox("Media Type", ["image", "video", "reel", "carousel"])
content_category = st.selectbox("Content Category", ["Technology", "Fitness", "Beauty", "Travel"])
traffic_source = st.selectbox("Traffic Source", ["Explore", "Home Feed", "Hashtags", "Profile"])
day_of_week = st.selectbox(
    "Day of Week",
    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
)

# -------------------------
# Prediction
# -------------------------
if st.button("Predict"):

    input_dict = {
        'follower_count': follower_count,
        'reach': reach,
        'impressions': impressions,
        'post_hour': post_hour,
        'hashtags_count': hashtags_count,
        'caption_length': caption_length,
        'account_type': account_type,
        'media_type': media_type,
        'content_category': content_category,
        'traffic_source': traffic_source,
        'day_of_week': day_of_week
    }

    input_df = pd.DataFrame([input_dict])

    # Encode input
    input_encoded = pd.get_dummies(input_df)

    # Match training columns
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Predict
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded).max()

    # Output
    if prediction == "viral":
        st.success("🔥 Viral Post Expected!")
    elif prediction == "high":
        st.info("📈 High Performance")
    elif prediction == "medium":
        st.warning("⚖️ Medium Performance")
    else:
        st.error("📉 Low Performance")

    st.subheader(f"Confidence: {probability:.2f}")
if st.button("Predict", key="predict_button"):
    probability = 0.85  # example
    st.progress(float(probability))