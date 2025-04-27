import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Load Models and Encoders ---

success_model = joblib.load('movie_success_model.pkl')
revenue_model = joblib.load('movie_revenue_model.pkl')
genre_binarizer_success = joblib.load('genre_binarizer.pkl')
genre_binarizer_revenue = joblib.load('genre_binarizer_revenue.pkl')
monthly_revenue = pd.read_pickle('monthly_revenue_trends.pkl')

# --- Header ---

st.title("🎬 Movie Success and Revenue Predictor")

st.markdown("""
Welcome! This tool predicts whether a movie will be a **box office success** and estimates its **total box office revenue**.

**Definitions:**
- **Success** = A movie earns at least **2× its production budget** in box office revenue.
- **Suggestions for improvement** are shown only if the predicted success probability is **below 60%**.
- **Disclaimer:** These are predictions based on historical movie data. Real-world results depend on many unpredictable factors.
""")

st.markdown("---")

# --- Movie Input Form ---

st.header("📋 Enter Movie Information")

budget = st.number_input("Budget ($)", min_value=1000000, step=500000, value=30000000)

genres_input = st.multiselect(
    "Select up to 3 Genres",
    options=sorted(genre_binarizer_success.classes_),
    max_selections=3
)

cast_count = st.slider("Top Billed Cast Count", min_value=0, max_value=3, value=2)

release_month = st.selectbox("Planned Release Month", list(range(1, 13)))

# --- Prediction Button ---

if st.button("Predict Movie Performance"):
    st.markdown("---")
    st.subheader("🎯 Prediction Results")
    
    # --- Build Features ---

    ## For Success Model
    input_success = pd.DataFrame()
    input_success['budget'] = [budget]
    input_success['release_month_strength'] = [monthly_revenue.get(release_month, monthly_revenue.mean())]

    genres_encoded_success = genre_binarizer_success.transform([genres_input])
    genres_df_success = pd.DataFrame(genres_encoded_success, columns=[f'genre_{g}' for g in genre_binarizer_success.classes_])
    input_success = pd.concat([input_success.reset_index(drop=True), genres_df_success.reset_index(drop=True)], axis=1)

    # Fill missing genres
    for col in genre_binarizer_success.classes_:
        if f'genre_{col}' not in input_success.columns:
            input_success[f'genre_{col}'] = 0

    input_success['cast_count'] = [cast_count]

    ## For Revenue Model
    input_revenue = pd.DataFrame()
    input_revenue['budget'] = [budget]
    input_revenue['release_month_strength'] = [monthly_revenue.get(release_month, monthly_revenue.mean())]

    genres_encoded_revenue = genre_binarizer_revenue.transform([genres_input])
    genres_df_revenue = pd.DataFrame(genres_encoded_revenue, columns=[f'genre_{g}' for g in genre_binarizer_revenue.classes_])
    input_revenue = pd.concat([input_revenue.reset_index(drop=True), genres_df_revenue.reset_index(drop=True)], axis=1)

    for col in genre_binarizer_revenue.classes_:
        if f'genre_{col}' not in input_revenue.columns:
            input_revenue[f'genre_{col}'] = 0

    input_revenue['cast_count'] = [cast_count]

    # --- Make Predictions ---

    ## Success
    success_pred = success_model.predict(input_success)[0]
    success_prob = success_model.predict_proba(input_success)[0][1]

    ## Revenue
    predicted_revenue = revenue_model.predict(input_revenue)[0]

    # --- Display Results ---

    if success_pred == 1:
        st.success(f"✅ The movie is predicted to be a SUCCESS!")
    else:
        st.error(f"❌ The movie is predicted to FAIL.")

    st.write(f"**Predicted Success Probability:** {success_prob:.2%}")
    st.write(f"**Estimated Box Office Revenue:** ${predicted_revenue:,.0f}")

    # --- Improvement Suggestions (if needed) ---

    if success_prob < 0.6:
        st.markdown("---")
        st.subheader("📈 Suggestions to Improve Success Chances")

        if budget < 50000000:
            st.write("- 💵 **Increase the budget to $50M+** to be more competitive in the market.")

        if monthly_revenue.get(release_month, monthly_revenue.mean()) < monthly_revenue.mean():
            best_month = monthly_revenue.idxmax()
            st.write(f"- 📅 **Consider releasing during a stronger month, like {best_month}** (historically higher box office).")

        if cast_count < 2:
            st.write("- 🎭 **Consider adding more top-billed well-known actors.**")

    st.markdown("---")
