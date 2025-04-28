import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Load Models and Lists ---

success_model = joblib.load('movie_success_model.pkl')
revenue_model = joblib.load('movie_revenue_model.pkl')
genre_binarizer_success = joblib.load('genre_binarizer.pkl')
genre_binarizer_revenue = joblib.load('genre_binarizer_revenue.pkl')
monthly_revenue = pd.read_pickle('monthly_revenue_trends.pkl')
top_actors = joblib.load('actor_list.pkl')
top_directors = joblib.load('director_list.pkl')

# --- Header ---

st.title("🎬 Movie Success and Revenue Predictor")

st.markdown("""
Welcome! This tool predicts whether a movie will be a **box office success** and estimates its **total box office revenue**.

**Definitions:**
- **Success** = A movie earns at least **2× its production budget** in box office revenue.
- **Suggestions for improvement** are shown only if the predicted success probability is **below 60%**.
- **Actors:** Select up to 3 major top-billed actors involved. If your movie has no famous stars, select **"None"**.
- **Director:** Select the director attached. If the director is not among the top known directors, select **"None"**.
- **Disclaimer:** These are predictions based on historical data and do not guarantee real-world performance.
""")

st.markdown("---")

# --- Movie Input Form ---

st.header("📋 Enter Movie Information")

budget = st.number_input("Budget ($)", min_value=1000000, step=500000, value=30000000)

genres_input = st.multiselect(
    "Select up to 3 Genres",
    options=sorted(genre_binarizer_success.classes_),
    max_selections=3,
    help="Select the genres that best fit your movie (up to 3)."
)

actors_dropdown = ['None'] + sorted(top_actors)
actors_input = st.multiselect(
    "Select up to 3 Top Billed Actors",
    options=actors_dropdown,
    max_selections=3,
    help="Select up to 3 major actors appearing in the movie, or 'None' if no major actors."
)

director_dropdown = ['None'] + sorted(top_directors)
director_input = st.selectbox(
    "Select a Director",
    options=director_dropdown,
    help="Select the movie's director, or 'None' if the director is not well-known."
)

release_month = st.selectbox(
    "Planned Release Month",
    list(range(1, 13)),
    help="Choose the intended release month (1 = January, 12 = December)."
)

# --- Prediction Button ---

if st.button("Predict Movie Performance"):
    st.markdown("---")
    st.subheader("🎯 Prediction Results")
    
    # --- Build Features for Prediction ---

    ## For Success Model
    input_success = pd.DataFrame()
    input_success['budget'] = [budget]
    input_success['release_month_strength'] = [monthly_revenue.get(release_month, monthly_revenue.mean())]

    genres_encoded_success = genre_binarizer_success.transform([genres_input])
    genres_df_success = pd.DataFrame(genres_encoded_success, columns=[f'genre_{g}' for g in genre_binarizer_success.classes_])
    input_success = pd.concat([input_success.reset_index(drop=True), genres_df_success.reset_index(drop=True)], axis=1)

    for col in genre_binarizer_success.classes_:
        if f'genre_{col}' not in input_success.columns:
            input_success[f'genre_{col}'] = 0

    if 'None' in actors_input or not actors_input:
        actor_count = 0
    else:
        actor_count = sum(1 for actor in actors_input if actor in top_actors)
    input_success['top_actor_count'] = [actor_count]

    if director_input == 'None':
        known_director = 0
    else:
        known_director = int(director_input in top_directors)
    input_success['known_director'] = [known_director]

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

    input_revenue['top_actor_count'] = [actor_count]
    input_revenue['known_director'] = [known_director]

    # --- Make Predictions ---

    ## Success
    success_pred = success_model.predict(input_success)[0]
    success_prob = success_model.predict_proba(input_success)[0][1]

    ## Revenue
    predicted_revenue = revenue_model.predict(input_revenue)[0]

    # --- Build Specific Outcome-Based Reasoning ---

    reasons = []

    # Calculate month strength
    month_strength = monthly_revenue.get(release_month, monthly_revenue.mean())

    if success_pred == 1:
        # Only list strong points
        if budget >= 50000000:
            reasons.append(f"💵 Budget of ${budget:,.0f} enables high production value and aggressive marketing.")
        if month_strength >= monthly_revenue.mean():
            reasons.append(f"📅 Releasing in month {release_month} captures peak seasonal audience turnout.")
        if actor_count >= 2:
            reasons.append(f"🎭 {actor_count} major top-billed actors attached boost marketing reach and opening buzz.")
        if known_director == 1:
            reasons.append("🎬 Directed by a top-recognized director, enhancing credibility and media attention.")
    else:
        # Only list weak points
        if budget < 50000000:
            reasons.append(f"💵 Budget of ${budget:,.0f} may limit production scale and promotional power.")
        if month_strength < monthly_revenue.mean():
            reasons.append(f"📅 Month {release_month} tends to have weaker box office performance compared to peak seasons.")
        if actor_count < 2:
            reasons.append(f"🎭 Having {actor_count} major actors may not provide enough audience draw.")
        if known_director == 0:
            reasons.append("🎬 Lack of a highly recognized director may limit critical acclaim and promotional leverage.")

    if not reasons:
    if success_pred == 1:
        reasons.append("✅ Overall, the combination of budget, release timing, cast, and director suggests a strong likelihood of success.")
    else:
        reasons.append("❌ Overall, multiple factors combined to suggest a lower chance of box office success.")

    # --- Display Results ---

    if success_pred == 1:
        st.success(f"✅ The movie is predicted to be a SUCCESS!")
    else:
        st.error(f"❌ The movie is predicted to FAIL.")

    st.write(f"**Predicted Success Probability:** {success_prob:.2%}")
    st.write(f"**Estimated Box Office Revenue:** ${predicted_revenue:,.0f}")

    st.markdown("### 📋 Reasoning Behind Prediction")
    for reason in reasons:
        st.write("-", reason)

    # --- Improvement Suggestions (only if probability < 60%) ---

    if success_prob < 0.6:
        st.markdown("---")
        st.subheader("📈 Suggestions to Improve Success Chances")

        if budget < 50000000:
            st.write("- 💵 **Consider increasing your budget to at least $50M** to better compete.")

        if month_strength < monthly_revenue.mean():
            best_month = monthly_revenue.idxmax()
            st.write(f"- 📅 **Consider releasing during a stronger month like {best_month} to maximize audience turnout.**")

        if actor_count < 2:
            st.write("- 🎭 **Adding more top-billed actors could significantly improve visibility and buzz.**")

        if known_director == 0:
            st.write("- 🎬 **Working with a more recognized director could enhance credibility and media attention.**")

    st.markdown("---")
