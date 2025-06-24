import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings

# Suppress InconsistentVersionWarning from scikit-learn if versions mismatch
# This doesn't fix the underlying issue but stops the warning spam in the console.
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# --- Load the pre-trained model ---
try:
    pipe = pickle.load(open('pipe.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: 'pipe.pkl' not found. Please ensure the trained model file is in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading the model (pipe.pkl): {e}")
    st.stop()

# --- Hardcoded lists for teams and cities ---
# These lists are derived directly from the processing in your ipl.ipynb notebook
# to ensure consistency with the model's training data.
teams = sorted([
    'Rajasthan Royals', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad',
    'Delhi Capitals', 'Chennai Super Kings', 'Gujarat Titans',
    'Lucknow Super Giants', 'Kolkata Knight Riders', 'Punjab Kings',
    'Mumbai Indians', 'Rising Pune Supergiants' # 'Kings XI Punjab' is replaced by 'Punjab Kings', 'Pune Warriors' by 'Rising Pune Supergiants'
])

cities = sorted([
    'Ahmedabad', 'Kolkata', 'Mumbai', 'Navi Mumbai', 'Pune', 'Dubai',
    'Sharjah', 'Abu Dhabi', 'Delhi', 'Chennai', 'Hyderabad',
    'Visakhapatnam', 'Chandigarh', 'Bengaluru', 'Jaipur', 'Indore',
    'Bangalore', 'Kanpur', 'Rajkot', 'Raipur', 'Ranchi', 'Cuttack',
    'Dharamsala', 'Nagpur', 'Johannesburg', 'Centurion', 'Durban',
    'Bloemfontein', 'Port Elizabeth', 'Kimberley', 'East London',
    'Cape Town'
])

# --- Streamlit App Interface ---
st.set_page_config(page_title="IPL Win Predictor")
st.title('IPL Win Predictor !! 🏏')

# Input fields for match details
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select the Batting Team', teams)

with col2:
    bowling_team = st.selectbox('Select the Bowling Team', teams)

selected_city = st.selectbox('Select Host City', cities)

target = st.number_input('Target Score', min_value=1, value=160,
                         help="The total score set by the first innings team.")

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Current Score', min_value=0, value=0,
                            help="Current runs scored by the batting team.")
with col4:
    overs = st.number_input('Overs Completed', min_value=0.0, max_value=19.5, step=0.1, value=0.0,
                            help="Overs completed (e.g., 5.1 for 5 overs and 1 ball). Max 19.5 for last ball of 20th over.")
with col5:
    wickets_fallen = st.number_input('Wickets Fallen', min_value=0, max_value=10, value=0,
                                     help="Number of wickets lost by the batting team.")

# Predict button
if st.button('Predict Win Probability'):
    # Input validation
    if batting_team == bowling_team:
        st.error("Batting and Bowling teams cannot be the same. Please select different teams.")
    else:
        # Calculate derived features for the model
        runs_left = target - score
        
        # Calculate balls left: 120 total balls - (overs completed * 6 + current ball in over)
        # Using int(overs * 10) % 10 for balls in current over (e.g., 5.1 -> 1, 5.0 -> 0)
        # And int(overs) for full overs
        balls_bowled = int(overs) * 6 + (int(overs * 10) % 10)
        balls_left = 120 - balls_bowled

        # 'wickets' feature in the model refers to wickets fallen (as per ipl.ipynb logic)
        wickets_for_model = wickets_fallen

        # Current Run Rate (CRR) calculation
        if balls_bowled == 0:
            crr = 0.0 # Avoid division by zero at the very start
        else:
            crr = (score * 6) / balls_bowled

        # Required Run Rate (RRR) calculation
        if balls_left <= 0: # Handle cases where all balls are bowled or target is met/exceeded
            rrr = 0.0
        else:
            rrr = (runs_left * 6) / balls_left
            if rrr < 0: # If target is already surpassed, RRR becomes irrelevant/negative
                rrr = 0.0

        # Create a Pandas DataFrame for the input, ensuring column names match the original training data
        # The ColumnTransformer inside 'pipe' expects these specific column names.
        input_df = pd.DataFrame({
            'BattingTeam': [batting_team],
            'BowlingTeam': [bowling_team],
            'City': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_for_model],
            'total_run_x': [target], # This corresponds to the target score in the notebook's final dataframe
            'crr': [crr],
            'rrr': [rrr]
        })

        try:
            # The 'pipe' object (which contains the ColumnTransformer and LogisticRegression)
            # will handle the one-hot encoding of categorical features internally.
            result = pipe.predict_proba(input_df)

            # Extract probabilities
            loss_prob = round(result[0][0] * 100)
            win_prob = round(result[0][1] * 100)

            st.markdown(f"## **<span style='color:green;'>{batting_team}</span> Win Probability: {win_prob}%**", unsafe_allow_html=True)
            st.markdown(f"## **<span style='color:red;'>{bowling_team}</span> Win Probability: {loss_prob}%**", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.warning("This might be due to an inconsistency between the loaded model and the input data structure. "
                       "Please ensure your 'pipe.pkl' was generated correctly in the `ipl.ipynb` notebook, "
                       "especially that the ColumnTransformer's steps are instantiated objects (e.g., `OneHotEncoder()`).")
