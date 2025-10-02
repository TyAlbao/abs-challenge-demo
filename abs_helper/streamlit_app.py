# streamlit_app.py
import streamlit as st
from abs_helper import ABSInterface
import matplotlib.pyplot as plt

# --- Page Setup ---
st.set_page_config(page_title="ABS Challenge Demo", layout="centered")
st.title("⚾ Automated Ball-Strike (ABS) Challenge Demo ⚙️")
st.caption("Note: All predictions are based on data from the 2025 MLB season.")

# --- Load Interface ---
@st.cache_resource
def load_interface():
    return ABSInterface(
        "saved_models/mcm_calib.pkl",
        "saved_models/wpm_calib.pkl",
        "saved_dcts/ump_name_dct.json",
        "saved_dcts/ump_zone_acc.joblib"
    )

abs_ui = load_interface()

# -----------------------------
# STEP 1: Incorrect Call Probability
# -----------------------------
st.header("Step 1: Estimate Incorrect Call Probability 🤔")

umpire = st.selectbox(
    "Select Umpire",
    options=list(abs_ui.helper.ump_name_dct.keys()),
    format_func=lambda x: abs_ui.helper.ump_name_dct[x]
)

# Pitch type mapping
pitch_type_map = {
    "CH": "Changeup",
    "CU": "Curveball",
    "FC": "Cutter",
    "EP": "Eephus",
    "FO": "Forkball",
    "FF": "Four-Seam Fastball",
    "KN": "Knuckleball",
    "KC": "Knuckle-curve",
    "SC": "Screwball",
    "SI": "Sinker",
    "SL": "Slider",
    "SV": "Slurve",
    "FS": "Splitter",
    "ST": "Sweeper",
    # Extras in your list
    "FA": "Fastball (unspecified)",
    "CS": "Slow Curve"
}

pitch_type = st.selectbox(
    "Pitch Type",
    options=list(pitch_type_map.keys()),
    format_func=lambda x: pitch_type_map[x]
)

pitch_loc = st.selectbox("Pitch Location", ["high_away", "high_middle", "high_inside",
                                            "mid_away", "mid_middle", "mid_inside",
                                            "low_away", "low_middle", "low_inside"])
call = st.radio("Umpire Call", ["strike", "ball"])
inning = st.number_input("Inning", min_value=1, max_value=12, value=7)
outs = st.number_input("Outs", min_value=0, max_value=2, value=1)
balls = st.number_input("Balls", min_value=0, max_value=3, value=2)
strikes = st.number_input("Strikes", min_value=0, max_value=2, value=1)
pitcher_rhand = st.checkbox("Pitcher Right-Handed?", value=True)
batter_rhand = st.checkbox("Batter Right-Handed?", value=False)

mcm_case = {
    "ump_id": umpire,
    "pitch_type": pitch_type,
    "pitch_loc": pitch_loc,
    "inning": inning,
    "call": call,
    "outs_when_up": outs,
    "pitcher_is_rhand": pitcher_rhand,
    "balls": balls,
    "strikes": strikes,
    "batter_is_rhand": batter_rhand,
}

if st.button("Compute Incorrect Call Probability"):
    prob = abs_ui.predict_incorrect_call_prob(mcm_case)
    st.session_state.incorrect_call_prob = prob

# Always show stored probability if it exists
if "incorrect_call_prob" in st.session_state:
    st.success(f"Incorrect Call Probability: **{st.session_state.incorrect_call_prob}%**")

# -----------------------------
# STEP 2: Challenge Decision
# -----------------------------
if "incorrect_call_prob" in st.session_state:
    st.header("Step 2: Challenge Decision? 🤷‍♂️")
    st.markdown(
        "ℹ️ **Note:** The features you entered in Step 1 "
        "(inning, outs, balls, strikes) "
        "are automatically carried over into this step. "
        "Here, you’re adding the broader **game context** "
        "needed to decide whether to challenge."
    )

    inning_tb = st.selectbox("Top or Bottom", ["Top", "Bot"])
    base_state = st.selectbox("Base State (binary runners)", 
                              ["000","001","010","011","100","101","110","111"])
    team_diff = st.number_input("Batting Team Score Diff", -10, 10, -1)

    wpm_case = {
        "inning_topbot": inning_tb,
        "base_state": base_state,
        "team_bat_score_diff": team_diff
    }

    if st.button("Compute ΔWE & Challenge Decision"):
        result = abs_ui.predict_dwe(wpm_case)
        st.session_state.delta_we_result = result["delta_we"]
        st.session_state.challenge_result = result["challenge"]

    # Always show results if they exist in session_state
    if "delta_we_result" in st.session_state and "challenge_result" in st.session_state:
        st.metric("Delta Win Expectancy", f"{st.session_state.delta_we_result:.5f}")
        st.metric("Challenge?", "✅ Yes" if st.session_state.challenge_result else "❌ No")


# -----------------------------
# STEP 3: Umpire Heatmap
# -----------------------------
if "delta_we_result" in st.session_state and "challenge_result" in st.session_state:
    st.header("Step 3: Umpire Zone Accuracy Heatmap 🔥")

    flip_perspective = st.checkbox("Show from Catcher's Perspective?", value=False)

    fig, ax = abs_ui.helper.show_ump_heatmap(
        ump_id=umpire,
        catcher_perspective=flip_perspective
    )
    st.pyplot(fig)

