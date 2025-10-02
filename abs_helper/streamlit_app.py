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
    "ST": "Sweeper"
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
    # base_state = st.selectbox("Base State (binary runners)", 
    #                           ["000","001","010","011","100","101","110","111"])
        
            # --- Helpers for base-state <-> flags ---
    def state_to_flags(s: str):
        # s = "XYZ" where X=1st, Y=2nd, Z=3rd (your current encoding)
        return {
            "b1": s[0] == "1",
            "b2": s[1] == "1",
            "b3": s[2] == "1",
        }

    def flags_to_state(b1: bool, b2: bool, b3: bool) -> str:
        return f"{int(b1)}{int(b2)}{int(b3)}"

    # --- Initialize session state syncing ---
    if "base_state" not in st.session_state:
        st.session_state.base_state = "000"

    # (A) Binary selector (still available for power users)
    base_state_choice = st.selectbox(
        "Base State (binary runners)",
        ["000", "001", "010", "011", "100", "101", "110", "111"],
        index=["000", "001", "010", "011", "100", "101", "110", "111"].index(st.session_state.base_state),
        key="base_state_select",
    )

    # If user changed the dropdown, sync flags
    if base_state_choice != st.session_state.base_state:
        st.session_state.base_state = base_state_choice

    # --- Build flags from current state ---
    flags = state_to_flags(st.session_state.base_state)
    b1, b2, b3 = flags["b1"], flags["b2"], flags["b3"]

    st.markdown("#### Or click/toggle on the diamond")

    # --- Draw diamond with matplotlib (visual only) ---
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Circle

    def draw_bases_fig(b1: bool, b2: bool, b3: bool):
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        ax.set_aspect('equal')
        ax.axis('off')

        # Diamond (square rotated 45°)
        diamond = Polygon([[0,1], [1,2], [2,1], [1,0]], closed=True, fill=False, linewidth=2)
        ax.add_patch(diamond)

        # Base coordinates (approx)
        pos_home = (1, 0)
        pos_1st  = (2, 1)
        pos_2nd  = (1, 2)
        pos_3rd  = (0, 1)

        # Draw bases as circles; fill if occupied
        def base_circle(xy, filled):
            c = Circle(xy, 0.12, fill=filled, linewidth=2)
            ax.add_patch(c)

        base_circle(pos_1st,  b1)
        base_circle(pos_2nd,  b2)
        base_circle(pos_3rd,  b3)
        base_circle(pos_home, False)  # home for reference

        ax.set_xlim(-0.2, 2.2)
        ax.set_ylim(-0.2, 2.2)
        return fig

    fig = draw_bases_fig(b1, b2, b3)
    st.pyplot(fig)

    # --- Clickable toggles mapped to bases (sync back to state) ---
    top = st.columns([1,2,1])
    with top[1]:
        b2_new = st.toggle("Runner on 2nd", value=b2, key="toggle_b2")

    mid = st.columns([1,1,1])
    with mid[0]:
        b3_new = st.toggle("Runner on 3rd", value=b3, key="toggle_b3")
    with mid[2]:
        b1_new = st.toggle("Runner on 1st", value=b1, key="toggle_b1")

    # If any toggle changed, update session_state.base_state and redraw
    new_state = flags_to_state(b1_new, b2_new, b3_new)
    if new_state != st.session_state.base_state:
        st.session_state.base_state = new_state
        # (Optional) live feedback badge
        st.caption(f"Selected base state: **{st.session_state.base_state}**")

    # Use the authoritative session_state value when building wpm_case
    base_state = st.session_state.base_state



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

