import pandas as pd
import numpy as np


# --- Base runner maps ---
WALK_NEW_BR_MAP = {
    '000':'100','001':'101','010':'110','011':'111',
    '100':'110','101':'111','110':'111','111':'111'
}
WALK_RUN_ADDED_MAP = {
    '000':0,'001':0,'010':0,'011':0,
    '100':0,'101':0,'110':0,'111':1
}

# --- Helpers ---
def handle_strikes(df, call_mask, outs, s, inn, diff):
    """Update DataFrame for called strikes (standing or overturned)."""
    inning_end = call_mask & (outs == 2) & (s == 2)
    atbat_end  = call_mask & (outs <= 1) & (s == 2)
    non_term   = call_mask & (s <= 1)
    game_end   = inning_end & (inn >= 9) & (diff < 0)

    # Inning-ending
    df.loc[inning_end, ['balls','strikes','outs_when_up']] = 0
    df.loc[inning_end, 'inning_topbot'] = np.where(
        df.loc[inning_end, 'inning_topbot'] == 'Top', 'Bot', 'Top'
    )
    df.loc[inning_end, 'inning'] = np.where(
        df.loc[inning_end, 'inning_topbot'] == 'Top',
        df.loc[inning_end, 'inning'] + 1,
        df.loc[inning_end, 'inning']
    )
    # Assigning flag for when Top/Bot inning switches to account
    # for change in win probability perspective
    df['inning_topbot_switch'] = False
    df.loc[inning_end & ~game_end, 'inning_topbot_switch'] = True
    df.loc[inning_end, 'base_state'] = np.where(
        df.loc[inning_end, 'inning'] <= 9, '000', '010'
    )
    df.loc[inning_end, 'team_bat_score_diff'] *= -1


    # At-bat-ending
    df.loc[atbat_end, ['balls','strikes']] = 0
    df.loc[atbat_end, 'outs_when_up'] += 1

    # Non-terminal
    df.loc[non_term, 'strikes'] += 1
    return df, game_end


def handle_walks(df, call_mask, b, inn, tb, diff, bs):
    """Update DataFrame for walks (standing or overturned)."""
    walk_mask    = call_mask & (b == 3)
    non_term     = call_mask & (b <= 2)
    walkoff_mask = call_mask & (b == 3) & (inn >= 9) & \
                   (tb == 'Bot') & (diff == 0) & (bs == '111')

    df.loc[walk_mask, ['balls','strikes']] = 0
    df.loc[walk_mask, 'base_state'] = df.loc[walk_mask, 'base_state'].map(WALK_NEW_BR_MAP)
    df.loc[walk_mask, 'team_bat_score_diff'] += df.loc[walk_mask, 'base_state'].map(WALK_RUN_ADDED_MAP)

    df.loc[non_term, 'balls'] += 1
    return df, walkoff_mask


# --- Main pipeline ---
def process_states(merged, feature_cols_wpm, cs, cb, outs, s, inn, diff, b, tb, bs):
    # start fresh
    S = merged[feature_cols_wpm].copy()
    O = merged[feature_cols_wpm].copy()

    # handle strikes
    S, game_end_S = handle_strikes(S, cs, outs, s, inn, diff)
    O, game_end_O = handle_strikes(O, cb, outs, s, inn, diff)

    # handle walks
    S, walkoff_S = handle_walks(S, cb, b, inn, tb, diff, bs)
    O, walkoff_O = handle_walks(O, cs, b, inn, tb, diff, bs)

    # carve-outs
    S_l = S.loc[game_end_S]
    O_l = O.loc[game_end_O]
    S_w = S.loc[walkoff_S]
    O_w = O.loc[walkoff_O]

    # drop carved rows
    S = S.drop(S_l.index.union(S_w.index))
    O = O.drop(O_l.index.union(O_w.index))

    return S, O, S_l, O_l, S_w, O_w

def add_win_prob_future_states(df, wp_model, feature_cols_wpm):
    if df.empty:
        return df
    # Predict raw win probability from features
    df['wp_raw'] = wp_model.predict_proba(df[feature_cols_wpm])[:, 1]

    # Correct for inning flips
    df['win_prob'] = np.where(df['inning_topbot_switch'],
                              1 - df['wp_raw'],  # complement if flipped
                              df['wp_raw'])

    # Drop helper col
    df.drop(columns='inning_topbot_switch', inplace=True)
    return df

