"""
abs_helper.py
-----------------
Main helper module for the Automated Ball-Strike (ABS) Challenge system.

Responsibilities:
- Loading and managing models and umpire data
- Predicting incorrect call probability (MCM)
- Predicting win probability and ΔWE (WPM)
- Providing visualization utilities (umpire zone heatmaps)
"""

# Standard library
import json
import requests

# Third-party
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import randint, loguniform, uniform

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    RandomizedSearchCV,
    TimeSeriesSplit,
    GroupKFold,
    train_test_split
)
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    make_scorer,
    f1_score,
    confusion_matrix,
    classification_report,
    brier_score_loss,
    average_precision_score
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Local modules
from xgb_clf_model import train_xgb, calibrate_model
from future_states import process_states, add_win_prob_future_states


# Plate is 17 inches wide → half-width in feet
PLATE_HALF_WIDTH = (17 / 2) / 12

# Ball diameter ~2.86 inches → radius in feet
BALL_RADIUS = (2.86 / 2) / 12


class ABSHelper:
    """
    Core helper class for the Automated Ball-Strike (ABS) Challenge system.

    Handles:
    - Model and dictionary loading
    - DataFrame preparation for predictions
    - Adding incorrect call probability and win probability
    - Creating future states and computing ΔWE
    - Generating umpire zone accuracy heatmaps
    """

    universal_lambda = 0.0058652361809045225

    cat_cols_mcm = ["ump_id", "pitch_type", "pitch_loc", "inning", 'call']
    num_cols_mcm = ["outs_when_up", "pitcher_is_rhand", 'balls', 'strikes', "batter_is_rhand"]
    feature_cols_mcm = cat_cols_mcm + num_cols_mcm
    
    cat_cols_wpm = ['inning_topbot', 'base_state']
    num_cols_wpm = ['inning', 'outs_when_up', 'balls', 'strikes', 'team_bat_score_diff']
    feature_cols_wpm = cat_cols_wpm + num_cols_wpm


    def __init__(self):
        
        self.raw_df = pd.DataFrame()
        self.df = pd.DataFrame()
        self.ump_name_dct = dict()
        self.ump_game_dct = dict()

    def load_raw_data(self, raw_df):
        
        # only select regular season games        
        self.raw_df = raw_df[raw_df['game_type']=='R']
        self.raw_df = self.raw_df.reset_index(drop=True)

    @staticmethod
    def _find_hp_ump_idx(officals_lst):
        for i in range(len(officals_lst)):
            if 'Home Plate' in officals_lst[i].values():
                return i
    
    @staticmethod
    def _get_hp_ump_dct(games):
        game_ump_map = dict()

        for game in games:
            url = f'https://statsapi.mlb.com/api/v1/game/{game}/boxscore'
            r = requests.get(url, timeout=10).json()
            hp_ump_idx = 0
            misidx=0
            officials_values = r['officials'][hp_ump_idx].values()
            if 'Home Plate' not in officials_values:
                hp_ump_idx = ABSHelper._find_hp_ump_idx(r['officials'])
                misidx+=1

            hp_ump = r['officials'][hp_ump_idx]
            hp_ump_id = hp_ump['official']['id']
            hp_ump_name = hp_ump['official']['fullName']

            game_ump_map[int(game)] = {'id': hp_ump_id, 'name': hp_ump_name}
        
        return game_ump_map
    
    @staticmethod
    def _ball_or_strike(row):
        if -PLATE_HALF_WIDTH - BALL_RADIUS <= row['plate_x'] <= PLATE_HALF_WIDTH + BALL_RADIUS:
            szt_lim = row['sz_top'] + BALL_RADIUS
            szb_lim = row['sz_bot'] - BALL_RADIUS

            if szb_lim <= row['plate_z'] <= szt_lim:
                return 'strike'
        
        return 'ball'
    
    def preprocess_data(self):

        # reset index
        df = self.raw_df.reset_index(drop=True)

        # only select pitches where umpire made a choice
        df = df[df['description'].isin(['called_strike','ball'])].dropna(subset=['pitch_type','zone'])

        games = df['game_pk'].unique()

        # create game ID: umpire ID map
        self.ump_game_dct = self._get_hp_ump_dct(games)
        ugd = self.ump_game_dct

        # create umpire ID: Name map 
        self.ump_name_dct = {ugd[game]['id']: ugd[game]['name'] for game in ugd}
        und = self.ump_name_dct

        # assign columns for umpire IDs and names
        df['ump_id'] = df['game_pk'].apply(lambda x: ugd[x]['id'])
        df['ump_name'] = df['game_pk'].apply(lambda x: ugd[x]['name'])

        # select only needed columns
        df = df[[
            'game_date', 'ump_name', 'player_name', 'ump_id', 'pitch_type', 'p_throws', 'inning', 'balls', 'strikes',
            'outs_when_up', 'pitcher','batter', 'stand', 'plate_x','plate_z','sz_top', 'sz_bot', 'zone', 'description',
        ]].copy()
        
        # adjusting data types
        df['description'] = df['description'].map({'called_strike':'strike', 'ball':'ball'})
        df.loc[:, ['p_throws','stand']] = df[['p_throws','stand']] == 'R'

        # changing data types
        df['game_date'] = pd.to_datetime(df['game_date'])
        df['zone'] = df['zone'].astype(int)
        df['balls'] = df['balls'].astype(int)
        df['strikes'] = df['strikes'].astype(int)
        df['inning'] = df['inning'].astype(int)
        df['outs_when_up'] = df['outs_when_up'].astype(int)
        df['sz_top'] = df['sz_top'].astype(float)
        df['sz_bot'] = df['sz_bot'].astype(float)

        # renaming columns
        df = df.rename(columns={'description':'call', 'p_throws':'pitcher_is_rhand', 'stand':'batter_is_rhand'})

        # determining if pitch was a strike or ball
        df['real'] = df.apply(ABSHelper._ball_or_strike, axis=1)
        df['correct'] = df['call'] == df['real']
        df['incorrect'] = ~df['correct']

        z_span = (df['sz_top'] - df['sz_bot']).clip(lower=1e-6)
        z_norm = (df['plate_z'] - df['sz_bot']) / z_span

        sign = np.where(df['batter_is_rhand'], 1.0, -1.0)
        x_signed = df['plate_x'] * sign

        df['pitch_loc_vert'] = pd.cut(z_norm, bins=[-float('inf'), 0.33, 0.66, float('inf')],
                                labels=['low','mid','high']).astype('category')

        df['pitch_loc_horiz'] = pd.cut(x_signed, bins=[-float('inf'), -0.25, 0.25, float('inf')],
                                labels=['away','middle','inside']).astype('category')

        df['pitch_loc'] = (df['pitch_loc_vert'].astype(str) + "_" + df['pitch_loc_horiz'].astype(str)).astype('category')

        self.df = df

    @staticmethod
    def save_dct(dct, path):
        with open(path, 'w') as f:
            json.dump(dct, f)

    @staticmethod
    def load_dct(path, int_keys=False):
        with open(path, "r") as f:
            d = json.load(f)
        if int_keys:
            d = {int(k): v for k, v in d.items()}
        return d

    def calculate_ump_zone_acc(self):
        self.ump_zone_acc = self.df.groupby(['ump_id', 'zone'])['correct'].mean()

    @staticmethod
    def save_series(ser, path):
        # ser.to_frame().to_parquet(path)
        joblib.dump(ser, path)
    
    @staticmethod
    def load_series(path):
        return joblib.load(path)

    def show_ump_heatmap(self, ump_id=None, catcher_perspective=False, ump_zone_acc=None, lg_avg=False):
        """
        Plot an umpire's zone accuracy heatmap.

        Args:
            ump_id (int, optional): Umpire ID to display.
            catcher_perspective (bool): If True, view from catcher's perspective.
            ump_zone_acc (pd.Series, optional): Zone accuracy data.
            lg_avg (bool): Show league average instead of individual ump.
        Returns:
            (fig, ax): Matplotlib figure and axis.
        """

        if ump_zone_acc is None:
            ump_zone_acc = self.ump_zone_acc
        ump_name_dct = self.ump_name_dct

        if ump_id is None:
            ump_id = int(np.random.choice(list(ump_name_dct.keys())))

        if lg_avg:
            zones = self.df.groupby('zone')['correct'].mean()
            ump_name = 'League Average'
        else:
            zones = ump_zone_acc.loc[ump_id]
            ump_name = ump_name_dct[ump_id]

        small_idx = [1,2,3,4,5,6,7,8,9]
        big_idx   = [11,12,13,14]

        # Data arrays in catcher perspective canonical order
        small = zones.reindex(small_idx).to_numpy().reshape(3,3)          # 1..9
        big   = zones.reindex(big_idx).to_numpy().reshape(2,2)            # [[11,12],[13,14]]

        zone_nums_small = np.array(small_idx).reshape(3,3)                # labels 1..9
        zone_nums_big   = np.array([[11,12],[13,14]])                     # labels 11..14

        # If pitcher perspective, mirror LEFT<->RIGHT for BOTH data and labels
        if not catcher_perspective:
            small = np.fliplr(small)
            big   = np.fliplr(big)
            zone_nums_small = np.fliplr(zone_nums_small)
            zone_nums_big   = np.fliplr(zone_nums_big)

        center = float(ump_zone_acc.mean())
        norm = TwoSlopeNorm(vmin=0.50, vcenter=center, vmax=1.00)

        fig, ax = plt.subplots(figsize=(7,7))

        # Geometry
        pad = 0.35
        big_extent   = [-pad, 2+pad, -pad, 2+pad]
        small_extent = [0, 2, 0, 2]

        # Draw heatmaps
        ax.imshow(big,   extent=big_extent,   origin="upper", cmap="bwr", norm=norm)
        im = ax.imshow(small, extent=small_extent, origin="upper", cmap="bwr", norm=norm)

        # Borders
        ax.plot([big_extent[0], big_extent[1]], [big_extent[2], big_extent[2]], color="k")
        ax.plot([big_extent[0], big_extent[1]], [big_extent[3], big_extent[3]], color="k")
        ax.plot([big_extent[0], big_extent[0]], [big_extent[2], big_extent[3]], color="k")
        ax.plot([big_extent[1], big_extent[1]], [big_extent[2], big_extent[3]], color="k")
        for v in [0, 2/3, 4/3, 2]:
            ax.plot([v, v], [0, 2], color="k")
            ax.plot([0, 2], [v, v], color="k")

        # Labels for zones 1..9
        cell_w, cell_h = 2/3, 2/3
        for r in range(3):
            for c in range(3):
                x_left  = c * cell_w
                x_c     = x_left + cell_w/2
                y_top   = 2 - r * cell_h
                y_c     = y_top - cell_h/2

                ax.text(x_left + 0.03, y_top - 0.03,
                        f"{zone_nums_small[r,c]}",
                        ha="left", va="top", fontsize=11, color="black")
                ax.text(x_c, y_c,
                        f"{small[r,c]:.0%}",
                        ha="center", va="center", fontsize=12, color="black")

        # Percentages for 11–14 (read from the *displayed* big array)
        z_tl, z_tr = big[0,0], big[0,1]
        z_bl, z_br = big[1,0], big[1,1]
        tab_y_top, tab_y_bottom = 2 + pad/2, -pad/2
        ax.text(0.5,  tab_y_top,    f"{z_tl:.0%}", ha="center", va="center", fontsize=12)
        ax.text(1.5,  tab_y_top,    f"{z_tr:.0%}", ha="center", va="center", fontsize=12)
        ax.text(0.5,  tab_y_bottom, f"{z_bl:.0%}", ha="center", va="center", fontsize=12)
        ax.text(1.5,  tab_y_bottom, f"{z_br:.0%}", ha="center", va="center", fontsize=12)

        # Corner labels 11–14 that match the mirrored orientation
        ax.text(big_extent[0]+0.04, big_extent[3]-0.04, f"{zone_nums_big[0,0]}", ha="left",  va="top",    fontsize=10)
        ax.text(big_extent[1]-0.04, big_extent[3]-0.04, f"{zone_nums_big[0,1]}", ha="right", va="top",    fontsize=10)
        ax.text(big_extent[0]+0.04, big_extent[2]+0.04, f"{zone_nums_big[1,0]}", ha="left",  va="bottom", fontsize=10)
        ax.text(big_extent[1]-0.04, big_extent[2]+0.04, f"{zone_nums_big[1,1]}", ha="right", va="bottom", fontsize=10)

        # Colorbar & cosmetics
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Correct call rate", rotation=270, labelpad=14)
        cbar.set_ticks([0.5, center, 1.0])
        cbar.set_ticklabels([f"0.50", f"{center:.2f}", "1.00"])

        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()

        perspective = 'Catcher' if catcher_perspective else 'Pitcher'
        plt.title(f"{ump_name}'s Zone Accuracy - From {perspective}'s Perspective")
        return fig, ax

    def prepare_train_test_data_mcm(self, target='incorrect', training_size=.75):
        self.target_mcm = target

        # self.feature_cols_mcm = self.cat_cols_mcm + self.num_cols_mcm

        # Ensure datetime sort
        df = self.df.sort_values("game_date")#.reset_index(drop=True)

        X_all = df[self.feature_cols_mcm].copy()
        y_all = df[self.target_mcm].astype(int).values

        # Hold-out test split by time (e.g., last 20% as test)
        cut = int(len(df) * training_size)
        self.X_train_mcm, self.y_train_mcm = X_all.iloc[:cut], y_all[:cut]
        self.X_test_mcm,  self.y_test_mcm  = X_all.iloc[cut:],  y_all[cut:]

    def train_xgb_clf_model(self, X_train, y_train, cat_cols, num_cols):
        # if not hasattr(self, "X_train"):
        #     raise ValueError("Must call prepare_training_data() before training")

        xgb_clf_model = train_xgb(
            X_train,
            y_train,
            cat_cols,
            num_cols
        )
        return xgb_clf_model
    
    def evaluate_xgb_clf_model(self, xgb_clf_model, X_test, y_test):
        """
        Evaluate the Missed Call Model (mcm) on the hold-out test set.
        Returns a dictionary of metrics.
        """
        # if not hasattr(self, "X_test"):
        #     raise ValueError("Must call prepare_training_data() before evaluation")

        model = xgb_clf_model.best_estimator_ if hasattr(xgb_clf_model, "best_estimator_") else xgb_clf_model

        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

        results = {
            "roc_auc": roc_auc_score(y_test, probs),
            "log_loss": log_loss(y_test, probs),
            "f1": f1_score(y_test, preds),
            "average_precision": average_precision_score(y_test, probs),
        }

        # Store results on the instance for later use
        self.eval_results = results
        return results
    
    def calibrate_mcm(self, method="isotonic"):
        model = self.mcm.best_estimator_ if hasattr(self.mcm, "best_estimator_") else self.mcm
        self.mcm_calib = calibrate_model(
            model,
            self.X_train_mcm,
            self.y_train_mcm,
            groups=self.X_train_mcm['game_pk'] if 'game_pk' in self.X_train_mcm else None,
            method=method
        )
        return self.mcm_calib

    @staticmethod
    def save_model(model, path):
        joblib.dump(model, path)

    @staticmethod
    def load_model(path):
        return joblib.load(path)

    def make_wp_df(self):

        # sort games so score can be calculated for final play
        sorted_games = self.raw_df.sort_values(['game_pk', 'inning', 'at_bat_number', 'pitch_number'])

        # get final scores of each game to determine winner
        games_scores = sorted_games.groupby('game_pk')[['post_away_score', 'post_home_score']].last()
        games_scores['home_team_win'] = games_scores.apply(lambda x: x['post_home_score'] > x['post_away_score'], axis=1)

        # map for home team winning
        home_won_dict = games_scores['home_team_win'].to_dict()

        data = self.raw_df

        # select only important columns
        wp_df = data[[
            'game_date', 'game_pk', 'away_team', 'home_team', 'inning_topbot', 'inning', 'at_bat_number', 'pitch_number',
            'outs_when_up', 'on_1b', 'on_2b', 'on_3b', 'balls', 'strikes',
            'away_score', 'home_score', 'post_away_score', 'post_home_score'
        ]].copy()

        # make baserunners' presence booleans
        wp_df[['on_1b','on_2b','on_3b']] = wp_df[['on_1b','on_2b','on_3b']].notna().astype(int).astype(str)
        wp_df['on_1b'] = wp_df[['on_1b','on_2b','on_3b']].sum(axis=1)

        # make perspective from team batting
        wp_df['team_bat_score_diff'] = np.where(wp_df['inning_topbot']=='Bot', wp_df['home_score']-wp_df['away_score'], wp_df['away_score']-wp_df['home_score'])

        # assign whether team batting won or lost
        home_won_ser = data['game_pk'].map(home_won_dict).astype(bool)
        wp_df['team_bat_won'] = np.where(data['inning_topbot']=='Bot', home_won_ser, ~home_won_ser)

        # renaming and dropping columns
        wp_df.rename(columns={
            'on_1b': 'base_state'
        }, inplace=True)

        wp_df.drop(columns={
            'on_2b', 'on_3b'
        }, inplace=True)

        self.wp_df = wp_df
        return wp_df

    def prepare_train_test_data_wpm(self, target='team_bat_won', training_size=.75):
        self.target_wpm = target

        wp_df = self.wp_df

        # self.feature_cols_wpm = self.cat_cols_wpm + self.num_cols_wpm

        # Ensure datetime sort
        wp_train_games, wp_test_games = train_test_split(wp_df['game_pk'].unique())

        wp_train_df = wp_df[wp_df['game_pk'].isin(wp_train_games)]
        wp_test_df = wp_df[wp_df['game_pk'].isin(wp_test_games)]

        self.X_train_wpm, self.X_test_wpm, self.y_train_wpm, self.y_test_wpm = wp_train_df[self.feature_cols_wpm], wp_test_df[self.feature_cols_wpm], wp_train_df[self.target_wpm], wp_test_df[self.target_wpm]
    

    def calibrate_wpm(self, method="isotonic"):
        model = self.wpm.best_estimator_ if hasattr(self.wpm, "best_estimator_") else self.wpm
        self.wpm_calib = calibrate_model(
            model,
            self.X_train_wpm,
            self.y_train_wpm,
            groups=None,  # or supply game_pk if available
            method=method
        )
        return self.wpm_calib

    def merge_main(self):
        wp_df = self.wp_df
        df = self.df
        mcm_calib = self.mcm_calib
        wpm_calib = self.wpm_calib
        feature_cols_mcm = self.feature_cols_mcm
        feature_cols_wpm = self.feature_cols_wpm


        merged = wp_df.drop(columns=['inning','outs_when_up', 'balls', 'strikes']).merge(
            df.drop(columns=['game_date']),
            left_index=True,
            right_index=True
        )

        incorrect_call_probs = mcm_calib.predict_proba(merged[feature_cols_mcm])[:, 1]

        win_probs = wpm_calib.predict_proba(merged[feature_cols_wpm])[:, 1]

        merged['events'] = self.raw_df.loc[merged.index, 'events']
        merged['incorrect_call_prob'] = incorrect_call_probs
        merged['win_prob'] = win_probs

        self.merged = merged
        return merged


    
    def make_future_states(self):
        """
        Compute future possible game states for overturned/confirmed calls.

        Uses `process_states` to simulate inning changes, walkoffs, etc.,
        and then attaches win probability estimates for each new state.
        """

        merged = self.merged
        feature_cols_wpm = self.feature_cols_wpm

        # defined masks to make state processing easier
        cs = merged['call'] == 'strike'
        cb = merged['call'] == 'ball'
        inn = merged['inning']
        outs = merged['outs_when_up']
        b = merged['balls']
        s = merged['strikes']
        diff = merged['team_bat_score_diff']
        tb = merged['inning_topbot']
        bs = merged['base_state']

        S, O, S_l, O_l, S_w, O_w = process_states(
            merged=merged,
            feature_cols_wpm=feature_cols_wpm,
            cs=cs,   # mask: called strike
            cb=cb,   # mask: called ball
            outs=outs,
            s=s,     # strikes
            inn=inn, # inning
            diff=diff, # score diff from batting team perspective
            b=b,     # balls
            tb=tb,   # inning_topbot
            bs=bs    # base_state
        )

        # add win probability for future states using model
        for df in (S, O):
            if not df.empty:
                add_win_prob_future_states(df, self.wpm_calib, self.feature_cols_wpm)

        # win probability for these scenarios is a guranteed loss
        for df in (S_l, O_l):
            if not df.empty:
                df.drop(columns = 'inning_topbot_switch', inplace=True)
                df['win_prob'] = 0

        # win probability for these scenarios is a guranteed win
        for df in (S_w, O_w):
            if not df.empty:
                df.drop(columns = 'inning_topbot_switch', inplace=True)
                df['win_prob'] = 1

        self.call_stands_df = pd.concat([S, S_l, S_w])
        self.call_overturned_df = pd.concat([O, O_l, O_w])

    def merge_final(self, include_unneeded_cols=False):
        merged = self.merged
        call_stands_df = self.call_stands_df
        call_overturned_df = self.call_overturned_df
        feature_cols_mcm = self.feature_cols_mcm
        feature_cols_wpm = self.feature_cols_wpm
        remaining_needed_cols = ['call', 'incorrect_call_prob', 'win_prob', 'win_prob_s', 'win_prob_o']
        remaining_unneeded_cols = ['ump_name', 'game_pk', 'events', 'real', 'correct', 'incorrect', 'incorrect_call_prob', 'win_prob']
        cs = merged['call'] == 'strike'

        merged_final = merged.join(call_stands_df['win_prob'].rename('win_prob_s'))
        merged_final = merged_final.join(call_overturned_df['win_prob'].rename('win_prob_o'))
        if include_unneeded_cols:
            merged_final = merged_final[feature_cols_mcm + feature_cols_wpm + remaining_needed_cols + remaining_unneeded_cols]
        else:
            merged_final = merged_final[feature_cols_mcm + feature_cols_wpm + remaining_needed_cols]
        merged_final = merged_final.loc[:, ~merged_final.columns.duplicated()]
        merged_final['win_prob_u'] = np.where(
            cs,
            merged_final['win_prob_o'] - merged_final['win_prob_s'],
            merged_final['win_prob_s'] - merged_final['win_prob_o']
        ).clip(0)

        self.merged_final = merged_final
        return merged_final
    
    @staticmethod
    def calculate_lambda(group):
        group_df = group.copy() #pd.DataFrame()
        group_df['p'] = group['incorrect_call_prob']
        group_df['c'] = 1 - group_df['p']
        group_df['v'] = group_df['p'] * group_df['win_prob_u']
        group_df['r'] = group_df['v'] / group_df['c']
        group_df = group_df.sort_values('r', ascending=False)
        return group_df[group_df['c'].cumsum() <= 2].iloc[-1]['r']

    @staticmethod
    def calculate_delta_we(u, p, lam):
        return (p * u) - (lam * (1-p))

    @staticmethod
    def calculate_cost_mean(df, lam):
        u = df['win_prob_u']
        p = df['incorrect_call_prob']
        positive_we_mask = ABSHelper.calculate_delta_we(u, p, lam) > 0
        
        return df.loc[positive_we_mask]\
        .groupby(['game_pk','inning_topbot'])[['incorrect_call_prob']]\
        .apply(lambda x: (1-x['incorrect_call_prob']).sum(), include_groups=False).mean()
    
    def find_initial_lambda(self):
        lambda_df = self.merged_final[['game_pk', 'inning_topbot', 'incorrect', 'incorrect_call_prob', 'win_prob_u']]
        lambda_groups = lambda_df.groupby(['game_pk', 'inning_topbot']).apply(self.calculate_lambda, include_groups=False)
        return lambda_groups.mean()

    def find_best_lambda(self, lam_range):
        lambda_df = self.merged_final[['game_pk', 'inning_topbot', 'incorrect', 'incorrect_call_prob', 'win_prob_u']]
        cost_means = list()
        for lam in lam_range:
            # positive_we_mask = self.calculate_delta_we(u, p, lam) > 0
            cost_mean = self.calculate_cost_mean(lambda_df, lam)
            cost_means.append(cost_mean)

        cost_means_df = pd.DataFrame()
        cost_means_df.index = lam_range
        cost_means_df['cost_mean'] = cost_means
        cost_means_df['closest'] = (cost_means_df['cost_mean'] - 2).abs()
        cost_means_df.sort_values('closest')
        best_lambda = float(cost_means_df['closest'].idxmin())

        plt.scatter(lam_range, cost_means)
        plt.axhline(2, c='red')

        self.best_lambda = best_lambda
        return best_lambda
    
    def add_delta_we_col(self, lam=None):
        """
        Add ΔWE (change in win expectancy) to the merged DataFrame.

        ΔWE is the difference between the win probability if the call
        is overturned versus if it is upheld.
        """

        df = self.merged_final
        u = df['win_prob_u']
        p = df['incorrect_call_prob']

        if lam is None:
            lam = self.universal_lambda
        
        df['delta_we'] = self.calculate_delta_we(u, p, lam)
        
        df['challenge'] = df['delta_we'] > 0
        
class ABSInterface:
    """
    Thin wrapper around ABSHelper for simple, user-friendly interaction.

    Provides a two-step workflow:
    1. `predict_incorrect_call_prob` → computes chance of incorrect call.
    2. `predict_dwe` → adds game context, computes ΔWE and challenge decision.
    """
    def __init__(self, model_path_mcm, model_path_wpm, ump_name_path, ump_zone_path):
        
        # create an ABSHelper instance
        self.helper = ABSHelper()
        self.helper.mcm_calib = self.helper.load_model(model_path_mcm)
        self.helper.wpm_calib = self.helper.load_model(model_path_wpm)
        self.helper.ump_name_dct = self.helper.load_dct(ump_name_path, int_keys=True)
        self.helper.ump_zone_acc = self.helper.load_series(ump_zone_path)

    def predict_incorrect_call_prob(self, feature_dct_mcm):
        """
        Step 1: Compute probability that a call was incorrect.

        Args:
            feature_dct_mcm (dict): Dictionary of features required by MCM.
        Returns:
            float: Probability (%) that call was incorrect.
        """
        self.feature_dct_mcm = feature_dct_mcm
        mc_df = pd.DataFrame([feature_dct_mcm])
        prob = round(float(self.helper.mcm_calib.predict_proba(mc_df)[:,1][0])*100, 3)
        return prob
    
    def predict_dwe(self, feature_dct_wpm):
        """
        Step 2: Compute ΔWE and challenge decision.

        Args:
            feature_dct_wpm (dict): Additional game context features.
        Returns:
            dict: { 'delta_we': float, 'challenge': bool }
        """
        
        self.feature_dct_all = self.feature_dct_mcm.copy()
        self.feature_dct_all.update(feature_dct_wpm)

        df = pd.DataFrame([self.feature_dct_all])

        self.helper.merged = df.copy()

        self.helper.merged['incorrect_call_prob'] = self.helper.mcm_calib.predict_proba(self.helper.merged[self.helper.feature_cols_mcm])[:,1]
        self.helper.merged['win_prob'] = self.helper.wpm_calib.predict_proba(self.helper.merged[self.helper.feature_cols_wpm])[:,1]
        self.helper.make_future_states()
        self.helper.merge_final()
        self.helper.add_delta_we_col()

        # self.helper.merged_final[['incorrect_call_prob', 'delta_we','challenge']]

        return {
            # "incorrect_call_prob": float(self.helper.merged_final['incorrect_call_prob'].values[0]),
            "delta_we": round(float(self.helper.merged_final['delta_we'].values[0]), 5),
            "challenge": bool(self.helper.merged_final['challenge'].values[0])
        }
