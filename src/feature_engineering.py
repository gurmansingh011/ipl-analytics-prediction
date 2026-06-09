"""Feature engineering for IPL match outcome prediction."""

from __future__ import annotations

import pandas as pd


class MatchFeatureEngineer:
    """Create second-innings chase state rows for win probability modeling."""

    def __init__(self, matches: pd.DataFrame, deliveries: pd.DataFrame):
        self.matches = matches
        self.deliveries = deliveries

    def build_training_data(self) -> pd.DataFrame:
        rows = []
        match_lookup = self.matches.set_index("id")
        for match_id, group in self.deliveries.groupby("match_id"):
            if match_id not in match_lookup.index:
                continue
            match = match_lookup.loc[match_id]
            innings = group.groupby("inning")
            if 1 not in innings.groups or 2 not in innings.groups:
                continue
            target = int(innings.get_group(1)["total_runs"].sum()) + 1
            chase = innings.get_group(2).sort_values(["over", "ball"]).copy()
            chase["current_score"] = chase["total_runs"].cumsum()
            chase["wickets_lost"] = chase["is_wicket"].cumsum().clip(upper=10)
            chase["overs_completed"] = chase["over"] + chase["ball"] / 6
            for _, row in chase.iterrows():
                batting_team = row["batting_team"]
                rows.append(
                    {
                        "batting_team": batting_team,
                        "bowling_team": row["bowling_team"],
                        "venue": match["venue"],
                        "target_score": target,
                        "current_score": int(row["current_score"]),
                        "wickets_lost": int(row["wickets_lost"]),
                        "overs_completed": float(row["overs_completed"]),
                        "runs_required": max(target - int(row["current_score"]), 0),
                        "balls_remaining": max(120 - int(row["over"]) * 6 - int(row["ball"]), 0),
                        "run_rate": int(row["current_score"]) / max(float(row["overs_completed"]), 0.1),
                        "required_run_rate": max(target - int(row["current_score"]), 0) / max((120 - int(row["over"]) * 6 - int(row["ball"])) / 6, 0.1),
                        "result": int(match["winner"] == batting_team),
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def add_runtime_features(input_df: pd.DataFrame) -> pd.DataFrame:
        df = input_df.copy()
        df["runs_required"] = (df["target_score"] - df["current_score"]).clip(lower=0)
        df["balls_remaining"] = (120 - (df["overs_completed"] * 6).round()).clip(lower=0)
        df["run_rate"] = df["current_score"] / df["overs_completed"].replace(0, 0.1)
        df["required_run_rate"] = df["runs_required"] / (df["balls_remaining"] / 6).replace(0, 0.1)
        return df
