"""Data loading and preprocessing utilities for IPL analytics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TEAM_ALIASES = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
}


@dataclass
class IPLDataPreprocessor:
    """Load IPL match and delivery files and expose cleaned dataframes."""

    data_dir: Path | str = Path("data")

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.matches_path = self._first_existing("matches.csv", "matches.csv.gz")
        self.deliveries_path = self._first_existing("deliveries.csv", "deliveries.csv.gz")

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
        """Return cleaned matches, deliveries, and a flag indicating demo data."""
        if self.matches_path.exists() and self.deliveries_path.exists():
            matches = pd.read_csv(self.matches_path)
            deliveries = pd.read_csv(self.deliveries_path)
            return self.clean_matches(matches), self.clean_deliveries(deliveries), False

        matches, deliveries = self._demo_data()
        return self.clean_matches(matches), self.clean_deliveries(deliveries), True

    def _first_existing(self, *names: str) -> Path:
        for name in names:
            path = self.data_dir / name
            if path.exists():
                return path
        return self.data_dir / names[0]

    def clean_matches(self, matches: pd.DataFrame) -> pd.DataFrame:
        matches = matches.copy().drop_duplicates()
        matches.columns = [c.strip().lower().replace(" ", "_") for c in matches.columns]
        matches = self._normalize_match_columns(matches)

        for col in ["team1", "team2", "toss_winner", "winner"]:
            if col in matches:
                matches[col] = self.standardize_team_names(matches[col])

        matches["season"] = matches["season"].fillna(matches.get("date", "")).astype(str).str[:4]
        matches["venue"] = matches["venue"].fillna("Unknown Venue")
        matches["city"] = matches.get("city", pd.Series("Unknown", index=matches.index)).fillna("Unknown")
        matches["winner"] = matches["winner"].fillna("No Result")
        matches["toss_decision"] = matches.get("toss_decision", "field")
        matches["toss_decision"] = matches["toss_decision"].fillna("field").str.lower()
        return matches

    def clean_deliveries(self, deliveries: pd.DataFrame) -> pd.DataFrame:
        deliveries = deliveries.copy().drop_duplicates()
        deliveries.columns = [c.strip().lower().replace(" ", "_") for c in deliveries.columns]
        deliveries = self._normalize_delivery_columns(deliveries)

        for col in ["batting_team", "bowling_team"]:
            if col in deliveries:
                deliveries[col] = self.standardize_team_names(deliveries[col])

        numeric_cols = [
            "match_id",
            "inning",
            "over",
            "ball",
            "batsman_runs",
            "extra_runs",
            "total_runs",
            "is_wicket",
        ]
        for col in numeric_cols:
            if col in deliveries:
                deliveries[col] = pd.to_numeric(deliveries[col], errors="coerce").fillna(0).astype(int)

        deliveries["batter"] = deliveries["batter"].fillna("Unknown Batter")
        deliveries["bowler"] = deliveries["bowler"].fillna("Unknown Bowler")
        deliveries["player_dismissed"] = deliveries.get("player_dismissed", pd.Series("", index=deliveries.index)).fillna("")
        deliveries["dismissal_kind"] = deliveries.get("dismissal_kind", pd.Series("", index=deliveries.index)).fillna("")
        return deliveries

    def merge_data(self, matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
        keep = ["id", "season", "city", "venue", "team1", "team2", "winner"]
        return deliveries.merge(matches[[c for c in keep if c in matches]], left_on="match_id", right_on="id", how="left")

    @staticmethod
    def standardize_team_names(values: Iterable) -> pd.Series:
        return pd.Series(values).replace(TEAM_ALIASES)

    @staticmethod
    def _normalize_match_columns(matches: pd.DataFrame) -> pd.DataFrame:
        rename = {"match_id": "id", "date_year": "season"}
        matches = matches.rename(columns={k: v for k, v in rename.items() if k in matches.columns})
        if "id" not in matches:
            matches["id"] = np.arange(1, len(matches) + 1)
        if "season" not in matches:
            matches["season"] = matches.get("date", "Unknown")
        return matches

    @staticmethod
    def _normalize_delivery_columns(deliveries: pd.DataFrame) -> pd.DataFrame:
        rename = {
            "id": "match_id",
            "matchid": "match_id",
            "battingteam": "batting_team",
            "bowlingteam": "bowling_team",
            "batsman": "batter",
            "batsman_runs": "batsman_runs",
            "non_striker": "non_striker",
        }
        deliveries = deliveries.rename(columns={k: v for k, v in rename.items() if k in deliveries.columns})
        defaults = {
            "match_id": 0,
            "inning": 1,
            "over": 0,
            "ball": 1,
            "batting_team": "Unknown",
            "bowling_team": "Unknown",
            "batter": "Unknown Batter",
            "bowler": "Unknown Bowler",
            "batsman_runs": 0,
            "extra_runs": 0,
            "total_runs": 0,
            "is_wicket": 0,
        }
        for col, default in defaults.items():
            if col not in deliveries:
                deliveries[col] = default
        return deliveries

    @staticmethod
    def _demo_data() -> tuple[pd.DataFrame, pd.DataFrame]:
        teams = [
            "Chennai Super Kings",
            "Mumbai Indians",
            "Royal Challengers Bengaluru",
            "Kolkata Knight Riders",
            "Delhi Capitals",
            "Punjab Kings",
        ]
        venues = ["Wankhede Stadium", "M. A. Chidambaram Stadium", "Eden Gardens", "Arun Jaitley Stadium"]
        matches = []
        deliveries = []
        players = {
            team: [f"{team.split()[0]} Batter {i}" for i in range(1, 6)] for team in teams
        }

        match_id = 1
        rng = np.random.default_rng(42)
        for season in range(2018, 2024):
            for i in range(6):
                team1 = teams[(i + season) % len(teams)]
                team2 = teams[(i + season + 2) % len(teams)]
                winner = team1 if (i + season) % 3 else team2
                venue = venues[(i + season) % len(venues)]
                toss_winner = team1 if i % 2 else team2
                matches.append(
                    {
                        "id": match_id,
                        "season": season,
                        "city": venue.split()[0],
                        "date": f"{season}-04-{i + 1:02d}",
                        "team1": team1,
                        "team2": team2,
                        "toss_winner": toss_winner,
                        "toss_decision": "field" if i % 2 else "bat",
                        "winner": winner,
                        "venue": venue,
                    }
                )

                for inning, batting, bowling in [(1, team1, team2), (2, team2, team1)]:
                    score = 0
                    for over in range(20):
                        for ball in range(1, 7):
                            runs = int(rng.choice([0, 1, 1, 2, 4, 6], p=[0.28, 0.32, 0.1, 0.14, 0.11, 0.05]))
                            wicket = int(rng.random() < 0.045)
                            batter = rng.choice(players[batting])
                            bowler = rng.choice(players[bowling])
                            score += runs
                            deliveries.append(
                                {
                                    "match_id": match_id,
                                    "inning": inning,
                                    "over": over,
                                    "ball": ball,
                                    "batting_team": batting,
                                    "bowling_team": bowling,
                                    "batter": batter,
                                    "bowler": bowler,
                                    "batsman_runs": runs,
                                    "extra_runs": 0,
                                    "total_runs": runs,
                                    "is_wicket": wicket,
                                    "player_dismissed": batter if wicket else "",
                                    "dismissal_kind": "caught" if wicket else "",
                                }
                            )
                            if inning == 2 and winner == batting and score > 150:
                                break
                        else:
                            continue
                        break
                match_id += 1

        return pd.DataFrame(matches), pd.DataFrame(deliveries)
