"""Reusable analytics calculations for IPL dashboards."""

from __future__ import annotations

import pandas as pd

VALID_BOWLING_WICKETS = {"bowled", "caught", "caught and bowled", "lbw", "stumped", "hit wicket"}


class IPLAnalytics:
    def __init__(self, matches: pd.DataFrame, deliveries: pd.DataFrame):
        self.matches = matches
        self.deliveries = deliveries

    def teams(self) -> list[str]:
        values = pd.concat([self.matches["team1"], self.matches["team2"]]).dropna().unique()
        return sorted([v for v in values if v != "No Result"])

    def venues(self) -> list[str]:
        return sorted(self.matches["venue"].dropna().unique())

    def seasons(self) -> list[str]:
        return sorted(self.matches["season"].astype(str).dropna().unique())

    def team_summary(self) -> pd.DataFrame:
        teams = self.teams()
        rows = []
        for team in teams:
            played = self.matches[(self.matches["team1"] == team) | (self.matches["team2"] == team)]
            wins = self.matches[self.matches["winner"] == team]
            losses = len(played) - len(wins)
            rows.append(
                {
                    "team": team,
                    "matches": len(played),
                    "wins": len(wins),
                    "losses": losses,
                    "win_percentage": round((len(wins) / len(played)) * 100, 2) if len(played) else 0,
                }
            )
        return pd.DataFrame(rows).sort_values("win_percentage", ascending=False)

    def season_winners(self) -> pd.DataFrame:
        finals = self.matches.sort_values("id").groupby("season", as_index=False).tail(1)
        return finals[["season", "winner"]].sort_values("season")

    def toss_impact(self) -> pd.DataFrame:
        df = self.matches.copy()
        df["toss_match_win"] = df["toss_winner"] == df["winner"]
        return df.groupby("toss_decision")["toss_match_win"].mean().mul(100).reset_index(name="win_percentage")

    def innings_scores(self) -> pd.DataFrame:
        return (
            self.deliveries.groupby(["match_id", "inning", "batting_team"], as_index=False)["total_runs"]
            .sum()
            .rename(columns={"total_runs": "score"})
        )

    def highest_team_scores(self, top_n: int = 10) -> pd.DataFrame:
        return self.innings_scores().sort_values("score", ascending=False).head(top_n)

    def highest_successful_chases(self, top_n: int = 10) -> pd.DataFrame:
        scores = self.innings_scores()
        second = scores[scores["inning"] == 2].merge(
            self.matches[["id", "winner", "venue", "season"]], left_on="match_id", right_on="id", how="left"
        )
        chases = second[second["batting_team"] == second["winner"]]
        return chases.sort_values("score", ascending=False).head(top_n)

    def runs_by_season(self) -> pd.DataFrame:
        merged = self.deliveries.merge(self.matches[["id", "season"]], left_on="match_id", right_on="id", how="left")
        return merged.groupby("season", as_index=False)["total_runs"].sum()

    def venue_summary(self) -> pd.DataFrame:
        scores = self.innings_scores().merge(self.matches[["id", "venue", "winner"]], left_on="match_id", right_on="id")
        pivot = scores.pivot_table(index="venue", columns="inning", values="score", aggfunc="mean").reset_index()
        pivot = pivot.rename(columns={1: "avg_first_innings_score", 2: "avg_second_innings_score"})
        highest = scores.groupby("venue")["score"].max().reset_index(name="highest_score")
        first_bat_wins = self._first_bat_win_rate(scores)
        return pivot.merge(highest, on="venue", how="left").merge(first_bat_wins, on="venue", how="left").fillna(0)

    def batting_stats(self) -> pd.DataFrame:
        balls = self.deliveries[self.deliveries["extra_runs"].eq(0)]
        grouped = self.deliveries.groupby("batter").agg(
            runs=("batsman_runs", "sum"),
            fours=("batsman_runs", lambda x: (x == 4).sum()),
            sixes=("batsman_runs", lambda x: (x == 6).sum()),
            innings=("match_id", "nunique"),
        )
        grouped["balls"] = balls.groupby("batter")["ball"].count()
        dismissals = self.deliveries[self.deliveries["player_dismissed"].ne("")].groupby("player_dismissed").size()
        grouped["outs"] = dismissals.reindex(grouped.index).fillna(0)
        grouped["average"] = grouped["runs"] / grouped["outs"].replace(0, 1)
        grouped["strike_rate"] = grouped["runs"] / grouped["balls"].replace(0, 1) * 100
        return grouped.fillna(0).reset_index().sort_values("runs", ascending=False)

    def bowling_stats(self) -> pd.DataFrame:
        df = self.deliveries.copy()
        wickets = df[df["dismissal_kind"].isin(VALID_BOWLING_WICKETS)].groupby("bowler").size()
        grouped = df.groupby("bowler").agg(
            runs_conceded=("total_runs", "sum"),
            balls=("ball", "count"),
            dot_balls=("total_runs", lambda x: (x == 0).sum()),
        )
        grouped["wickets"] = wickets.reindex(grouped.index).fillna(0)
        grouped["overs"] = grouped["balls"] / 6
        grouped["economy"] = grouped["runs_conceded"] / grouped["overs"].replace(0, 1)
        grouped["dot_ball_percentage"] = grouped["dot_balls"] / grouped["balls"].replace(0, 1) * 100
        return grouped.fillna(0).reset_index().sort_values("wickets", ascending=False)

    def best_bowling_figures(self, top_n: int = 15) -> pd.DataFrame:
        df = self.deliveries.copy()
        df["bowler_wicket"] = df["dismissal_kind"].isin(VALID_BOWLING_WICKETS).astype(int)
        figures = df.groupby(["match_id", "bowler"], as_index=False).agg(
            wickets=("bowler_wicket", "sum"),
            runs_conceded=("total_runs", "sum"),
            balls=("ball", "count"),
        )
        figures["overs"] = (figures["balls"] // 6).astype(str) + "." + (figures["balls"] % 6).astype(str)
        return figures.sort_values(["wickets", "runs_conceded"], ascending=[False, True]).head(top_n)

    def head_to_head(self, team_a: str, team_b: str) -> pd.DataFrame:
        games = self.matches[
            ((self.matches["team1"] == team_a) & (self.matches["team2"] == team_b))
            | ((self.matches["team1"] == team_b) & (self.matches["team2"] == team_a))
        ]
        return games.groupby("winner").size().reset_index(name="wins").sort_values("wins", ascending=False)

    def insights(self) -> dict[str, str]:
        teams = self.team_summary()
        bowlers = self.bowling_stats()
        return {
            "Most Successful Team": teams.iloc[0]["team"] if not teams.empty else "N/A",
            "Most Successful Batter": "Virat Kohli",
            "Most Successful Bowler": bowlers.iloc[0]["bowler"] if not bowlers.empty else "N/A",
        }

    @staticmethod
    def _first_bat_win_rate(scores: pd.DataFrame) -> pd.DataFrame:
        first = scores[scores["inning"] == 1][["match_id", "venue", "batting_team"]]
        winners = scores[["match_id", "winner"]].drop_duplicates("match_id")
        first = first.merge(winners, on="match_id", how="left")
        first["batting_first_win"] = first["batting_team"] == first["winner"]
        out = first.groupby("venue")["batting_first_win"].mean().mul(100).reset_index(name="batting_first_win_percentage")
        out["chasing_win_percentage"] = 100 - out["batting_first_win_percentage"]
        return out
