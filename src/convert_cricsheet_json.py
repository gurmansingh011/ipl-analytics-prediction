"""Convert Cricsheet IPL JSON ZIP files into matches.csv and deliveries.csv."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import pandas as pd


def convert_cricsheet_zip(zip_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Create Kaggle-style matches.csv and deliveries.csv from Cricsheet JSON."""
    matches = []
    deliveries = []

    with zipfile.ZipFile(zip_path) as archive:
        json_names = sorted(name for name in archive.namelist() if name.lower().endswith(".json"))
        if not json_names:
            raise ValueError(f"No JSON files found in {zip_path}")

        for name in json_names:
            match_json = json.loads(archive.read(name))
            info = match_json.get("info", {})
            match_id = int(Path(name).stem)
            teams = info.get("teams", ["Unknown Team 1", "Unknown Team 2"])
            outcome = info.get("outcome", {})
            toss = info.get("toss", {})
            event = info.get("event", {})
            dates = info.get("dates", [])
            date = str(dates[0]) if dates else ""
            season = str(info.get("season", date[:4] if date else "Unknown"))

            matches.append(
                {
                    "id": match_id,
                    "season": season,
                    "city": info.get("city", "Unknown"),
                    "date": date,
                    "match_number": event.get("match_number", ""),
                    "team1": teams[0] if len(teams) > 0 else "Unknown Team 1",
                    "team2": teams[1] if len(teams) > 1 else "Unknown Team 2",
                    "toss_winner": toss.get("winner", ""),
                    "toss_decision": toss.get("decision", ""),
                    "winner": outcome.get("winner", "No Result"),
                    "result": outcome.get("result", "normal"),
                    "player_of_match": ", ".join(info.get("player_of_match", [])),
                    "venue": info.get("venue", "Unknown Venue"),
                }
            )

            for inning_number, inning in enumerate(match_json.get("innings", []), start=1):
                batting_team = inning.get("team", "Unknown")
                bowling_team = next((team for team in teams if team != batting_team), "Unknown")

                for over in inning.get("overs", []):
                    over_number = int(over.get("over", 0))
                    for ball_number, delivery in enumerate(over.get("deliveries", []), start=1):
                        runs = delivery.get("runs", {})
                        wickets = delivery.get("wickets", [])
                        extras = delivery.get("extras", {})
                        dismissal = wickets[0] if wickets else {}

                        deliveries.append(
                            {
                                "match_id": match_id,
                                "inning": inning_number,
                                "over": over_number,
                                "ball": ball_number,
                                "batting_team": batting_team,
                                "bowling_team": bowling_team,
                                "batter": delivery.get("batter", ""),
                                "non_striker": delivery.get("non_striker", ""),
                                "bowler": delivery.get("bowler", ""),
                                "batsman_runs": runs.get("batter", 0),
                                "extra_runs": runs.get("extras", 0),
                                "total_runs": runs.get("total", 0),
                                "extras_type": ",".join(extras.keys()),
                                "is_wicket": int(bool(wickets)),
                                "player_dismissed": dismissal.get("player_out", ""),
                                "dismissal_kind": dismissal.get("kind", ""),
                                "fielder": ",".join(
                                    fielder.get("name", "") for fielder in dismissal.get("fielders", [])
                                ),
                            }
                        )

    output_dir.mkdir(parents=True, exist_ok=True)
    matches_path = output_dir / "matches.csv"
    deliveries_path = output_dir / "deliveries.csv"
    pd.DataFrame(matches).sort_values(["season", "date", "id"]).to_csv(matches_path, index=False)
    pd.DataFrame(deliveries).to_csv(deliveries_path, index=False)
    return matches_path, deliveries_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Cricsheet IPL JSON ZIP to app-ready CSV files.")
    parser.add_argument("zip_path", type=Path, help="Path to ipl_json.zip")
    parser.add_argument("--output-dir", type=Path, default=Path("data"), help="Folder where CSV files will be saved")
    args = parser.parse_args()

    matches_path, deliveries_path = convert_cricsheet_zip(args.zip_path, args.output_dir)
    print(f"Saved {matches_path}")
    print(f"Saved {deliveries_path}")


if __name__ == "__main__":
    main()
