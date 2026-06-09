from __future__ import annotations

import pickle
from html import escape
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics import IPLAnalytics
from src.data_preprocessing import IPLDataPreprocessor
from src.feature_engineering import MatchFeatureEngineer
from src.train_model import MatchPredictorTrainer
from src.visualization import bar_chart, confusion_matrix_heatmap, line_chart, pie_chart


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "models" / "match_predictor.pkl"


st.set_page_config(page_title="IPL Analytics & Prediction", layout="wide")


LABEL_OVERRIDES = {
    "display_label": "",
    "team": "Team",
    "season": "Season",
    "type": "Type",
    "matches": "Matches",
    "wins": "Wins",
    "win_percentage": "Win %",
    "total_runs": "Total Runs",
    "toss_decision": "Toss Decision",
    "runs": "Runs",
    "fours": "Fours",
    "sixes": "Sixes",
    "strike_rate": "Strike Rate",
    "wickets": "Wickets",
    "economy": "Economy",
    "dot_ball_percentage": "Dot Ball %",
    "avg_first_innings_score": "Avg 1st Innings Score",
    "avg_second_innings_score": "Avg 2nd Innings Score",
    "highest_score": "Highest Score",
    "chasing_win_percentage": "Chasing Win %",
    "model": "Model",
    "accuracy": "Accuracy",
}


def label_for(name: str | None) -> str:
    if not name:
        return ""
    return LABEL_OVERRIDES.get(str(name), str(name).replace("_", " ").title())


def labels_for(*names: str) -> dict[str, str]:
    return {name: label_for(name) for name in names if name}


def hover_label(name: str | None) -> str:
    label = label_for(name)
    return label if label else "Name"


def bar_chart(df, x, y, title, color=None, orientation="v", height=430, top_n=None, sort_ascending=False):
    """Local themed bar chart wrapper to avoid stale helper imports in Streamlit sessions."""
    chart_df = df.copy()
    if top_n:
        chart_df = chart_df.sort_values(y, ascending=sort_ascending).head(top_n)

    colorway = ["#18c8ff", "#f7c948", "#e6408a", "#31d0aa", "#7c5cff", "#ff8a3d", "#4ea1ff", "#f95f62"]
    if orientation == "h":
        chart_df = chart_df.sort_values(y, ascending=True)
        fig = px.bar(
            chart_df,
            x=y,
            y=x,
            title=title,
            color=color,
            orientation="h",
            text=y,
            color_discrete_sequence=colorway,
            labels=labels_for(x, y, color),
        )
        fig.update_traces(texttemplate="%{x:.3s}", textposition="outside", cliponaxis=False)
        fig.update_traces(hovertemplate=f"{hover_label(x)}: %{{y}}<br>{hover_label(y)}: %{{x}}<extra></extra>")
        margin = {"l": 150, "r": 52, "t": 62, "b": 34}
    else:
        fig = px.bar(
            chart_df,
            x=x,
            y=y,
            title=title,
            color=color,
            text=y,
            color_discrete_sequence=colorway,
            labels=labels_for(x, y, color),
        )
        fig.update_traces(texttemplate="%{y:.3s}", textposition="outside", cliponaxis=False)
        fig.update_traces(hovertemplate=f"{hover_label(x)}: %{{x}}<br>{hover_label(y)}: %{{y}}<extra></extra>")
        fig.update_xaxes(tickangle=0)
        margin = {"l": 24, "r": 42, "t": 62, "b": 42}

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,20,61,0.25)",
        font={"family": "Inter, Segoe UI, sans-serif", "color": "#dfeaff"},
        title={"font": {"size": 18, "color": "#ffffff"}, "x": 0.02, "xanchor": "left"},
        margin=margin,
        height=height,
        showlegend=bool(color and color != x),
    )
    if orientation == "h":
        fig.update_xaxes(title_text=label_for(y), gridcolor="rgba(255,255,255,0.10)", tickfont={"color": "#c7d6f7", "size": 11})
        fig.update_yaxes(title_text="", gridcolor="rgba(255,255,255,0.08)", tickfont={"color": "#c7d6f7", "size": 11})
    else:
        fig.update_xaxes(title_text=label_for(x), gridcolor="rgba(255,255,255,0.10)", tickfont={"color": "#c7d6f7", "size": 11})
        fig.update_yaxes(title_text=label_for(y), gridcolor="rgba(255,255,255,0.08)", tickfont={"color": "#c7d6f7", "size": 11})
    return fig


def line_chart(df, x, y, title, color=None):
    colorway = ["#18c8ff", "#f7c948", "#e6408a", "#31d0aa", "#7c5cff", "#ff8a3d", "#4ea1ff", "#f95f62"]
    fig = px.line(df, x=x, y=y, color=color, markers=True, title=title, color_discrete_sequence=colorway, labels=labels_for(x, y, color))
    fig.update_traces(line={"width": 3}, marker={"size": 8})
    fig.update_traces(hovertemplate=f"{hover_label(x)}: %{{x}}<br>{hover_label(y)}: %{{y}}<extra></extra>")
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,20,61,0.25)",
        font={"family": "Inter, Segoe UI, sans-serif", "color": "#dfeaff"},
        title={"font": {"size": 18, "color": "#ffffff"}, "x": 0.02, "xanchor": "left"},
        margin={"l": 52, "r": 34, "t": 62, "b": 48},
        height=420,
        showlegend=bool(color),
    )
    fig.update_xaxes(title_text=label_for(x), gridcolor="rgba(255,255,255,0.10)", tickfont={"color": "#c7d6f7", "size": 11})
    fig.update_yaxes(title_text=label_for(y), gridcolor="rgba(255,255,255,0.08)", tickfont={"color": "#c7d6f7", "size": 11})
    return fig


def pie_chart(df, names, values, title):
    colorway = ["#18c8ff", "#f7c948", "#e6408a", "#31d0aa", "#7c5cff", "#ff8a3d", "#4ea1ff", "#f95f62"]
    fig = px.pie(df, names=names, values=values, title=title, hole=0.48, color_discrete_sequence=colorway, labels=labels_for(names, values))
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        marker={"line": {"color": "#07143d", "width": 2}},
        textfont={"color": "#ffffff", "size": 13},
        hovertemplate=f"{hover_label(names)}: %{{label}}<br>{hover_label(values)}: %{{value}}<br>Share: %{{percent}}<extra></extra>",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,20,61,0.25)",
        font={"family": "Inter, Segoe UI, sans-serif", "color": "#dfeaff"},
        title={"font": {"size": 18, "color": "#ffffff"}, "x": 0.02, "xanchor": "left"},
        margin={"l": 24, "r": 24, "t": 62, "b": 34},
        height=420,
        showlegend=False,
    )
    return fig


def confusion_matrix_heatmap(matrix, labels, title="Confusion Matrix"):
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale=[
                [0.0, "#13245d"],
                [0.5, "#18c8ff"],
                [1.0, "#f7c948"],
            ],
            text=matrix,
            texttemplate="%{text}",
            textfont={"color": "#ffffff", "size": 16},
            showscale=False,
            hovertemplate="Actual %{y}<br>Predicted %{x}<br>Count %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,20,61,0.25)",
        font={"family": "Inter, Segoe UI, sans-serif", "color": "#dfeaff"},
        title={"text": title, "font": {"size": 18, "color": "#ffffff"}, "x": 0.02, "xanchor": "left"},
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=420,
        margin={"l": 70, "r": 30, "t": 62, "b": 62},
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)", tickfont={"color": "#c7d6f7", "size": 12})
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", tickfont={"color": "#c7d6f7", "size": 12})
    return fig


def apply_ipl_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ipl-navy: #07143d;
            --ipl-blue: #123a8c;
            --ipl-cyan: #18c8ff;
            --ipl-gold: #f7c948;
            --ipl-pink: #e6408a;
            --ipl-text: #f7fbff;
            --ipl-muted: #8ea4d2;
            --ipl-panel: #0d1b4c;
            --ipl-line: rgba(255, 255, 255, 0.14);
        }

        .stApp {
            background:
                radial-gradient(circle at 12% 8%, rgba(24, 200, 255, 0.18), transparent 28%),
                linear-gradient(135deg, #061033 0%, #10225b 48%, #07143d 100%);
            color: var(--ipl-text);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #061033 0%, #0b1a4c 100%);
            border-right: 1px solid var(--ipl-line);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 22px;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p {
            color: var(--ipl-text) !important;
        }

        [data-testid="stSidebar"] .stRadio > label {
            display: none;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] {
            gap: 10px;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label {
            min-height: 48px;
            margin: 0 0 10px;
            padding: 0 !important;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.12);
            transition: transform 180ms ease, background 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
            overflow: hidden;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            transform: translateX(5px);
            background: rgba(24, 200, 255, 0.14);
            border-color: rgba(24, 200, 255, 0.46);
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.2);
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: linear-gradient(90deg, rgba(24, 200, 255, 0.28), rgba(247, 201, 72, 0.13));
            border-color: rgba(247, 201, 72, 0.7);
            box-shadow: inset 4px 0 0 var(--ipl-gold), 0 12px 34px rgba(0, 0, 0, 0.18);
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
            display: none;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label p {
            font-size: 15px;
            font-weight: 850;
            padding: 12px 14px;
        }

        .sidebar-brand {
            margin: 0 0 20px;
            padding: 18px 16px;
            border-radius: 8px;
            background:
                linear-gradient(135deg, rgba(24, 200, 255, 0.18), rgba(230, 64, 138, 0.12)),
                rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.14);
        }

        .sidebar-brand-mark {
            width: 52px;
            height: 52px;
            display: grid;
            place-items: center;
            margin-bottom: 12px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--ipl-cyan), var(--ipl-blue));
            color: #ffffff;
            font-weight: 950;
            border: 2px solid rgba(255, 255, 255, 0.52);
        }

        .sidebar-brand-title {
            color: #ffffff;
            font-size: 22px;
            font-weight: 950;
            line-height: 1.05;
        }

        .sidebar-brand-subtitle {
            color: var(--ipl-muted);
            font-size: 12px;
            font-weight: 800;
            margin-top: 6px;
            text-transform: uppercase;
        }

        .main .block-container {
            max-width: 1440px;
            padding-top: 1.25rem;
        }

        h1, h2, h3 {
            color: var(--ipl-text) !important;
            letter-spacing: 0;
        }

        .ipl-topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 18px;
            padding: 12px 18px;
            margin-bottom: 18px;
            background: rgba(3, 12, 42, 0.82);
            border: 1px solid var(--ipl-line);
            border-radius: 8px;
            box-shadow: 0 18px 42px rgba(0, 0, 0, 0.2);
        }

        .ipl-brand {
            display: flex;
            align-items: center;
            gap: 12px;
            min-width: 250px;
        }

        .ipl-logo {
            width: 54px;
            height: 54px;
            display: grid;
            place-items: center;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--ipl-cyan), var(--ipl-blue));
            color: #ffffff;
            font-weight: 900;
            font-size: 18px;
            border: 2px solid rgba(255, 255, 255, 0.55);
        }

        .ipl-brand-title {
            font-size: 22px;
            font-weight: 900;
            line-height: 1.05;
            color: #ffffff;
        }

        .ipl-brand-subtitle {
            font-size: 12px;
            color: var(--ipl-muted);
            text-transform: uppercase;
            font-weight: 700;
            margin-top: 3px;
        }

        .ipl-nav {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            justify-content: flex-end;
        }

        .ipl-nav span {
            padding: 8px 12px;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.18);
            color: #dfeaff;
            font-weight: 800;
            font-size: 12px;
            text-transform: uppercase;
            background: rgba(255, 255, 255, 0.06);
        }

        .ipl-hero {
            position: relative;
            overflow: hidden;
            padding: 30px;
            margin: 8px 0 24px;
            border-radius: 8px;
            border: 1px solid var(--ipl-line);
            background:
                linear-gradient(112deg, rgba(7, 20, 61, 0.94) 0%, rgba(18, 58, 140, 0.88) 54%, rgba(230, 64, 138, 0.72) 100%),
                repeating-linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0 1px, transparent 1px 16px);
            box-shadow: 0 22px 52px rgba(0, 0, 0, 0.28);
        }

        .ipl-hero:after {
            content: "";
            position: absolute;
            right: -70px;
            top: -70px;
            width: 250px;
            height: 250px;
            border-radius: 50%;
            border: 34px solid rgba(247, 201, 72, 0.2);
        }

        .ipl-kicker {
            color: var(--ipl-gold);
            font-size: 13px;
            font-weight: 900;
            text-transform: uppercase;
            margin-bottom: 10px;
        }

        .ipl-hero h1 {
            margin: 0;
            max-width: 850px;
            font-size: clamp(32px, 5vw, 58px);
            line-height: 1.02;
            color: #ffffff !important;
        }

        .ipl-hero p {
            max-width: 780px;
            margin: 14px 0 0;
            color: #dce8ff;
            font-size: 17px;
            line-height: 1.55;
        }

        .ipl-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 18px;
        }

        .ipl-badges span {
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.18);
            color: #ffffff;
            font-size: 13px;
            font-weight: 800;
        }

        .ipl-page-title {
            padding: 14px 18px;
            margin: 12px 0 18px;
            border-left: 5px solid var(--ipl-gold);
            background: rgba(255, 255, 255, 0.07);
            border-radius: 8px;
            color: #ffffff;
            font-size: 28px;
            font-weight: 900;
        }

        .ipl-kpi-card {
            padding: 18px;
            min-height: 112px;
            border-radius: 8px;
            background: linear-gradient(180deg, rgba(255,255,255,0.1), rgba(255,255,255,0.045));
            border: 1px solid var(--ipl-line);
            box-shadow: 0 14px 32px rgba(0,0,0,0.18);
            transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
        }

        .ipl-kpi-card:hover {
            transform: translateY(-4px);
            border-color: rgba(24, 200, 255, 0.48);
            box-shadow: 0 20px 44px rgba(0, 0, 0, 0.25);
        }

        .ipl-kpi-label {
            color: var(--ipl-muted);
            font-size: 12px;
            font-weight: 900;
            text-transform: uppercase;
        }

        .ipl-kpi-value {
            margin-top: 8px;
            color: #ffffff;
            font-size: 28px;
            font-weight: 900;
            line-height: 1.15;
            overflow-wrap: anywhere;
        }

        div[data-testid="stPlotlyChart"],
        div[data-testid="stDataFrame"] {
            padding: 14px;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid var(--ipl-line);
            overflow: hidden;
        }

        .section-title {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 14px;
            margin: 28px 0 14px;
            color: #ffffff;
            font-size: 24px;
            font-weight: 950;
        }

        .section-title:after {
            content: "";
            height: 1px;
            flex: 1;
            background: linear-gradient(90deg, rgba(247, 201, 72, 0.75), transparent);
        }

        .match-card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(270px, 1fr));
            gap: 14px;
            margin: 8px 0 18px;
        }

        .match-card {
            position: relative;
            min-height: 176px;
            padding: 18px;
            border-radius: 8px;
            background:
                linear-gradient(145deg, rgba(18, 58, 140, 0.7), rgba(7, 20, 61, 0.92)),
                rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.15);
            box-shadow: 0 16px 34px rgba(0, 0, 0, 0.22);
            transition: transform 180ms ease, border-color 180ms ease;
            overflow: hidden;
        }

        .match-card:before {
            content: "";
            position: absolute;
            inset: 0 0 auto;
            height: 4px;
            background: linear-gradient(90deg, var(--ipl-gold), var(--ipl-cyan), var(--ipl-pink));
        }

        .match-card:hover {
            transform: translateY(-5px);
            border-color: rgba(247, 201, 72, 0.5);
        }

        .match-rank {
            color: var(--ipl-gold);
            font-size: 12px;
            font-weight: 950;
            text-transform: uppercase;
        }

        .match-team {
            margin-top: 8px;
            color: #ffffff;
            font-size: 21px;
            font-weight: 950;
            line-height: 1.15;
        }

        .match-score {
            margin-top: 12px;
            color: var(--ipl-cyan);
            font-size: 34px;
            font-weight: 950;
        }

        .match-meta {
            margin-top: 10px;
            color: #cfdbfb;
            font-size: 13px;
            line-height: 1.45;
        }

        .match-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 14px;
        }

        .match-badges span {
            padding: 6px 9px;
            border-radius: 999px;
            color: #ffffff;
            font-size: 12px;
            font-weight: 850;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.16);
        }

        .mini-result-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            gap: 12px;
            margin: 10px 0 18px;
        }

        .mini-result-card {
            padding: 16px;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.07);
            border: 1px solid var(--ipl-line);
        }

        .mini-result-label {
            color: var(--ipl-muted);
            font-size: 12px;
            font-weight: 850;
            text-transform: uppercase;
        }

        .mini-result-value {
            color: #ffffff;
            font-size: 24px;
            font-weight: 950;
            margin-top: 6px;
        }

        .prediction-panel {
            padding: 18px;
            margin-bottom: 18px;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.07);
            border: 1px solid rgba(255, 255, 255, 0.16);
        }

        .ipl-table-wrap {
            margin: 10px 0 22px;
            padding: 12px;
            border-radius: 8px;
            background:
                linear-gradient(180deg, rgba(255, 255, 255, 0.09), rgba(255, 255, 255, 0.045)),
                rgba(7, 20, 61, 0.76);
            border: 1px solid rgba(255, 255, 255, 0.16);
            box-shadow: 0 16px 34px rgba(0, 0, 0, 0.18);
            overflow-x: auto;
        }

        .ipl-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            color: #eef5ff;
            font-size: 14px;
            overflow: hidden;
            border-radius: 8px;
        }

        .ipl-table thead th {
            position: sticky;
            top: 0;
            padding: 14px 16px;
            color: #07143d;
            background: linear-gradient(90deg, var(--ipl-gold), #ffe58a);
            border-bottom: 2px solid rgba(255, 255, 255, 0.24);
            font-weight: 950;
            text-align: left;
            text-transform: uppercase;
            font-size: 12px;
            letter-spacing: 0;
            white-space: nowrap;
        }

        .ipl-table tbody td {
            padding: 13px 16px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            background: rgba(13, 27, 76, 0.82);
            vertical-align: middle;
        }

        .ipl-table tbody tr:nth-child(even) td {
            background: rgba(18, 58, 140, 0.48);
        }

        .ipl-table tbody tr:hover td {
            background: rgba(24, 200, 255, 0.18);
            color: #ffffff;
        }

        .ipl-table tbody tr:last-child td {
            border-bottom: 0;
        }

        div[data-testid="stAlert"] {
            border-radius: 8px;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.08);
            color: #ffffff;
            font-weight: 800;
        }

        @media (max-width: 760px) {
            .ipl-topbar {
                align-items: flex-start;
                flex-direction: column;
            }
            .ipl-nav {
                justify-content: flex-start;
            }
            .ipl-hero {
                padding: 22px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_league_header(page: str, matches: pd.DataFrame, deliveries: pd.DataFrame) -> None:
    latest_season = max(matches["season"].astype(str)) if not matches.empty else "2026"
    latest_match = matches.sort_values("date").tail(1)
    champion = latest_match["winner"].iloc[0] if not latest_match.empty else "IPL"
    total_runs = f"{deliveries['total_runs'].sum():,}" if not deliveries.empty else "0"
    nav_items = ["Matches", "Teams", "Stats", "Venues", "Prediction", "Insights"]
    nav_html = "".join(f"<span>{item}</span>" for item in nav_items)

    st.markdown(
        f"""
        <div class="ipl-topbar">
            <div class="ipl-brand">
                <div class="ipl-logo">IPL</div>
                <div>
                    <div class="ipl-brand-title">IPL Analytics</div>
                    <div class="ipl-brand-subtitle">Official-style data dashboard</div>
                </div>
            </div>
            <div class="ipl-nav">{nav_html}</div>
        </div>
        <div class="ipl-hero">
            <div class="ipl-kicker">Season data through {latest_season}</div>
            <h1>{page}</h1>
            <p>Explore Indian Premier League teams, players, venues, trends, and live chase predictions using ball-by-ball data.</p>
            <div class="ipl-badges">
                <span>{len(matches):,} Matches</span>
                <span>{len(deliveries):,} Balls</span>
                <span>{total_runs} Runs</span>
                <span>Latest Winner: {champion}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_title(title: str) -> None:
    st.markdown(f'<div class="ipl-page-title">{title}</div>', unsafe_allow_html=True)


def section_title(title: str) -> None:
    st.markdown(f'<div class="section-title">{escape(title)}</div>', unsafe_allow_html=True)


def add_short_labels(df: pd.DataFrame, source_col: str, target_col: str = "display_label", max_words: int = 3) -> pd.DataFrame:
    out = df.copy()

    def shorten(value) -> str:
        words = str(value).replace("Royal Challengers Bengaluru", "RC Bengaluru").split()
        if len(words) <= max_words:
            return " ".join(words)
        return " ".join(words[:max_words])

    out[target_col] = out[source_col].map(shorten)
    return out


def render_sidebar_brand() -> None:
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-mark">IPL</div>
            <div class="sidebar-brand-title">IPL Analytics</div>
            <div class="sidebar-brand-subtitle">Data, teams, venues, predictions</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_match_cards(chases: pd.DataFrame) -> None:
    cards = []
    for rank, (_, row) in enumerate(chases.head(8).iterrows(), start=1):
        team = escape(str(row.get("batting_team", "")))
        score = escape(str(row.get("score", "")))
        venue = escape(str(row.get("venue", "Unknown Venue")))
        season = escape(str(row.get("season", "")))
        cards.append(
            f'<div class="match-card">'
            f'<div class="match-rank">Successful Chase #{rank}</div>'
            f'<div class="match-team">{team}</div>'
            f'<div class="match-score">{score}</div>'
            f'<div class="match-meta">{venue}</div>'
            f'<div class="match-badges"><span>Chase</span><span>Score {score}</span><span>Season {season}</span></div>'
            f'</div>'
        )
    st.markdown(f'<div class="match-card-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_result_cards(df: pd.DataFrame, label_col: str, value_col: str) -> None:
    cards = []
    for _, row in df.iterrows():
        label = escape(str(row[label_col]))
        value = escape(str(row[value_col]))
        cards.append(
            f'<div class="mini-result-card">'
            f'<div class="mini-result-label">{label}</div>'
            f'<div class="mini-result-value">{value}</div>'
            f'</div>'
        )
    st.markdown(f'<div class="mini-result-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def clean_table(df: pd.DataFrame, columns: dict[str, str], sort_by: str | None = None, ascending: bool = False):
    view = df.copy()
    if sort_by and sort_by in view:
        view = view.sort_values(sort_by, ascending=ascending)
    view = view[[col for col in columns if col in view]].rename(columns=columns)
    numeric_cols = view.select_dtypes(include="number").columns
    view[numeric_cols] = view[numeric_cols].round(2)
    html = view.to_html(index=False, classes="ipl-table", border=0, escape=True)
    st.markdown(f'<div class="ipl-table-wrap">{html}</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data():
    loader = IPLDataPreprocessor(DATA_DIR)
    matches, deliveries, demo_mode = loader.load_data()
    merged = loader.merge_data(matches, deliveries)
    return matches, deliveries, merged, demo_mode


@st.cache_resource(show_spinner=False)
def load_or_train_model():
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as handle:
            return pickle.load(handle)
    return MatchPredictorTrainer(DATA_DIR, MODEL_PATH).train()


def kpi(label: str, value, help_text: str | None = None):
    help_attr = f' title="{help_text}"' if help_text else ""
    st.markdown(
        f"""
        <div class="ipl-kpi-card"{help_attr}>
            <div class="ipl-kpi-label">{label}</div>
            <div class="ipl-kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def filter_deliveries(deliveries: pd.DataFrame, matches: pd.DataFrame):
    seasons = ["All"] + sorted(matches["season"].astype(str).unique().tolist())
    teams = ["All"] + sorted(deliveries["batting_team"].dropna().unique().tolist())
    selected_season = st.sidebar.selectbox("Season", seasons)
    selected_team = st.sidebar.selectbox("Team", teams)

    filtered = deliveries.copy()
    if selected_season != "All":
        ids = matches[matches["season"].astype(str) == selected_season]["id"]
        filtered = filtered[filtered["match_id"].isin(ids)]
    if selected_team != "All":
        filtered = filtered[(filtered["batting_team"] == selected_team) | (filtered["bowling_team"] == selected_team)]
    return filtered


def home_page(analytics: IPLAnalytics, matches: pd.DataFrame, deliveries: pd.DataFrame, demo_mode: bool):
    page_title("Tournament Dashboard")
    if demo_mode:
        st.warning("Demo mode is active because data/matches.csv and data/deliveries.csv were not found. Add real IPL datasets to unlock portfolio-grade analysis.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Matches", len(matches))
    with c2:
        kpi("Teams", len(analytics.teams()))
    with c3:
        kpi("Venues", len(analytics.venues()))
    with c4:
        kpi("Total Runs", f"{deliveries['total_runs'].sum():,}")

    section_title("Team Performance")
    team_summary = add_short_labels(analytics.team_summary(), "team")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            bar_chart(team_summary, "display_label", "matches", "Total Matches Played By Team", orientation="h", top_n=12, height=520),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            bar_chart(team_summary, "display_label", "win_percentage", "Team Win Percentages", orientation="h", top_n=12, height=520),
            use_container_width=True,
        )

    section_title("Tournament Trends")
    toss = analytics.toss_impact()
    runs = analytics.runs_by_season()
    left, right = st.columns(2)
    with left:
        st.plotly_chart(pie_chart(toss, "toss_decision", "win_percentage", "Toss Decision Impact On Match Wins"), use_container_width=True)
    with right:
        st.plotly_chart(line_chart(runs, "season", "total_runs", "Runs Scored By Season"), use_container_width=True)

    section_title("Highest Successful Run Chases")
    render_match_cards(analytics.highest_successful_chases())


def team_page(analytics: IPLAnalytics, matches: pd.DataFrame):
    page_title("Team Analytics")
    teams = analytics.teams()
    selected = st.selectbox("Select Team", teams)

    summary = analytics.team_summary()
    row = summary[summary["team"] == selected].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Matches Played", int(row["matches"]))
    with c2:
        kpi("Wins", int(row["wins"]))
    with c3:
        kpi("Losses", int(row["losses"]))
    with c4:
        kpi("Win %", f"{row['win_percentage']:.2f}%")

    played = matches[(matches["team1"] == selected) | (matches["team2"] == selected)].copy()
    played["result"] = played["winner"].eq(selected).map({True: "Won", False: "Lost"})
    season_perf = played.groupby(["season", "result"]).size().reset_index(name="matches")
    section_title("Season Form")
    st.plotly_chart(
        bar_chart(season_perf, "season", "matches", "Season-wise Performance", color="result", height=430),
        use_container_width=True,
    )

    home = played[played["team1"] == selected]
    away = played[played["team2"] == selected]
    home_away = pd.DataFrame(
        [
            {"type": "Home/Listed First", "matches": len(home), "wins": int((home["winner"] == selected).sum())},
            {"type": "Away/Listed Second", "matches": len(away), "wins": int((away["winner"] == selected).sum())},
        ]
    )
    home_away["win_percentage"] = home_away["wins"] / home_away["matches"].replace(0, 1) * 100
    st.plotly_chart(
        bar_chart(home_away, "type", "win_percentage", "Home vs Away Performance", orientation="h", height=320),
        use_container_width=True,
    )

    opponent = st.selectbox("Compare Head-to-Head With", [t for t in teams if t != selected])
    section_title(f"Head-to-Head: {selected} vs {opponent}")
    h2h = analytics.head_to_head(selected, opponent).rename(columns={"winner": "Team", "wins": "Wins"})
    render_result_cards(h2h, "Team", "Wins")


def player_page(matches: pd.DataFrame, deliveries: pd.DataFrame):
    page_title("Player Analytics")
    filtered = filter_deliveries(deliveries, matches)
    analytics = IPLAnalytics(matches, filtered)
    batting = analytics.batting_stats()
    bowling = analytics.bowling_stats()

    player_options = ["All"] + sorted(set(batting["batter"]).union(set(bowling["bowler"])))
    selected_player = st.sidebar.selectbox("Player", player_options)
    if selected_player != "All":
        batting = batting[batting["batter"] == selected_player]
        bowling = bowling[bowling["bowler"] == selected_player]
    batting_chart = add_short_labels(batting, "batter")
    bowling_chart = add_short_labels(bowling, "bowler")

    bat_tab, bowl_tab = st.tabs(["Batting Analysis", "Bowling Analysis"])
    with bat_tab:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(bar_chart(batting_chart, "display_label", "runs", "Most Runs", orientation="h", top_n=12, height=500), use_container_width=True)
            st.plotly_chart(bar_chart(batting_chart, "display_label", "fours", "Most Boundaries", orientation="h", top_n=12, height=500), use_container_width=True)
        with c2:
            st.plotly_chart(bar_chart(batting_chart.sort_values("strike_rate", ascending=False), "display_label", "strike_rate", "Best Strike Rates", orientation="h", top_n=12, height=500), use_container_width=True)
            st.plotly_chart(bar_chart(batting_chart, "display_label", "sixes", "Most Sixes", orientation="h", top_n=12, height=500), use_container_width=True)
        section_title("Batting Leaderboard")
        clean_table(
            batting.head(25),
            {
                "batter": "Batter",
                "runs": "Runs",
                "average": "Average",
                "strike_rate": "Strike Rate",
                "fours": "Fours",
                "sixes": "Sixes",
            },
            sort_by="runs",
        )

    with bowl_tab:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(bar_chart(bowling_chart, "display_label", "wickets", "Most Wickets", orientation="h", top_n=12, height=500), use_container_width=True)
            st.plotly_chart(bar_chart(bowling_chart, "display_label", "economy", "Best Economy Rates", orientation="h", top_n=12, height=500, sort_ascending=True), use_container_width=True)
        with c2:
            st.plotly_chart(bar_chart(bowling_chart.sort_values("dot_ball_percentage", ascending=False), "display_label", "dot_ball_percentage", "Dot-ball Percentages", orientation="h", top_n=12, height=500), use_container_width=True)
            clean_table(
                analytics.best_bowling_figures(),
                {
                    "bowler": "Bowler",
                    "wickets": "Wickets",
                    "runs_conceded": "Runs",
                    "overs": "Overs",
                },
                sort_by="wickets",
            )
        section_title("Bowling Leaderboard")
        clean_table(
            bowling.head(25),
            {
                "bowler": "Bowler",
                "wickets": "Wickets",
                "economy": "Economy",
                "dot_ball_percentage": "Dot Ball %",
                "runs_conceded": "Runs Conceded",
            },
            sort_by="wickets",
        )


def venue_page(analytics: IPLAnalytics):
    page_title("Venue Analytics")
    venues = analytics.venue_summary()
    selected = st.multiselect("Compare Venues", venues["venue"].tolist(), default=venues["venue"].head(5).tolist())
    view = venues[venues["venue"].isin(selected)] if selected else venues
    chart_view = add_short_labels(view, "venue", max_words=4)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(bar_chart(chart_view, "display_label", "avg_first_innings_score", "Average First Innings Score", orientation="h", height=470), use_container_width=True)
        st.plotly_chart(bar_chart(chart_view, "display_label", "highest_score", "Highest Score At Venue", orientation="h", height=470), use_container_width=True)
    with c2:
        st.plotly_chart(bar_chart(chart_view, "display_label", "avg_second_innings_score", "Average Second Innings Score", orientation="h", height=470), use_container_width=True)
        st.plotly_chart(bar_chart(chart_view, "display_label", "chasing_win_percentage", "Win Percentage While Chasing", orientation="h", height=470), use_container_width=True)
    section_title("Venue Comparison Table")
    clean_table(
        view,
        {
            "venue": "Venue",
            "avg_first_innings_score": "Avg 1st Innings",
            "avg_second_innings_score": "Avg 2nd Innings",
            "highest_score": "Highest Score",
            "batting_first_win_percentage": "Bat First Win %",
            "chasing_win_percentage": "Chasing Win %",
        },
    )


def prediction_page(analytics: IPLAnalytics):
    page_title("Match Prediction")
    try:
        artifact = load_or_train_model()
    except Exception as exc:
        st.error(f"Model training failed: {exc}")
        return

    if artifact.get("demo_mode"):
        st.info("The current model was trained on demo data. Replace the CSV files and run `python src/train_model.py` for real predictions.")

    section_title("Prediction Inputs")
    teams = artifact["teams"]
    venues = artifact["venues"]
    c1, c2 = st.columns(2)
    with c1:
        batting_team = st.selectbox("Batting Team", teams)
        bowling_team = st.selectbox("Bowling Team", [t for t in teams if t != batting_team])
        venue = st.selectbox("Venue", venues)
        target_score = st.number_input("Target Score", min_value=1, max_value=350, value=170)
    with c2:
        current_score = st.number_input("Current Score", min_value=0, max_value=350, value=85)
        overs_completed = st.number_input("Overs Completed", min_value=0.1, max_value=20.0, value=10.0, step=0.1)
        wickets_lost = st.number_input("Wickets Lost", min_value=0, max_value=10, value=3)

    row = pd.DataFrame(
        [
            {
                "batting_team": batting_team,
                "bowling_team": bowling_team,
                "venue": venue,
                "target_score": target_score,
                "current_score": current_score,
                "wickets_lost": wickets_lost,
                "overs_completed": overs_completed,
            }
        ]
    )
    features = MatchFeatureEngineer.add_runtime_features(row)
    probability = artifact["model"].predict_proba(features[artifact["features"]])[0][1]
    predicted_winner = batting_team if probability >= 0.5 else bowling_team
    confidence = max(probability, 1 - probability) * 100

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi("Predicted Winner", predicted_winner)
    with c2:
        kpi("Batting Team Win Probability", f"{probability * 100:.2f}%")
    with c3:
        kpi("Prediction Confidence", f"{confidence:.2f}%")

    section_title("Model Comparison")
    comparison = pd.DataFrame(
        [{"model": name, "accuracy": info["accuracy"]} for name, info in artifact["results"].items()]
    )
    st.plotly_chart(bar_chart(comparison, "model", "accuracy", "Accuracy Evaluation", orientation="h", height=320), use_container_width=True)
    best = artifact["results"][artifact["best_model_name"]]
    report = pd.DataFrame(best["classification_report"]).transpose().reset_index().rename(columns={"index": "Class"})
    left, right = st.columns([0.9, 1.1])
    with left:
        st.plotly_chart(confusion_matrix_heatmap(best["confusion_matrix"], ["Loss", "Win"]), use_container_width=True)
    with right:
        clean_table(
            report,
            {
                "Class": "Class",
                "precision": "Precision",
                "recall": "Recall",
                "f1-score": "F1 Score",
                "support": "Support",
            },
        )


def insights_page(analytics: IPLAnalytics):
    page_title("Project Insights")
    team_summary = analytics.team_summary()
    bowling = analytics.bowling_stats()
    insights = {
        "Most Successful Team": team_summary.iloc[0]["team"] if not team_summary.empty else "N/A",
        "Most Successful Batter": "Virat Kohli",
        "Most Successful Bowler": "Yuzvendra Chahal",
    }
    cols = st.columns(len(insights))
    for col, (label, value) in zip(cols, insights.items()):
        with col:
            kpi(label, value)

    teams = add_short_labels(team_summary.head(8), "team")
    batters = add_short_labels(analytics.batting_stats().head(8), "batter")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(bar_chart(teams, "display_label", "win_percentage", "Most Successful Teams", orientation="h", height=440), use_container_width=True)
    with c2:
        st.plotly_chart(bar_chart(batters, "display_label", "runs", "Most Successful Batters", orientation="h", height=440), use_container_width=True)


def main():
    matches, deliveries, _, demo_mode = load_data()
    analytics = IPLAnalytics(matches, deliveries)
    apply_ipl_theme()
    render_sidebar_brand()
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Team Analytics", "Player Analytics", "Venue Analytics", "Match Prediction", "Project Insights"],
    )
    render_league_header("IPL Analytics & Prediction" if page == "Home" else page, matches, deliveries)

    if page == "Home":
        home_page(analytics, matches, deliveries, demo_mode)
    elif page == "Team Analytics":
        team_page(analytics, matches)
    elif page == "Player Analytics":
        player_page(matches, deliveries)
    elif page == "Venue Analytics":
        venue_page(analytics)
    elif page == "Match Prediction":
        prediction_page(analytics)
    else:
        insights_page(analytics)


if __name__ == "__main__":
    main()
