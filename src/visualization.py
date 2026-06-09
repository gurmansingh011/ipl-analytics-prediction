"""Plotly visualization helpers."""

import plotly.express as px
import plotly.graph_objects as go


IPL_COLORS = [
    "#18c8ff",
    "#f7c948",
    "#e6408a",
    "#31d0aa",
    "#7c5cff",
    "#ff8a3d",
    "#4ea1ff",
    "#f95f62",
]


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


def label_for(name):
    if not name:
        return ""
    return LABEL_OVERRIDES.get(str(name), str(name).replace("_", " ").title())


def labels_for(*names):
    return {name: label_for(name) for name in names if name}


def hover_label(name):
    label = label_for(name)
    return label if label else "Name"


def apply_chart_theme(fig, height=430, showlegend=False, margin=None):
    chart_margin = margin or {"l": 20, "r": 24, "t": 62, "b": 34}
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,20,61,0.25)",
        font={"family": "Inter, Segoe UI, sans-serif", "color": "#dfeaff"},
        title={"font": {"size": 18, "color": "#ffffff"}, "x": 0.02, "xanchor": "left"},
        margin=chart_margin,
        height=height,
        colorway=IPL_COLORS,
        showlegend=showlegend,
    )
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.10)",
        zerolinecolor="rgba(255,255,255,0.16)",
        title_font={"color": "#8ea4d2"},
        tickfont={"color": "#c7d6f7", "size": 11},
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.08)",
        zerolinecolor="rgba(255,255,255,0.16)",
        title_font={"color": "#8ea4d2"},
        tickfont={"color": "#c7d6f7", "size": 11},
    )
    return fig


def bar_chart(df, x, y, title, color=None, orientation="v", height=430, top_n=None, sort_ascending=False):
    chart_df = df.copy()
    if top_n:
        chart_df = chart_df.sort_values(y, ascending=sort_ascending).head(top_n)

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
            orientation="v",
            text=y,
            labels=labels_for(x, y, color),
        )
        fig.update_traces(texttemplate="%{y:.3s}", textposition="outside", cliponaxis=False)
        fig.update_traces(hovertemplate=f"{hover_label(x)}: %{{x}}<br>{hover_label(y)}: %{{y}}<extra></extra>")
        fig.update_xaxes(tickangle=0)
        margin = {"l": 24, "r": 42, "t": 62, "b": 42}

    fig.update_layout(barmode="group" if color else "relative")
    fig = apply_chart_theme(fig, height=height, showlegend=bool(color and color != x), margin=margin)
    if orientation == "h":
        fig.update_xaxes(title_text=label_for(y))
        fig.update_yaxes(title_text="")
    else:
        fig.update_xaxes(title_text=label_for(x))
        fig.update_yaxes(title_text=label_for(y))
    return fig


def line_chart(df, x, y, title, color=None):
    fig = px.line(df, x=x, y=y, color=color, markers=True, title=title, labels=labels_for(x, y, color))
    fig.update_traces(line={"width": 3}, marker={"size": 8})
    fig.update_traces(hovertemplate=f"{hover_label(x)}: %{{x}}<br>{hover_label(y)}: %{{y}}<extra></extra>")
    fig = apply_chart_theme(fig, height=420, showlegend=bool(color))
    fig.update_xaxes(title_text=label_for(x))
    fig.update_yaxes(title_text=label_for(y))
    return fig


def pie_chart(df, names, values, title):
    fig = px.pie(df, names=names, values=values, title=title, hole=0.48, color_discrete_sequence=IPL_COLORS, labels=labels_for(names, values))
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        marker={"line": {"color": "#07143d", "width": 2}},
        hovertemplate=f"{hover_label(names)}: %{{label}}<br>{hover_label(values)}: %{{value}}<br>Share: %{{percent}}<extra></extra>",
    )
    return apply_chart_theme(fig, height=420, showlegend=False)


def confusion_matrix_heatmap(matrix, labels, title="Confusion Matrix"):
    fig = go.Figure(data=go.Heatmap(z=matrix, x=labels, y=labels, colorscale="Blues", text=matrix, texttemplate="%{text}"))
    fig.update_layout(title=title, xaxis_title="Predicted", yaxis_title="Actual")
    return apply_chart_theme(fig, height=380)
