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
            hover_data=chart_df.columns,
        )
        fig.update_traces(texttemplate="%{x:.3s}", textposition="outside", cliponaxis=False)
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
            hover_data=chart_df.columns,
        )
        fig.update_traces(texttemplate="%{y:.3s}", textposition="outside", cliponaxis=False)
        fig.update_xaxes(tickangle=0)
        margin = {"l": 24, "r": 42, "t": 62, "b": 42}

    fig.update_layout(barmode="group" if color else "relative")
    return apply_chart_theme(fig, height=height, showlegend=bool(color and color != x), margin=margin)


def line_chart(df, x, y, title, color=None):
    fig = px.line(df, x=x, y=y, color=color, markers=True, title=title)
    fig.update_traces(line={"width": 3}, marker={"size": 8})
    return apply_chart_theme(fig, height=420, showlegend=bool(color))


def pie_chart(df, names, values, title):
    fig = px.pie(df, names=names, values=values, title=title, hole=0.48, color_discrete_sequence=IPL_COLORS)
    fig.update_traces(textposition="inside", textinfo="percent+label", marker={"line": {"color": "#07143d", "width": 2}})
    return apply_chart_theme(fig, height=420, showlegend=False)


def confusion_matrix_heatmap(matrix, labels, title="Confusion Matrix"):
    fig = go.Figure(data=go.Heatmap(z=matrix, x=labels, y=labels, colorscale="Blues", text=matrix, texttemplate="%{text}"))
    fig.update_layout(title=title, xaxis_title="Predicted", yaxis_title="Actual")
    return apply_chart_theme(fig, height=380)
