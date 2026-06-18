"""
charts.py
---------
Plotly chart builders for the Streamlit dashboard.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


VERDICT_COLORS = {
    "Verified": "#22c55e",
    "Inaccurate": "#f59e0b",
    "False": "#ef4444",
}


def pie_chart(summary: dict) -> go.Figure:
    """Donut chart showing verdict distribution."""
    labels = ["Verified", "Inaccurate", "False"]
    values = [summary["verified"], summary["inaccurate"], summary["false"]]
    colors = [VERDICT_COLORS[l] for l in labels]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker=dict(colors=colors, line=dict(color="#1e293b", width=2)),
            textinfo="label+percent",
            textfont=dict(size=13),
            hovertemplate="%{label}: %{value} claims<extra></extra>",
        )
    )
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
    )
    return fig


def confidence_bar_chart(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of confidence scores per claim."""
    df_sorted = df.sort_values("Confidence", ascending=True).tail(15)

    colors = [VERDICT_COLORS.get(v, "#94a3b8") for v in df_sorted["Verdict"]]

    # Truncate long claim text for display
    labels = [c[:60] + "…" if len(c) > 60 else c for c in df_sorted["Claim"]]

    fig = go.Figure(
        go.Bar(
            x=df_sorted["Confidence"],
            y=labels,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{c:.0%}" for c in df_sorted["Confidence"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Confidence: %{x:.0%}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis=dict(range=[0, 1.1], tickformat=".0%", title="Confidence Score"),
        yaxis=dict(title=""),
        margin=dict(t=10, b=40, l=10, r=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        height=max(300, len(df_sorted) * 38),
    )
    return fig


def category_breakdown_chart(df: pd.DataFrame) -> go.Figure:
    """Stacked bar chart of verdicts by claim category."""
    if df.empty:
        return go.Figure()

    grouped = (
        df.groupby(["Category", "Verdict"])
        .size()
        .reset_index(name="Count")
    )

    fig = px.bar(
        grouped,
        x="Category",
        y="Count",
        color="Verdict",
        barmode="stack",
        color_discrete_map=VERDICT_COLORS,
        labels={"Count": "Number of Claims", "Category": "Claim Category"},
    )
    fig.update_layout(
        margin=dict(t=10, b=40, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
    )
    return fig
