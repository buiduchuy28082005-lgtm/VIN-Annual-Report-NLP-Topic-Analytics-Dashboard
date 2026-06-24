from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

ART = Path("dashboard_artifacts")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Topic Dashboard (Theme)", layout="wide")

# =========================
# CSS THEME (DARK + PRO SIDEBAR)
# =========================
st.markdown("""
<style>
/* ===== Global ===== */
.stApp{
  background: radial-gradient(1200px 800px at 15% 10%, rgba(244,63,94,0.10), transparent 55%),
              radial-gradient(900px 600px at 85% 25%, rgba(59,130,246,0.10), transparent 55%),
              #0B1220;
  color: rgba(255,255,255,0.92);
}
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1350px; }
h1 { letter-spacing: .2px; }
.small-note { opacity: 0.78; font-size: 0.92rem; }

/* ===== Sidebar base ===== */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02));
  border-right: 1px solid rgba(255,255,255,0.10);
}
section[data-testid="stSidebar"] > div{ padding-top: 0.9rem; }
section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3{ color: rgba(255,255,255,0.92); }
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"]{
  font-weight: 720;
  color: rgba(255,255,255,0.90);
  margin-bottom: 0.15rem;
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label{
  color: rgba(255,255,255,0.82);
  font-size: 0.95rem;
}

/* ===== Sidebar "cards" (container border=True) ===== */
section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"]{
  background: rgba(255,255,255,0.035);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  padding: 12px 12px 8px 12px;
  box-shadow: 0 10px 22px rgba(0,0,0,0.25);
  margin-bottom: 12px;
}

/* ===== Sidebar section title ===== */
.sb-title{
  font-size: 0.95rem;
  font-weight: 800;
  letter-spacing: 0.2px;
  color: rgba(255,255,255,0.92);
  margin: 0 0 6px 0;
}
.sb-hint{
  font-size: 0.88rem;
  color: rgba(255,255,255,0.62);
  margin-top: -2px;
  margin-bottom: 8px;
}

/* ===== Multiselect tags (chips) ===== */
section[data-testid="stSidebar"] div[data-baseweb="tag"],
section[data-testid="stSidebar"] span[data-baseweb="tag"]{
  background: rgba(244,63,94,0.18) !important;
  border: 1px solid rgba(244,63,94,0.28) !important;
  color: rgba(255,255,255,0.92) !important;
  border-radius: 999px !important;
}
section[data-testid="stSidebar"] div[data-baseweb="tag"] svg{
  opacity: 0.85 !important;
}

/* ===== Slider look (subtle) ===== */
section[data-testid="stSidebar"] div[data-baseweb="slider"]{
  padding-top: 0.15rem;
}
section[data-testid="stSidebar"] div[data-baseweb="slider"] *{
  font-size: 0.92rem;
}

/* ===== Tabs ===== */
.stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 650; padding: 10px 14px; }

/* ===== Metrics & DataFrame (unchanged style, slightly polished) ===== */
[data-testid="stMetric"] {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.10);
  padding: 12px 14px; border-radius: 12px;
}
div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* ===== Buttons ===== */
.stDownloadButton button,
.stButton button{
  border-radius: 12px !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  font-weight: 700 !important;
}
.stDownloadButton button:hover,
.stButton button:hover{
  border-color: rgba(255,255,255,0.22) !important;
  transform: translateY(-1px);
}

/* ===== Plotly container border (optional, subtle) ===== */
div[data-testid="stPlotlyChart"]{
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  padding: 10px 10px 2px 10px;
  background: rgba(255,255,255,0.02);
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOADERS
# =========================
@st.cache_data(show_spinner=False)
def load_segments():
    pq = ART / "segments.parquet"
    csv = ART / "segments.csv"

    if pq.exists():
        df = pd.read_parquet(pq)
    elif csv.exists():
        try:
            df = pd.read_csv(csv, encoding="utf-8-sig")
        except Exception:
            df = pd.read_csv(csv, encoding="utf-8")
    else:
        raise FileNotFoundError("Không tìm thấy segments.parquet/segments.csv trong dashboard_artifacts.")

    required = {"Year", "topic_id", "topic_prob", "theme", "SourceFile", "text_clean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu các cột bắt buộc: {missing}. Hãy kiểm tra file export artifacts.")

    df["Year"] = df["Year"].astype(int)
    df["topic_id"] = df["topic_id"].astype(int)
    df["topic_prob"] = pd.to_numeric(df["topic_prob"], errors="coerce")
    df["theme"] = df["theme"].astype(str)
    df["SourceFile"] = df["SourceFile"].astype(str)
    df["text_clean"] = df["text_clean"].astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_keywords():
    p = ART / "topic_keywords.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


segments = load_segments()
keywords = load_keywords()

# =========================
# HELPERS
# =========================
def short_label(s: str, max_len: int = 42) -> str:
    s = str(s)
    return s if len(s) <= max_len else s[:max_len - 1] + "…"


def theme_shares(dfin: pd.DataFrame, exclude_outlier: bool = True) -> pd.DataFrame:
    d = dfin.copy()
    if exclude_outlier:
        d = d[d["topic_id"] != -1]
    g = d.groupby(["Year", "theme"]).size().reset_index(name="n")
    g["share"] = g["n"] / g.groupby("Year")["n"].transform("sum")
    return g


def build_topN_pivot(shares_long: pd.DataFrame, top_n: int = 8, keep_others: bool = True):
    avg = shares_long.groupby("theme")["share"].mean().sort_values(ascending=False)
    top_themes = avg.head(top_n).index.tolist()

    tmp = shares_long.copy()
    if keep_others:
        tmp["theme_plot"] = tmp["theme"].where(tmp["theme"].isin(top_themes), "Others")
        tmp = tmp.groupby(["Year", "theme_plot"])["share"].sum().reset_index()
        pivot = tmp.pivot_table(index="Year", columns="theme_plot", values="share", fill_value=0).sort_index()
        col_order = [t for t in top_themes if t in pivot.columns] + (["Others"] if "Others" in pivot.columns else [])
        pivot = pivot[col_order]
    else:
        tmp = tmp[tmp["theme"].isin(top_themes)]
        pivot = tmp.pivot_table(index="Year", columns="theme", values="share", fill_value=0).sort_index()
        pivot = pivot[top_themes]

    rename_map = {c: short_label(c) for c in pivot.columns}
    pivot_plot = pivot.rename(columns=rename_map)

    return pivot_plot, avg


def compute_trend_table(shares_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for theme, sub in shares_long.groupby("theme"):
        sub = sub.sort_values("Year")
        x = sub["Year"].values.astype(float)
        y = sub["share"].values.astype(float)
        if len(x) < 2:
            continue

        slope, intercept = np.polyfit(x, y, 1)
        yhat = slope * x + intercept
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        delta = float(y[-1] - y[0])

        rows.append({
            "theme": theme,
            "share_first": float(y[0]),
            "share_last": float(y[-1]),
            "delta": delta,
            "slope_per_year": float(slope),
            "r2": float(r2) if not np.isnan(r2) else np.nan
        })

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("slope_per_year", ascending=False)
    return out


def _sidebar_card():
    """
    Streamlit mới có container(border=True). Nếu bản Streamlit cũ không hỗ trợ,
    fallback về container() để app không lỗi (UI vẫn ổn nhờ CSS tổng).
    """
    try:
        return st.sidebar.container(border=True)
    except TypeError:
        return st.sidebar.container()

# =========================
# HEADER
# =========================
st.title("Annual Report Topic Dashboard (Theme-level)")


# =========================
# SIDEBAR FILTERS (PRO LAYOUT)
# =========================
st.sidebar.markdown("## Filters")

years = sorted(segments["Year"].unique().tolist())

# Card 1: Time window
with _sidebar_card():
    st.markdown('<div class="sb-title">Time window</div><div class="sb-hint">Chọn giai đoạn phân tích</div>', unsafe_allow_html=True)
    year_range = st.slider("Year range", min(years), max(years), (min(years), max(years)), 1)
    exclude_outlier = st.checkbox("Exclude outlier topic (-1) in shares", value=True)

# Default theme selection: top themes (to reduce clutter)
all_themes = sorted(segments["theme"].dropna().unique().tolist())
shares_all = theme_shares(
    segments[(segments["Year"] >= year_range[0]) & (segments["Year"] <= year_range[1])],
    exclude_outlier=exclude_outlier
)
avg_all = shares_all.groupby("theme")["share"].mean().sort_values(ascending=False)
default_top = avg_all.head(12).index.tolist() if len(avg_all) else all_themes

# Card 2: Theme selection
with _sidebar_card():
    theme_mode = st.radio("Theme selection default", ["Top themes", "All themes"], index=0)
    default_sel = default_top if theme_mode == "Top themes" else all_themes
    theme_sel = st.multiselect("Themes", all_themes, default=default_sel)

# Card 3: Evidence threshold
with _sidebar_card():
    st.markdown('<div class="sb-title">Evidence threshold</div><div class="sb-hint">Lọc mức tin cậy cho Evidence/Search</div>', unsafe_allow_html=True)
    min_prob = st.slider("Min topic_prob (for evidence/search)", 0.0, 1.0, 0.0, 0.05)

# Apply filters (unchanged logic)
df = segments[
    (segments["Year"] >= year_range[0]) &
    (segments["Year"] <= year_range[1]) &
    (segments["theme"].isin(theme_sel)) &
    (segments["topic_prob"].fillna(1.0) >= min_prob)
].copy()

# =========================
# KPIs
# =========================
c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1.2])
c1.metric("Segments", f"{len(df):,}")
c2.metric("Themes", f"{df['theme'].nunique()}")
outlier_rate = (df["topic_id"] == -1).mean() if len(df) else 0
c3.metric("Outlier rate (-1)", f"{outlier_rate:.2%}")
c4.metric("Years", f"{year_range[0]}–{year_range[1]}")
with c5:
    st.download_button(
        "Download filtered (CSV)",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="filtered_segments.csv",
        mime="text/csv",
        use_container_width=True
    )

tabs = st.tabs(["Overview", "Explorer", "Search"])

# =========================
# TAB 1: OVERVIEW
# =========================
with tabs[0]:
    st.subheader("Theme share over time")

    if df.empty:
        st.warning("Không có dữ liệu sau lọc.")
    else:
        colA, colB, colC, colD = st.columns([1, 1, 1, 2])
        with colA:
            top_n = st.slider("Top N themes (chart)", 3, 15, 8, 1)
        with colB:
            keep_others = st.checkbox("Group remaining as Others", value=True)
        with colC:
            chart_height = st.slider("Chart height", 420, 820, 560, 20)
        with colD:
            view_mode = st.radio("View", ["Stacked area", "Line trends"], horizontal=True)

        shares = theme_shares(df, exclude_outlier=exclude_outlier)

        if shares.empty:
            st.warning("Không có dữ liệu share sau khi exclude outlier/filters.")
        else:
            pivot_plot, avg = build_topN_pivot(shares, top_n=top_n, keep_others=keep_others)

            if view_mode == "Stacked area":
                fig = px.area(
                    pivot_plot,
                    x=pivot_plot.index,
                    y=pivot_plot.columns,
                    template="plotly_white",
                    title="Theme share by year (Top N + Others)"
                )
            else:
                fig = px.line(
                    pivot_plot,
                    x=pivot_plot.index,
                    y=pivot_plot.columns,
                    template="plotly_white",
                    markers=True,
                    title="Theme trends by year (Top N + Others)"
                )

            fig.update_layout(
                height=chart_height,
                hovermode="x unified",
                legend_title_text="Theme",
                legend=dict(orientation="h", y=-0.25, x=0),
                margin=dict(l=60, r=20, t=70, b=90),
                font=dict(size=14),
            )
            fig.update_xaxes(dtick=1, showgrid=True)
            fig.update_yaxes(tickformat=".0%", showgrid=True)

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": True, "toImageButtonOptions": {"format": "png", "scale": 2}}
            )

            st.subheader("Heatmap Year × Theme (share)")
            cols_sorted = sorted(pivot_plot.columns, key=lambda c: pivot_plot[c].mean(), reverse=True)
            heat = pivot_plot[cols_sorted].T

            fig2 = px.imshow(
                heat,
                aspect="auto",
                template="plotly_white",
                text_auto=".2f",
                labels=dict(x="Year", y="Theme", color="Share"),
                title="Theme share heatmap (values shown)"
            )
            fig2.update_layout(
                height=min(720, 140 + 36 * len(heat.index)),
                margin=dict(l=170, r=20, t=70, b=50),
                font=dict(size=13),
            )
            fig2.update_xaxes(dtick=1)
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Top themes per year (table)")
            topk = st.slider("Top K (table)", 3, 10, 5, 1)
            top_tbl = (
                shares.sort_values(["Year", "share"], ascending=[True, False])
                      .groupby("Year").head(topk)
            )
            top_tbl["theme"] = top_tbl["theme"].map(lambda x: short_label(x, 80))
            st.dataframe(top_tbl, use_container_width=True)

            st.subheader("Top rising / falling (THEME trend)")
            trend = compute_trend_table(shares)
            if trend.empty:
                st.info("Chưa đủ dữ liệu để tính trend.")
            else:
                trend_show = trend.copy()
                trend_show["theme"] = trend_show["theme"].map(lambda x: short_label(x, 90))
                cL, cR = st.columns(2)
                with cL:
                    st.markdown("**Top Rising**")
                    st.dataframe(trend_show.head(8), use_container_width=True)
                with cR:
                    st.markdown("**Top Falling**")
                    st.dataframe(trend_show.sort_values("slope_per_year").head(8), use_container_width=True)

# =========================
# TAB 2: EXPLORER
# =========================
with tabs[1]:
    st.subheader("Theme → Topic → Evidence (text_clean)")

    if df.empty:
        st.warning("Không có dữ liệu sau lọc.")
    else:
        theme_pick = st.selectbox("Pick a theme", sorted(df["theme"].unique().tolist()))
        dft = df[df["theme"] == theme_pick].copy()

        st.markdown("**Theme share by year (selected theme)**")
        theme_only = theme_shares(df, exclude_outlier=exclude_outlier)
        theme_line = theme_only[theme_only["theme"] == theme_pick].sort_values("Year")
        if len(theme_line):
            fig_t = px.line(theme_line, x="Year", y="share", template="plotly_white", markers=True)
            fig_t.update_layout(height=260, margin=dict(l=40, r=20, t=10, b=40))
            fig_t.update_yaxes(tickformat=".0%")
            fig_t.update_xaxes(dtick=1)
            st.plotly_chart(fig_t, use_container_width=True)
        else:
            st.info("Theme này không có share (do filters hoặc exclude outlier).")

        st.markdown("**Topic distribution inside this theme**")
        topic_counts = (
            dft[dft["topic_id"] != -1]
            .groupby("topic_id").size().reset_index(name="n")
            .sort_values("n", ascending=False)
        )
        st.dataframe(topic_counts, use_container_width=True)

        topic_pick = st.selectbox("Pick a topic_id", topic_counts["topic_id"].tolist() if len(topic_counts) else [-1])

        st.markdown("**Topic keywords (c-TF-IDF)**")
        kw = keywords.get(str(topic_pick)) or keywords.get(int(topic_pick))
        if kw:
            st.write(", ".join([x["word"] for x in kw[:18]]))
        else:
            st.info("Không có keywords cho topic này (hoặc topic = -1).")

        st.markdown("**Representative segments (highest topic_prob)**")
        ev = (
            dft[dft["topic_id"] == topic_pick]
            .sort_values("topic_prob", ascending=False)
            .head(15)
        )

        if ev.empty:
            st.info("Không có segment cho topic này sau filters.")
        else:
            for _, r in ev.iterrows():
                prob_val = r["topic_prob"]
                prob_str = f"{prob_val:.3f}" if pd.notna(prob_val) else "NA"
                st.markdown(
                    f"""
                    <div style="padding:12px 14px; border:1px solid rgba(255,255,255,0.10);
                                border-radius:12px; margin-bottom:10px;">
                        <div style="font-size:0.92rem; opacity:0.85;">
                            <b>Year:</b> {r['Year']} &nbsp; | &nbsp;
                            <b>Theme:</b> {short_label(r['theme'], 80)} &nbsp; | &nbsp;
                            <b>Topic:</b> {r['topic_id']} &nbsp; | &nbsp;
                            <b>Prob:</b> {prob_str} &nbsp; | &nbsp;
                            <b>File:</b> {short_label(r['SourceFile'], 60)}
                        </div>
                        <div style="margin-top:8px; font-size:1.03rem; line-height:1.55;">
                            {r['text_clean']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# =========================
# TAB 3: SEARCH
# =========================
with tabs[2]:
    st.subheader("Search in text_clean")

    if df.empty:
        st.warning("Không có dữ liệu sau lọc.")
    else:
        q = st.text_input("Keyword / phrase", "")
        if q.strip():
            mask = df["text_clean"].str.contains(q, case=False, na=False)
            res = df[mask].copy()

            st.write(f"Found **{len(res)}** segments.")
            res = res.sort_values(["Year", "topic_prob"], ascending=[True, False]).head(200)

            st.download_button(
                "Download search results (CSV)",
                data=res.to_csv(index=False).encode("utf-8-sig"),
                file_name="search_results.csv",
                mime="text/csv"
            )

            show = res[["Year", "theme", "topic_id", "topic_prob", "SourceFile", "text_clean"]].copy()
            show["theme"] = show["theme"].map(lambda x: short_label(x, 80))
            show["SourceFile"] = show["SourceFile"].map(lambda x: short_label(x, 60))
            st.dataframe(show, use_container_width=True)
        else:
            st.info("Nhập keyword để tìm trong text_clean và xem theme/topic tương ứng.")
