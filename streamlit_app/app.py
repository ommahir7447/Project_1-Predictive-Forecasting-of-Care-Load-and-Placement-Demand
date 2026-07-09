import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="UAC Forecasting Intelligence",
    page_icon="assets/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #080b14;
    color: #e2e8f0;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #2d3748; border-radius: 2px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1a2236 !important;
    padding-top: 0 !important;
    width: 280px !important;
}
section[data-testid="stSidebar"] > div { padding: 0 !important; }
section[data-testid="stSidebar"] * { color: #94a3b8 !important; }
section[data-testid="stSidebar"] .stCheckbox label { font-size: 0.82rem !important; letter-spacing: 0.01em; }
section[data-testid="stSidebar"] .stSlider > div > div > div { background: #334155 !important; }

/* ── Main content padding ── */
.main .block-container { padding: 0 2rem 2rem 2rem !important; max-width: 100% !important; }

/* ── Nav bar ── */
.topbar {
    position: sticky; top: 0; z-index: 999;
    background: rgba(8,11,20,0.92);
    backdrop-filter: blur(24px) saturate(180%);
    -webkit-backdrop-filter: blur(24px) saturate(180%);
    border-bottom: 1px solid #1a2236;
    padding: 0 2rem;
    display: flex; align-items: center; justify-content: space-between;
    height: 56px; margin: 0 -2rem 2rem -2rem;
}
.topbar-brand {
    font-size: 0.9rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #f1f5f9;
}
.topbar-brand span { color: #6366f1; }
.topbar-meta {
    display: flex; gap: 1.5rem; align-items: center;
}
.topbar-badge {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; padding: 0.25rem 0.7rem;
    border-radius: 100px; border: 1px solid;
}
.badge-live   { color: #34d399; border-color: rgba(52,211,153,0.3); background: rgba(52,211,153,0.08); }
.badge-models { color: #818cf8; border-color: rgba(129,140,248,0.3); background: rgba(129,140,248,0.08); }
.topbar-date  { font-size: 0.72rem; color: #475569; font-family: 'JetBrains Mono', monospace; }

/* ── KPI strip ── */
.kpi-strip {
    display: grid; grid-template-columns: repeat(5,1fr);
    gap: 1px; background: #1a2236;
    border: 1px solid #1a2236; border-radius: 16px;
    overflow: hidden; margin-bottom: 2rem;
}
.kpi-cell {
    background: #0d1117; padding: 1.4rem 1.5rem;
    position: relative; overflow: hidden;
    transition: background 0.2s;
}
.kpi-cell:hover { background: #111827; }
.kpi-cell::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
}
.kpi-c0::before { background: linear-gradient(90deg,#6366f1,#8b5cf6); }
.kpi-c1::before { background: linear-gradient(90deg,#10b981,#34d399); }
.kpi-c2::before { background: linear-gradient(90deg,#f43f5e,#fb7185); }
.kpi-c3::before { background: linear-gradient(90deg,#f59e0b,#fcd34d); }
.kpi-c4::before { background: linear-gradient(90deg,#06b6d4,#67e8f9); }
.kpi-sup  { font-size: 0.65rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: #475569; margin-bottom: 0.6rem; }
.kpi-num  { font-size: 2rem; font-weight: 800; line-height: 1; margin-bottom: 0.3rem; font-variant-numeric: tabular-nums; }
.kpi-c0 .kpi-num { color: #818cf8; }
.kpi-c1 .kpi-num { color: #34d399; }
.kpi-c2 .kpi-num { color: #fb7185; }
.kpi-c3 .kpi-num { color: #fcd34d; }
.kpi-c4 .kpi-num { color: #67e8f9; }
.kpi-sub  { font-size: 0.72rem; color: #475569; }

/* ── Section label ── */
.section-label {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: #475569;
    margin-bottom: 0.75rem; padding-bottom: 0.5rem;
    border-bottom: 1px solid #1a2236;
    display: flex; align-items: center; gap: 0.5rem;
}
.section-label::before {
    content: ''; display: inline-block; width: 3px; height: 12px;
    border-radius: 2px; background: #6366f1;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1a2236 !important;
    gap: 0 !important; padding: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important; border-radius: 0 !important;
    color: #475569 !important; font-size: 0.78rem !important;
    font-weight: 600 !important; letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    padding: 0.75rem 1.5rem !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.15s !important;
}
.stTabs [aria-selected="true"] {
    color: #818cf8 !important;
    border-bottom: 2px solid #6366f1 !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem !important; }

/* ── Sidebar internals ── */
.sb-section {
    padding: 1rem 1.25rem 0.75rem;
    border-bottom: 1px solid #1a2236;
}
.sb-section-label {
    font-size: 0.6rem; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: #334155;
    margin-bottom: 0.75rem;
}
.sb-brand {
    padding: 1.25rem 1.25rem 1rem;
    border-bottom: 1px solid #1a2236;
}
.sb-brand-name {
    font-size: 0.85rem; font-weight: 800; letter-spacing: 0.06em;
    text-transform: uppercase; color: #f1f5f9;
}
.sb-brand-name span { color: #6366f1; }
.sb-brand-sub { font-size: 0.68rem; color: #334155; margin-top: 0.2rem; }

/* ── Alerts / info boxes ── */
.info-box {
    background: #111827; border: 1px solid #1e293b;
    border-radius: 10px; padding: 0.75rem 1rem;
    font-size: 0.82rem; color: #64748b;
    margin: 0.5rem 0;
}
.warn-box {
    background: rgba(244,63,94,0.06);
    border: 1px solid rgba(244,63,94,0.2);
    border-left: 3px solid #f43f5e;
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 1rem; font-size: 0.8rem; color: #fb7185;
    margin: 0.3rem 0;
}
.success-box {
    background: rgba(16,185,129,0.06);
    border: 1px solid rgba(16,185,129,0.2);
    border-left: 3px solid #10b981;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem; font-size: 0.82rem; color: #34d399;
    margin: 0.5rem 0;
}
.scenario-row {
    background: #0d1117; border: 1px solid #1a2236;
    border-left: 3px solid;
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.2rem; margin: 0.35rem 0;
    display: flex; justify-content: space-between; align-items: center;
}
.scenario-horizon { font-size: 0.72rem; color: #475569; font-family: 'JetBrains Mono', monospace; }
.scenario-model   { font-size: 0.85rem; font-weight: 600; color: #e2e8f0; }
.scenario-acc     { font-size: 1rem; font-weight: 800; font-variant-numeric: tabular-nums; }

/* ── Dataframe ── */
.stDataFrame { border-radius: 10px !important; border: 1px solid #1a2236 !important; overflow: hidden; }
iframe[title="st.dataframe"] { background: #0d1117 !important; }

/* ── Divider ── */
hr { border: none !important; border-top: 1px solid #1a2236 !important; margin: 1.5rem 0 !important; }

/* ── Footer ── */
.footer {
    text-align: center; padding: 2rem 0 1rem;
    font-size: 0.68rem; color: #1e293b;
    letter-spacing: 0.06em; text-transform: uppercase;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden !important; }
.stDeployButton { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Charting constants ────────────────────────────────────────────────────────
BG       = 'rgba(0,0,0,0)'
GRID     = 'rgba(26,34,54,1)'
TICK     = '#475569'
HOVER_BG = 'rgba(13,17,23,0.97)'

MODEL_COLORS = {
    'Naïve Persistence': '#f43f5e',
    'Moving Average':    '#f59e0b',
    'ARIMA':             '#6366f1',
    'SARIMA':            '#a855f7',
    'Exp. Smoothing':    '#06b6d4',
    'Random Forest':     '#10b981',
    'Gradient Boosting': '#eab308',
}

def apply_theme(fig, title='', height=440, legend_bottom=False):
    fig.update_layout(
        height=height, title=dict(text=title, font=dict(color='#64748b', size=11,
            family='Inter'), x=0, xanchor='left'),
        plot_bgcolor=BG, paper_bgcolor=BG,
        font=dict(color=TICK, family='Inter', size=11),
        xaxis=dict(showgrid=False, color=TICK, linecolor=GRID,
                   zeroline=False, tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor=GRID, color=TICK,
                   linecolor='rgba(0,0,0,0)', zeroline=False,
                   tickfont=dict(size=10)),
        hovermode='x unified',
        hoverlabel=dict(bgcolor=HOVER_BG, font_color='#e2e8f0',
                        bordercolor='#1a2236', font_size=11,
                        font_family='JetBrains Mono'),
        margin=dict(l=0, r=0, t=36, b=0),
        legend=dict(
            bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)',
            font=dict(color='#64748b', size=10),
            orientation='h' if legend_bottom else 'v',
            yanchor='bottom' if legend_bottom else 'middle',
            y=1.04 if legend_bottom else 0.5,
            xanchor='right' if legend_bottom else 'left',
            x=1 if legend_bottom else 1.01
        )
    )
    return fig

def rgba(hex_color, alpha=0.15):
    h = hex_color.lstrip('#')
    r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f'rgba({r},{g},{b},{alpha})'

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        return (
            pd.read_csv('preprocessed_data.csv', parse_dates=['date']),
            pd.read_csv('modeling_data.csv',     parse_dates=['date']),
            pd.read_csv('forecast_results.csv',  parse_dates=['date']),
            pd.read_csv('test_data.csv',         parse_dates=['date']),
        )
    except FileNotFoundError as e:
        st.error(f"Missing file: {e}")
        st.stop()

@st.cache_data
def load_preds():
    try:
        with open('model_preds.pkl','rb') as f: return pickle.load(f)
    except: return {}

df, df_model, forecasts, test = load_data()
model_preds = load_preds()
ALL_MODELS  = list(MODEL_COLORS.keys())

def kpis(fc, models, thresh, hz):
    sub  = fc[fc['model'].isin(models) & (fc['horizon_day']<=hz)].copy()
    mape = sub.groupby('model').apply(
        lambda x: np.mean(np.abs((x['actual']-x['predicted'])/x['actual'].replace(0,np.nan)))*100
    )
    acc   = max(0, 100 - mape.mean())
    bprob = sub[sub['predicted']>thresh]['date'].nunique() / max(sub['date'].nunique(),1)*100
    cv    = sub.groupby('model')['predicted'].std() / sub.groupby('model')['predicted'].mean()
    stab  = (1/cv.mean()) if cv.mean()>0 else 0
    return acc, bprob, stab

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
      <div class="sb-brand-name">UAC<span>.</span>Forecast</div>
      <div class="sb-brand-sub">HHS Intelligence Platform</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sb-section"><div class="sb-section-label">Forecast Horizon</div>', unsafe_allow_html=True)
    horizon = st.select_slider("", options=[7,14,30,60], value=30, label_visibility='collapsed')
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section"><div class="sb-section-label">Active Models</div>', unsafe_allow_html=True)
    selected_models = [m for m in ALL_MODELS if st.checkbox(m, value=True, key=f'm_{m}')]
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section"><div class="sb-section-label">Surge Threshold</div>', unsafe_allow_html=True)
    capacity = st.number_input("", min_value=1000, max_value=15000,
                               value=2511, step=50, label_visibility='collapsed')
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section"><div class="sb-section-label">Confidence Interval</div>', unsafe_allow_html=True)
    show_ci  = st.toggle("Show CI bands", value=True)
    ci_pct   = st.slider("Width (±%)", 5, 25, 10) if show_ci else 10
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style='padding:1rem 1.25rem;'>
      <div style='font-size:0.62rem;color:#1e293b;text-align:center;letter-spacing:0.08em;text-transform:uppercase'>
        {len(ALL_MODELS)} models · {len(df)} records
      </div>
    </div>""", unsafe_allow_html=True)

# ── Topbar ────────────────────────────────────────────────────────────────────
import datetime
now = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M")

st.markdown(f"""
<div class="topbar">
  <div class="topbar-brand">UAC<span>.</span>Forecast &nbsp;/&nbsp; Intelligence Dashboard</div>
  <div class="topbar-meta">
    <span class="topbar-badge badge-live">Live</span>
    <span class="topbar-badge badge-models">{len(selected_models)} of {len(ALL_MODELS)} models</span>
    <span class="topbar-date">{now}</span>
  </div>
</div>""", unsafe_allow_html=True)

if not selected_models:
    st.markdown('<div class="warn-box">Select at least one model from the sidebar to continue.</div>', unsafe_allow_html=True)
    st.stop()

# ── KPI strip ─────────────────────────────────────────────────────────────────
avg_acc, breach_pct, stability = kpis(forecasts, selected_models, capacity, horizon)

kpi_data = [
    ("Forecast Accuracy",        f"{avg_acc:.1f}%",           "Average across selected models"),
    ("Capacity Breach Prob.",    f"{breach_pct:.1f}%",         f"Days above {capacity:,} threshold"),
    ("Stability Index",          f"{stability:.2f}",           "Inverse coefficient of variation"),
    ("Active Horizon",           f"{horizon}d",                "Days ahead being forecast"),
    ("Models Running",           str(len(selected_models)),    f"of {len(ALL_MODELS)} available"),
]

cells = "".join([
    f'<div class="kpi-cell kpi-c{i}"><div class="kpi-sup">{d[0]}</div>'
    f'<div class="kpi-num">{d[1]}</div><div class="kpi-sub">{d[2]}</div></div>'
    for i,d in enumerate(kpi_data)
])
st.markdown(f'<div class="kpi-strip">{cells}</div>', unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Care Load Forecast",
    "Discharge Demand",
    "Model Comparison",
    "Scenario Analysis",
])

sub_base = forecasts[
    forecasts['model'].isin(selected_models) &
    (forecasts['horizon_day'] <= horizon)
].copy()

# ════════════════════════════════════════════════════════
# TAB 1 — CARE LOAD FORECAST
# ════════════════════════════════════════════════════════
with tab1:
    c_left, c_right = st.columns([3, 1])

    with c_left:
        st.markdown('<div class="section-label">Future Care Load Forecast</div>', unsafe_allow_html=True)

        fig = go.Figure()

        # Actual test data (get first to know test_start for bridging)
        act = (sub_base[sub_base['model']==selected_models[0]]
               [['date','actual']].drop_duplicates().sort_values('date'))
        test_start = act['date'].min() if not act.empty else None

        # Historical — clip right up to test start so there's zero gap
        hist_all  = df.sort_values('date').copy()
        hist_pre  = hist_all[hist_all['date'] < test_start].tail(90) if test_start is not None else hist_all.tail(90)

        # Shade the historical window with a rectangle (avoids fill-to-zero bug)
        if not hist_pre.empty and test_start is not None:
            fig.add_vrect(
                x0=str(hist_pre['date'].min()), x1=str(test_start),
                fillcolor='rgba(30,41,59,0.3)', layer='below', line_width=0,
                annotation_text='HISTORICAL', annotation_position='top left',
                annotation_font=dict(color='#334155', size=9)
            )

        # Historical line — NO fill, just a line
        fig.add_trace(go.Scatter(
            x=hist_pre['date'], y=hist_pre['hhs_care'], name='Historical',
            line=dict(color='#334155', width=1.5), mode='lines',
            hovertemplate='Historical: %{y:,.0f}<extra></extra>'
        ))

        # Actual test line
        fig.add_trace(go.Scatter(
            x=act['date'], y=act['actual'], name='Actual',
            line=dict(color='#94a3b8', width=2, dash='dot'),
            hovertemplate='Actual: %{y:,.0f}<extra></extra>'
        ))

        # Surge line
        fig.add_hline(y=capacity, line_dash='dash',
                      line_color='rgba(244,63,94,0.4)', line_width=1,
                      annotation_text=f'Threshold  {capacity:,}',
                      annotation_font=dict(color='#f43f5e', size=10),
                      annotation_position='top right')

        for model in selected_models:
            m     = sub_base[sub_base['model']==model].sort_values('date')
            color = MODEL_COLORS.get(model, '#888')
            ci_v  = m['predicted'] * (ci_pct/100)
            if show_ci:
                fig.add_trace(go.Scatter(
                    x=pd.concat([m['date'], m['date'][::-1]]),
                    y=pd.concat([m['predicted']+ci_v, (m['predicted']-ci_v)[::-1]]),
                    fill='toself', fillcolor=rgba(color, 0.08),
                    line=dict(color='rgba(0,0,0,0)'),
                    showlegend=False, hoverinfo='skip', name=f'{model} CI'
                ))
            fig.add_trace(go.Scatter(
                x=m['date'], y=m['predicted'], name=model,
                line=dict(color=color, width=2),
                hovertemplate=f'<b>{model}</b>: %{{y:,.0f}}<extra></extra>'
            ))

        apply_theme(fig, f'HHS Care Load — {horizon}-Day Forecast', 460, legend_bottom=True)
        st.plotly_chart(fig, use_container_width=True)

    with c_right:
        st.markdown('<div class="section-label">Breach Alerts</div>', unsafe_allow_html=True)
        any_breach = False
        for model in selected_models:
            m = sub_base[sub_base['model']==model]
            b = m[m['predicted']>capacity]
            if not b.empty:
                any_breach = True
                col = MODEL_COLORS.get(model,'#888')
                st.markdown(f"""
                <div style='background:rgba(244,63,94,0.06);border:1px solid rgba(244,63,94,0.15);
                            border-left:3px solid {col};border-radius:0 8px 8px 0;
                            padding:0.7rem 0.9rem;margin-bottom:0.4rem;'>
                  <div style='font-size:0.72rem;font-weight:700;color:{col};letter-spacing:0.04em;
                              text-transform:uppercase;margin-bottom:0.2rem'>{model}</div>
                  <div style='font-size:1.4rem;font-weight:800;color:#f43f5e;line-height:1'>{len(b)}</div>
                  <div style='font-size:0.68rem;color:#475569;margin-top:0.1rem'>breach days</div>
                </div>""", unsafe_allow_html=True)
        if not any_breach:
            st.markdown('<div class="success-box" style="font-size:0.78rem">No capacity breaches detected in the forecast window.</div>', unsafe_allow_html=True)

        st.markdown('<br><div class="section-label">Model Legend</div>', unsafe_allow_html=True)
        for model in selected_models:
            col = MODEL_COLORS.get(model,'#888')
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:0.5rem;padding:0.3rem 0;
                        border-bottom:1px solid #0d1117;'>
              <div style='width:20px;height:2px;background:{col};border-radius:1px;flex-shrink:0'></div>
              <div style='font-size:0.75rem;color:#64748b'>{model}</div>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# TAB 2 — DISCHARGE DEMAND
# ════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-label">Historical Discharge Patterns</div>', unsafe_allow_html=True)

    df_s = df.sort_values('date').copy()
    df_s['discharge'] = df_s['hhs_care'].diff(-1).clip(lower=0)
    df_s['net_flow']  = df_s['hhs_care'].diff()
    df_s['roll7']     = df_s['discharge'].rolling(7).mean()
    tail = df_s.tail(90)

    col1, col2 = st.columns(2)

    with col1:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=tail['date'], y=tail['discharge'], name='Daily Discharge',
            marker=dict(color=rgba('#6366f1', 0.6),
                        line=dict(color=rgba('#6366f1', 0.9), width=0.5)),
            hovertemplate='%{y:,.0f} children<extra></extra>'
        ))
        fig2.add_trace(go.Scatter(
            x=tail['date'], y=tail['roll7'], name='7-Day Avg',
            line=dict(color='#818cf8', width=2),
            hovertemplate='Avg: %{y:,.0f}<extra></extra>'
        ))
        apply_theme(fig2, 'Daily Discharge Volume (last 90 days)', 320)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        colors_nf = [rgba('#f43f5e', 0.7) if v>0 else rgba('#10b981', 0.7)
                     for v in tail['net_flow']]
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=tail['date'], y=tail['net_flow'], name='Net Flow',
            marker=dict(color=colors_nf, line=dict(width=0)),
            hovertemplate='%{y:+,.0f}<extra></extra>'
        ))
        fig3.add_hline(y=0, line_color=rgba('#94a3b8', 0.3), line_width=1)
        apply_theme(fig3, 'Net Flow — Intake minus Discharge', 320)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-label">Forecasted Discharge Demand by Model</div>', unsafe_allow_html=True)

    sub2 = sub_base.sort_values(['model','date']).copy()
    sub2['f_discharge'] = sub2.groupby('model')['predicted'].diff(-1).clip(lower=0)

    fig4 = go.Figure()
    for model in selected_models:
        m = sub2[sub2['model']==model]
        fig4.add_trace(go.Scatter(
            x=m['date'], y=m['f_discharge'], name=model,
            line=dict(color=MODEL_COLORS.get(model,'#888'), width=2),
            mode='lines',
            hovertemplate=f'<b>{model}</b>: %{{y:,.0f}}<extra></extra>'
        ))
    apply_theme(fig4, f'Forecasted Discharge Demand — Next {horizon} Days', 360, legend_bottom=True)
    st.plotly_chart(fig4, use_container_width=True)

# ════════════════════════════════════════════════════════
# TAB 3 — MODEL COMPARISON
# ════════════════════════════════════════════════════════
with tab3:
    metrics = []
    for model in selected_models:
        m = sub_base[sub_base['model']==model]
        if m.empty: continue
        a,p  = m['actual'].values, m['predicted'].values
        mae  = np.mean(np.abs(a-p))
        rmse = np.sqrt(np.mean((a-p)**2))
        mape = np.mean(np.abs((a-p)/np.where(a==0,1,a)))*100
        acc  = max(0,100-mape)
        metrics.append({'Model':model, 'Accuracy':round(acc,2),
                        'MAE':round(mae,1), 'RMSE':round(rmse,1),
                        'MAPE':round(mape,2),
                        'Breach Days':int((p>capacity).sum())})

    mdf = pd.DataFrame(metrics).sort_values('Accuracy', ascending=False).reset_index(drop=True)

    col_tbl, col_radar = st.columns([1, 1.4])

    with col_tbl:
        st.markdown('<div class="section-label">Performance Metrics</div>', unsafe_allow_html=True)
        st.dataframe(
            mdf.style
               .highlight_max(subset=['Accuracy'], color='#0f2d1f')
               .highlight_min(subset=['RMSE','MAE','MAPE'], color='#0f2d1f')
               .highlight_max(subset=['RMSE','MAE','MAPE'], color='#2d0f18')
               .format({'Accuracy':'{:.2f}','MAE':'{:.1f}','RMSE':'{:.1f}','MAPE':'{:.2f}'}),
            use_container_width=True, hide_index=True, height=280
        )
        if not mdf.empty:
            best = mdf.iloc[0]
            col  = MODEL_COLORS.get(best['Model'],'#6366f1')
            st.markdown(f"""
            <div class="success-box" style="border-left-color:{col}">
              <div style='font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;
                          color:#475569;margin-bottom:0.3rem'>Best Performing Model</div>
              <div style='font-size:0.95rem;font-weight:700;color:{col}'>{best['Model']}</div>
              <div style='font-size:0.75rem;color:#475569;margin-top:0.2rem'>
                {best['Accuracy']:.2f}% accuracy · RMSE {best['RMSE']:.1f}</div>
            </div>""", unsafe_allow_html=True)

    with col_radar:
        st.markdown('<div class="section-label">Multi-Dimensional Radar</div>', unsafe_allow_html=True)
        cats = ['Accuracy','Precision (MAE)','Stability (RMSE)','Consistency (MAPE)','Low Breach']
        fig_r = go.Figure()
        for _,row in mdf.iterrows():
            vals = [
                row['Accuracy'],
                max(0,100-row['MAE']/10),
                max(0,100-row['RMSE']/10),
                max(0,100-row['MAPE']),
                max(0,100-row['Breach Days']*5)
            ]
            vals += [vals[0]]
            col = MODEL_COLORS.get(row['Model'],'#888')
            fig_r.add_trace(go.Scatterpolar(
                r=vals, theta=cats+[cats[0]], name=row['Model'],
                fill='toself', fillcolor=rgba(col,0.07),
                line=dict(color=col, width=1.5),
                hovertemplate='%{theta}: %{r:.1f}<extra></extra>'
            ))
        fig_r.update_layout(
            height=340,
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(visible=True, range=[0,105],
                                gridcolor=GRID, color='#334155',
                                tickfont=dict(size=9,color='#334155')),
                angularaxis=dict(gridcolor=GRID, color='#64748b',
                                 tickfont=dict(size=10,color='#64748b'))
            ),
            paper_bgcolor=BG,
            font=dict(color=TICK, family='Inter', size=10),
            legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)',
                        font=dict(color='#475569', size=9),
                        orientation='h', y=-0.15, x=0.5, xanchor='center'),
            margin=dict(l=20,r=20,t=20,b=20)
        )
        st.plotly_chart(fig_r, use_container_width=True)

    st.markdown('<div class="section-label">Actual vs Predicted — Deep Dive</div>', unsafe_allow_html=True)

    sel   = st.selectbox("Select model", selected_models, label_visibility='collapsed')
    mi    = sub_base[sub_base['model']==sel].sort_values('date')
    mcol  = MODEL_COLORS.get(sel,'#6366f1')
    ci_v2 = mi['predicted'] * (ci_pct/100)

    fig6 = go.Figure()
    if show_ci:
        fig6.add_trace(go.Scatter(
            x=pd.concat([mi['date'],mi['date'][::-1]]),
            y=pd.concat([mi['predicted']+ci_v2,(mi['predicted']-ci_v2)[::-1]]),
            fill='toself', fillcolor=rgba(mcol,0.1),
            line=dict(color='rgba(0,0,0,0)'),
            name=f'±{ci_pct}% CI', showlegend=True, hoverinfo='skip'
        ))
    # Error region
    fig6.add_trace(go.Scatter(
        x=pd.concat([mi['date'],mi['date'][::-1]]),
        y=pd.concat([mi['actual'],(mi['predicted'])[::-1]]),
        fill='toself', fillcolor='rgba(244,63,94,0.05)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Error Region', showlegend=True, hoverinfo='skip'
    ))
    fig6.add_trace(go.Scatter(
        x=mi['date'], y=mi['actual'], name='Actual',
        line=dict(color='#e2e8f0', width=2),
        hovertemplate='Actual: %{y:,.0f}<extra></extra>'
    ))
    fig6.add_trace(go.Scatter(
        x=mi['date'], y=mi['predicted'], name='Predicted',
        line=dict(color=mcol, width=2, dash='dash'),
        hovertemplate='Predicted: %{y:,.0f}<extra></extra>'
    ))
    apply_theme(fig6, f'{sel} — Actual vs Predicted', 380, legend_bottom=True)
    st.plotly_chart(fig6, use_container_width=True)

# ════════════════════════════════════════════════════════
# TAB 4 — SCENARIO ANALYSIS
# ════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-label">Cross-Horizon RMSE Comparison</div>', unsafe_allow_html=True)

    h_sel = st.multiselect(
        "", options=[7,14,30,60], default=[7,30],
        label_visibility='collapsed', key='h_sel'
    )

    if len(h_sel) < 2:
        st.markdown('<div class="info-box">Select at least two forecast horizons to run scenario analysis.</div>', unsafe_allow_html=True)
    else:
        fig7 = make_subplots(
            rows=1, cols=len(h_sel),
            subplot_titles=[f'{h}d Horizon' for h in h_sel],
            shared_yaxes=True, horizontal_spacing=0.04
        )
        for ci2, h in enumerate(h_sel):
            sub = forecasts[forecasts['model'].isin(selected_models) &
                            (forecasts['horizon_day']<=h)]
            for model in selected_models:
                m = sub[sub['model']==model]
                if m.empty: continue
                rmse = np.sqrt(np.mean((m['actual'].values-m['predicted'].values)**2))
                fig7.add_trace(go.Bar(
                    x=[model], y=[rmse],
                    name=model if ci2==0 else None,
                    marker=dict(color=rgba(MODEL_COLORS.get(model,'#888'),0.8),
                                line=dict(color=MODEL_COLORS.get(model,'#888'),width=1)),
                    showlegend=(ci2==0),
                    hovertemplate=f'RMSE: %{{y:.1f}}<extra>{model}</extra>'
                ), row=1, col=ci2+1)

        fig7.update_layout(
            height=400, barmode='group', plot_bgcolor=BG, paper_bgcolor=BG,
            font=dict(color=TICK, family='Inter', size=10),
            title=dict(text='RMSE by Model across Forecast Horizons  —  lower is better',
                       font=dict(color='#475569', size=11), x=0),
            legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)',
                        font=dict(color='#475569', size=9),
                        orientation='h', y=1.12, x=1, xanchor='right'),
            margin=dict(l=0,r=0,t=60,b=0),
            hoverlabel=dict(bgcolor=HOVER_BG, font_color='#e2e8f0',
                            bordercolor='#1a2236', font_family='JetBrains Mono')
        )
        for i in range(1, len(h_sel)+1):
            fig7.update_xaxes(showgrid=False, color=TICK, linecolor=GRID, tickfont=dict(size=9), row=1, col=i)
            fig7.update_yaxes(showgrid=True, gridcolor=GRID, color=TICK,
                              linecolor='rgba(0,0,0,0)', row=1, col=i)
        st.plotly_chart(fig7, use_container_width=True)

        st.markdown('<div class="section-label" style="margin-top:1.5rem">Best Model per Horizon</div>', unsafe_allow_html=True)
        for h in h_sel:
            sub       = forecasts[forecasts['model'].isin(selected_models) & (forecasts['horizon_day']<=h)]
            best_m, best_a = None, -999
            for model in selected_models:
                m = sub[sub['model']==model]
                if m.empty: continue
                mape = np.mean(np.abs((m['actual'].values-m['predicted'].values)/
                               np.where(m['actual'].values==0,1,m['actual'].values)))*100
                acc  = 100-mape
                if acc > best_a: best_a, best_m = acc, model
            col = MODEL_COLORS.get(best_m,'#6366f1')
            st.markdown(f"""
            <div class="scenario-row" style="border-left-color:{col}">
              <span class="scenario-horizon">{h:02d}d  horizon</span>
              <span class="scenario-model">{best_m}</span>
              <span class="scenario-acc" style="color:{col}">{best_a:.1f}%</span>
            </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  UAC Forecast Intelligence Platform &nbsp;&mdash;&nbsp;
  HHS Unaccompanied Alien Children Program &nbsp;&mdash;&nbsp;
  Internship Research Project
</div>""", unsafe_allow_html=True)
