import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import time
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# ══════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="CropAI Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════
#  PROFESSIONAL CSS
# ══════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@700;800&display=swap');

:root {
  --bg:        #0d1117;
  --surface:   #161b22;
  --card:      #1a2030;
  --card-alt:  #1e2535;
  --border:    #2a3245;
  --border-lt: #334060;
  --teal:      #2dd4bf;
  --teal-dim:  rgba(45,212,191,.12);
  --teal-bdr:  rgba(45,212,191,.30);
  --amber:     #f59e0b;
  --amber-dim: rgba(245,158,11,.12);
  --amber-bdr: rgba(245,158,11,.30);
  --blue:      #60a5fa;
  --blue-dim:  rgba(96,165,250,.12);
  --blue-bdr:  rgba(96,165,250,.28);
  --green:     #4ade80;
  --violet:    #a78bfa;
  --rose:      #f87171;
  --sky:       #38bdf8;
  --text:      #e2e8f0;
  --text-sec:  #94a3b8;
  --text-dim:  #4a5568;
}

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  background: var(--bg) !important;
  color: var(--text) !important;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { display: none; }
.block-container { padding: 0 2.5rem 5rem !important; max-width: 1500px; }

/* ── HERO ── */
.hero-wrap {
  background: linear-gradient(160deg, #0d1520 0%, #111827 50%, #0d1117 100%);
  border-bottom: 1px solid var(--border);
  padding: 3.2rem 3rem 2.8rem;
  margin: 0 -2.5rem 3rem;
  position: relative;
  overflow: hidden;
}
.hero-wrap::before {
  content: '';
  position: absolute; inset: 0;
  background:
    radial-gradient(ellipse 55% 70% at 0% 50%,  rgba(45,212,191,.07) 0%, transparent 65%),
    radial-gradient(ellipse 45% 60% at 100% 20%, rgba(245,158,11,.06) 0%, transparent 60%),
    radial-gradient(ellipse 35% 50% at 55% 100%, rgba(96,165,250,.05) 0%, transparent 55%);
}
.hero-wrap::after {
  content: '🌾';
  position: absolute; right: 3.5rem; top: 1.5rem;
  font-size: 6rem; opacity: .06; filter: blur(1px);
  pointer-events: none;
}
.hero-wrap h1 {
  font-family: 'Playfair Display', serif;
  font-size: 2.9rem; font-weight: 800;
  margin: 0 0 .4rem; letter-spacing: -.01em;
  color: var(--text);
  position: relative;
}
.hero-wrap h1 span {
  background: linear-gradient(90deg, var(--teal) 0%, var(--blue) 60%, var(--amber) 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
}
.hero-wrap .sub {
  font-size: .82rem; color: var(--text-sec);
  letter-spacing: .1em; text-transform: uppercase;
  position: relative; margin-bottom: 1.2rem;
}
.hero-pills { display: flex; gap: .6rem; position: relative; flex-wrap: wrap; }
.hero-pill {
  display: inline-flex; align-items: center; gap: .35rem;
  padding: .28rem .85rem; border-radius: 6px;
  font-size: .7rem; font-weight: 600;
  letter-spacing: .07em; text-transform: uppercase;
  font-family: 'DM Mono', monospace;
}
.p1 { background: var(--teal-dim);  color: var(--teal);  border: 1px solid var(--teal-bdr); }
.p2 { background: var(--blue-dim);  color: var(--blue);  border: 1px solid var(--blue-bdr); }
.p3 { background: var(--amber-dim); color: var(--amber); border: 1px solid var(--amber-bdr); }

/* ── SECTION TITLES ── */
.stitle {
  font-family: 'DM Sans', sans-serif;
  font-size: .72rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: .16em;
  color: var(--text-sec);
  margin: 2.8rem 0 1.2rem;
  display: flex; align-items: center; gap: .75rem;
}
.stitle .dot {
  width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0;
}
.stitle::after {
  content: ''; flex: 1; height: 1px;
  background: linear-gradient(90deg, var(--border), transparent);
}

/* ── KPI CARDS ── */
.kpi {
  border-radius: 12px; padding: 1.5rem 1.4rem 1.3rem;
  background: var(--card);
  position: relative; overflow: hidden; cursor: default;
  transition: transform .18s ease, box-shadow .18s ease;
}
.kpi:hover { transform: translateY(-2px); }
.kpi .glow {
  position: absolute; width: 80px; height: 80px;
  border-radius: 50%; top: -15px; right: -15px;
  filter: blur(30px); opacity: .18;
}
.kpi .icon  { font-size: 1.4rem; margin-bottom: .6rem; display: block; }
.kpi .lbl   {
  font-size: .62rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: .13em;
  color: var(--text-dim); margin-bottom: .35rem;
  font-family: 'DM Mono', monospace;
}
.kpi .val   {
  font-family: 'Playfair Display', serif;
  font-size: 1.8rem; font-weight: 700; line-height: 1.1;
  margin-bottom: .3rem; color: var(--text);
}
.kpi .hint  { font-size: .67rem; color: var(--text-sec); }

.k1 { border: 1px solid var(--teal-bdr);  } .k1 .val { color: var(--teal);  } .k1 .glow { background: var(--teal);  }
.k2 { border: 1px solid var(--teal-bdr);  } .k2 .val { color: var(--teal);  } .k2 .glow { background: var(--teal);  }
.k3 { border: 1px solid var(--amber-bdr); } .k3 .val { color: var(--amber); } .k3 .glow { background: var(--amber); }
.k4 { border: 1px solid var(--amber-bdr); } .k4 .val { color: var(--amber); } .k4 .glow { background: var(--amber); }
.k5 { border: 1px solid var(--blue-bdr);  } .k5 .val { color: var(--blue);  } .k5 .glow { background: var(--blue);  }
.k6 { border: 1px solid var(--blue-bdr);  } .k6 .val { color: var(--blue);  } .k6 .glow { background: var(--blue);  }
.k7 { border: 1px solid rgba(74,222,128,.28); } .k7 .val { color: var(--green); } .k7 .glow { background: var(--green); }

/* ── PREDICTION CARD ── */
.pred-result {
  border-radius: 16px;
  padding: 2rem 1.8rem;
  background: linear-gradient(135deg, #0e2a33 0%, #0d1a28 100%);
  border: 1.5px solid var(--teal-bdr);
  text-align: center;
  margin-top: 1rem;
}
.pred-result .crop-icon { font-size: 3.5rem; margin-bottom: .5rem; }
.pred-result .crop-name {
  font-family: 'Playfair Display', serif;
  font-size: 2.2rem; font-weight: 800;
  color: var(--teal);
  text-transform: capitalize;
  margin-bottom: .3rem;
}
.pred-result .crop-sub { font-size: .8rem; color: var(--text-sec); letter-spacing: .08em; text-transform: uppercase; }
.pred-result .conf-bar-wrap {
  margin: 1.2rem auto 0; max-width: 280px;
  background: rgba(255,255,255,.05); border-radius: 8px; height: 8px; overflow: hidden;
}
.pred-result .conf-bar {
  height: 100%; border-radius: 8px;
  background: linear-gradient(90deg, var(--teal), var(--blue));
}
.pred-result .conf-txt {
  font-size: .72rem; color: var(--text-sec); margin-top: .5rem;
  font-family: 'DM Mono', monospace;
}

/* top-k alternatives */
.alt-pill {
  display: inline-flex; align-items: center; gap: .4rem;
  padding: .3rem .8rem; border-radius: 20px;
  background: var(--card-alt); border: 1px solid var(--border);
  font-size: .72rem; color: var(--text-sec);
  margin: .3rem .2rem;
}
.alt-pill b { color: var(--text); font-family: 'DM Mono', monospace; }

/* input panel */
.input-panel {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.4rem 1.2rem;
}

/* ── STREAMLIT WIDGETS ── */
.stSelectbox > div > div {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
}
.stRadio > div { color: var(--text) !important; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
  gap: .4rem; background: transparent !important;
  border-bottom: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-bottom: none !important;
  border-radius: 8px 8px 0 0 !important;
  color: var(--text-sec) !important;
  font-size: .78rem !important; font-weight: 600 !important;
  padding: .45rem 1.1rem !important;
}
.stTabs [aria-selected="true"] {
  background: var(--card-alt) !important;
  border-color: var(--teal-bdr) !important;
  color: var(--teal) !important;
}

/* ── PROGRESS BAR ── */
.stProgress > div > div > div {
  background: linear-gradient(90deg, var(--teal), var(--blue)) !important;
}
.stDataFrame { border: 1px solid var(--border) !important; border-radius: 10px !important; }

.footer {
  text-align: center; color: var(--text-dim);
  font-size: .68rem; margin-top: 5rem;
  letter-spacing: .13em; text-transform: uppercase;
  font-family: 'DM Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  PLOTLY THEME
# ══════════════════════════════════════════════════════
VP = ['#2dd4bf','#60a5fa','#f59e0b','#4ade80','#a78bfa','#f87171','#38bdf8']
_AXIS = dict(
    gridcolor='#1e2535', linecolor='#2a3245', tickcolor='#4a5568',
    tickfont=dict(color='#94a3b8', size=11),
    title_font=dict(color='#94a3b8'),
)
PT = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='DM Sans, sans-serif', color='#e2e8f0', size=12),
    title=dict(font=dict(family='DM Sans, sans-serif', size=15, color='#e2e8f0'), pad=dict(b=8)),
    xaxis=_AXIS, yaxis=_AXIS,
    legend=dict(bgcolor='rgba(26,32,48,.85)', bordercolor='#2a3245', borderwidth=1,
                font=dict(color='#94a3b8', size=11)),
    colorway=VP,
    margin=dict(l=44, r=24, t=52, b=44),
)
def sf(fig):
    fig.update_layout(**PT)
    return fig

# ══════════════════════════════════════════════════════
#  CROP EMOJI MAP
# ══════════════════════════════════════════════════════
CROP_EMOJI = {
    "rice": "🌾", "maize": "🌽", "chickpea": "🫘", "kidneybeans": "🫘",
    "pigeonpeas": "🫘", "mothbeans": "🫘", "mungbean": "🫘", "blackgram": "🫘",
    "lentil": "🫘", "pomegranate": "🍎", "banana": "🍌", "mango": "🥭",
    "grapes": "🍇", "watermelon": "🍉", "muskmelon": "🍈", "apple": "🍎",
    "orange": "🍊", "papaya": "🍈", "coconut": "🥥", "cotton": "🌿",
    "jute": "🌿", "coffee": "☕",
}

# ══════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════
st.markdown("""
<div class="hero-wrap">
  <h1>Crop<span>AI</span> Dashboard</h1>
  <div class="sub">Machine Learning · Model Comparison · Crop Recommendation System</div>
  <div class="hero-pills">
    <span class="hero-pill p1">🤖 7 ML Models</span>
    <span class="hero-pill p2">📊 Cross-Validated</span>
    <span class="hero-pill p3">🌾 Crop Intelligence</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════
TEST_SIZE   = 0.20
RANDOM_SEED = 42
CV_FOLDS    = 5

# ══════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════
@st.cache_data
def load_data():
    return pd.read_csv("Crop_recommendation.csv")

try:
    data = load_data()
except FileNotFoundError:
    st.error("⚠️  `Crop_recommendation.csv` not found. Place it in the same directory as this script.")
    st.stop()

X           = data.drop("label", axis=1)
y_raw       = data["label"]
le          = LabelEncoder()
y           = le.fit_transform(y_raw)
class_names = le.classes_

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)
scaler = StandardScaler()
X_tr   = scaler.fit_transform(X_train)
X_te   = scaler.transform(X_test)

# Feature min/max for sliders (from training data)
feat_stats = {
    col: {"min": float(X[col].min()), "max": float(X[col].max()), "mean": float(X[col].mean())}
    for col in X.columns
}

# ══════════════════════════════════════════════════════
#  MODEL REGISTRY  (tuned for best accuracy)
# ══════════════════════════════════════════════════════
MODELS = {
    "Random Forest":       RandomForestClassifier(n_estimators=300, max_features='sqrt',
                               min_samples_leaf=1, n_jobs=-1, random_state=RANDOM_SEED),
    "SVM":                 SVC(C=100, gamma='scale', kernel='rbf',
                               probability=True, random_state=RANDOM_SEED),
    "XGBoost":             XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8, eval_metric='mlogloss',
                               random_state=RANDOM_SEED, verbosity=0),
    "Logistic Regression": LogisticRegression(max_iter=2000, C=10, solver='lbfgs',
                               random_state=RANDOM_SEED),
    "Decision Tree":       DecisionTreeClassifier(max_depth=None, min_samples_leaf=1,
                               random_state=RANDOM_SEED),
    "KNN":                 KNeighborsClassifier(n_neighbors=3, weights='distance',
                               metric='euclidean'),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, max_depth=5,
                               learning_rate=0.08, subsample=0.85, random_state=RANDOM_SEED),
}

# ══════════════════════════════════════════════════════
#  TRAIN ALL MODELS
# ══════════════════════════════════════════════════════
@st.cache_resource
def train_all_models():
    results, train_times, trained_models, cv_scores = [], {}, {}, {}
    for name, model in MODELS.items():
        t0 = time.time()
        model.fit(X_tr, y_train)
        elapsed = time.time() - t0
        trained_models[name] = model
        train_times[name]    = elapsed

        yp   = model.predict(X_te)
        acc  = accuracy_score(y_test, yp)
        prec = precision_score(y_test, yp, average='weighted', zero_division=0)
        rec  = recall_score(y_test, yp, average='weighted', zero_division=0)
        f1   = f1_score(y_test, yp, average='weighted', zero_division=0)
        results.append([name, acc, prec, rec, f1, elapsed])

        fresh = model.__class__(**model.get_params())
        cv = cross_val_score(
            fresh, X_tr, y_train,
            cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED),
            scoring='accuracy', n_jobs=-1
        )
        cv_scores[name] = cv

    res = pd.DataFrame(results, columns=["Model","Accuracy","Precision","Recall","F1 Score","Train Time (s)"])
    res = res.sort_values("Accuracy", ascending=False).reset_index(drop=True)
    return res, train_times, trained_models, cv_scores

pbar = st.progress(0, text="🚀 Initialising training pipeline …")
for i, name in enumerate(MODELS.keys()):
    pbar.progress(i / len(MODELS), text=f"⚙️  Training  **{name}**  ({i+1}/{len(MODELS)}) …")

res, train_times, trained_models, cv_scores = train_all_models()
pbar.progress(1.0, text="✅ All models trained successfully!")
time.sleep(0.4)
pbar.empty()

best = res.iloc[0]

# ══════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════
def kpi_html(cls, icon, lbl, val, hint):
    return f"""
    <div class="kpi {cls}">
      <div class="glow"></div>
      <span class="icon">{icon}</span>
      <div class="lbl">{lbl}</div>
      <div class="val">{val}</div>
      <div class="hint">{hint}</div>
    </div>"""

def sec(label, dot_color):
    st.markdown(
        f'<div class="stitle">'
        f'<span class="dot" style="background:{dot_color};box-shadow:0 0 5px {dot_color}80"></span>'
        f'{label}</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════
#  NAVIGATION TABS  (top-level)
# ══════════════════════════════════════════════════════
nav_tab1, nav_tab2, nav_tab3 = st.tabs([
    "📊 Model Analytics",
    "🌾 Crop Predictor",
    "🔬 Data Explorer",
])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 : MODEL ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with nav_tab1:

    # ── Overview KPIs ──────────────────────────────────────────────────────
    sec("Overview", "#2dd4bf")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_html("k1","🏆","Best Model",    best["Model"],
        f"Top of {len(MODELS)} trained models"), unsafe_allow_html=True)
    c2.markdown(kpi_html("k2","🎯","Best Accuracy", f"{best['Accuracy']:.4f}",
        f"Test split {int(TEST_SIZE*100)}%"), unsafe_allow_html=True)
    c3.markdown(kpi_html("k3","📐","F1 Score",      f"{best['F1 Score']:.4f}",
        "Weighted average"), unsafe_allow_html=True)
    c4.markdown(kpi_html("k4","⚡","Fastest Model",
        min(train_times, key=train_times.get),
        f"{min(train_times.values()):.3f}s"), unsafe_allow_html=True)

    st.write("")
    d1, d2, d3 = st.columns(3)
    d1.markdown(kpi_html("k5","🗂","Total Samples", f"{len(data):,}",
        f"{int(len(data)*(1-TEST_SIZE)):,} train · {int(len(data)*TEST_SIZE):,} test"), unsafe_allow_html=True)
    d2.markdown(kpi_html("k6","🔬","Features", str(X.shape[1]),
        " · ".join(X.columns.tolist()[:3]) + " …"), unsafe_allow_html=True)
    d3.markdown(kpi_html("k7","🌱","Crop Classes", str(len(class_names)),
        "Unique crop labels"), unsafe_allow_html=True)

    # ── Full Results Table ──────────────────────────────────────────────────
    sec("Full Results Table", "#2dd4bf")
    disp = res.copy()
    for col in ["Accuracy","Precision","Recall","F1 Score"]:
        disp[col] = disp[col].apply(lambda v: f"{v:.4f}")
    disp["Train Time (s)"] = disp["Train Time (s)"].apply(lambda v: f"{v:.3f}s")
    st.dataframe(disp, use_container_width=True, hide_index=True)

    # ── Model Performance ───────────────────────────────────────────────────
    sec("Model Performance", "#60a5fa")
    pt1, pt2 = st.tabs(["📊 Grouped Bar", "🕸 Radar Chart"])

    with pt1:
        melt = res.melt(id_vars="Model",
            value_vars=["Accuracy","Precision","Recall","F1 Score"],
            var_name="Metric", value_name="Score")
        fig_bar = px.bar(melt, x="Model", y="Score", color="Metric", barmode="group",
            color_discrete_map={"Accuracy":"#2dd4bf","Precision":"#60a5fa",
                                "Recall":"#f59e0b","F1 Score":"#4ade80"},
            text_auto=".3f")
        fig_bar.update_traces(textfont_size=9, textposition='outside',
                              cliponaxis=False, textfont_color='#e2e8f0')
        fig_bar.update_yaxes(range=[0, 1.12])
        sf(fig_bar)
        st.plotly_chart(fig_bar, use_container_width=True)

    with pt2:
        met   = ["Accuracy","Precision","Recall","F1 Score"]
        fig_r = go.Figure()
        for idx, row in res.iterrows():
            v = [row[m] for m in met] + [row[met[0]]]
            fig_r.add_trace(go.Scatterpolar(
                r=v, theta=met + [met[0]],
                fill='toself', name=row["Model"],
                line_color=VP[idx % len(VP)], opacity=.70,
            ))
        fig_r.update_layout(
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(visible=True, range=[0,1], gridcolor='#2a3245',
                                linecolor='#2a3245', tickfont=dict(color='#94a3b8',size=9)),
                angularaxis=dict(gridcolor='#2a3245', linecolor='#2a3245',
                                 tickfont=dict(color='#94a3b8',size=11)),
            ), **PT)
        st.plotly_chart(fig_r, use_container_width=True)

    # ── Accuracy Leaderboard ────────────────────────────────────────────────
    sec("Accuracy Leaderboard", "#f59e0b")
    lb = res.sort_values("Accuracy")
    fig_lb = go.Figure()
    for idx, row in lb.iterrows():
        clr = VP[int(row.name) % len(VP)]
        fig_lb.add_trace(go.Bar(
            y=[row["Model"]], x=[row["Accuracy"]], orientation='h',
            marker=dict(color=clr, opacity=0.85),
            text=f"  {row['Accuracy']:.4f}", textposition='outside',
            textfont=dict(color='#94a3b8', size=12, family='DM Mono'),
            name=row["Model"], showlegend=False,
        ))
    fig_lb.update_xaxes(range=[0, 1.12])
    fig_lb.update_layout(barmode='stack', height=320, **PT)
    st.plotly_chart(fig_lb, use_container_width=True)

    # ── Training Time ───────────────────────────────────────────────────────
    sec("Training Time", "#a78bfa")
    tt = res.sort_values("Train Time (s)", ascending=True)
    fig_tt = px.bar(tt, x="Train Time (s)", y="Model", orientation='h',
        color="Train Time (s)",
        color_continuous_scale=[[0,'#2dd4bf'],[0.5,'#60a5fa'],[1,'#a78bfa']],
        text=tt["Train Time (s)"].apply(lambda v: f"{v:.3f}s"))
    fig_tt.update_traces(textposition='outside', textfont_color='#94a3b8')
    fig_tt.update_coloraxes(showscale=False)
    sf(fig_tt)
    st.plotly_chart(fig_tt, use_container_width=True)

    # ── Cross-Validation ────────────────────────────────────────────────────
    sec(f"Cross-Validation  ({CV_FOLDS}-Fold Stratified KFold)", "#4ade80")
    cv_data = [
        {"Model": n, "Fold": f"F{i+1}", "Score": s}
        for n, scores in cv_scores.items()
        for i, s in enumerate(scores)
    ]
    cv_df = pd.DataFrame(cv_data)

    cb1, cb2 = st.columns(2)
    with cb1:
        fig_box = px.box(cv_df, x="Model", y="Score", color="Model",
            color_discrete_sequence=VP, title="Score Distribution — Box Plot")
        fig_box.update_layout(showlegend=False)
        sf(fig_box)
        st.plotly_chart(fig_box, use_container_width=True)
    with cb2:
        fig_vio = px.violin(cv_df, x="Model", y="Score", color="Model",
            box=True, points="all",
            color_discrete_sequence=VP, title="Score Distribution — Violin Plot")
        fig_vio.update_layout(showlegend=False)
        sf(fig_vio)
        st.plotly_chart(fig_vio, use_container_width=True)

    cv_sum = cv_df.groupby("Model")["Score"].agg(["mean","std"]).reset_index()
    cv_sum.columns = ["Model","Mean","Std"]
    cv_sum = cv_sum.sort_values("Mean", ascending=False).reset_index(drop=True)
    fig_cv = go.Figure()
    for idx, row in cv_sum.iterrows():
        c = VP[idx % len(VP)]
        fig_cv.add_trace(go.Bar(
            x=[row["Model"]], y=[row["Mean"]],
            error_y=dict(type='data', array=[row["Std"]], visible=True,
                         color='rgba(255,255,255,.35)', thickness=1.5),
            marker_color=c,
            text=f"{row['Mean']:.4f} ±{row['Std']:.4f}",
            textposition='outside',
            textfont=dict(color='#94a3b8', size=10),
            name=row["Model"], showlegend=False,
        ))
    fig_cv.update_layout(yaxis_range=[0,1.12], title_text="Mean CV Accuracy ± Std Dev", **PT)
    st.plotly_chart(fig_cv, use_container_width=True)

    # ── ROC Curves ──────────────────────────────────────────────────────────
    sec("ROC Curves  (Macro One-vs-Rest)", "#38bdf8")
    n_cls      = len(np.unique(y))
    y_test_bin = label_binarize(y_test, classes=np.arange(n_cls))
    fig_roc = go.Figure()
    for idx, (name, mdl) in enumerate(trained_models.items()):
        if hasattr(mdl, "predict_proba"):
            yp_prob     = mdl.predict_proba(X_te)
            fpr, tpr, _ = roc_curve(y_test_bin.ravel(), yp_prob.ravel())
            ra           = auc(fpr, tpr)
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode='lines',
                name=f"{name}  (AUC = {ra:.3f})",
                line=dict(color=VP[idx % len(VP)], width=2),
            ))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
        line=dict(dash='dot', color='#4a5568', width=1), name="Random Chance"))
    fig_roc.update_layout(xaxis_title="False Positive Rate",
                          yaxis_title="True Positive Rate",
                          title_text="ROC Curves — All Models", **PT)
    st.plotly_chart(fig_roc, use_container_width=True)

    # ── Confusion Matrix ────────────────────────────────────────────────────
    sec("Confusion Matrix", "#60a5fa")
    cm_a, cm_b = st.columns([1, 3])
    with cm_a:
        sel_cm = st.selectbox("Select Model", list(trained_models.keys()), key="cm")
        norm   = st.radio("View", ["Raw Counts", "Normalised"], horizontal=True)
    with cm_b:
        yp_cm = trained_models[sel_cm].predict(X_te)
        cm_m  = confusion_matrix(y_test, yp_cm)
        if norm == "Normalised":
            cm_p  = cm_m.astype(float) / cm_m.sum(axis=1, keepdims=True)
            texts = [[f"{v:.2f}" for v in row] for row in cm_p]
        else:
            cm_p  = cm_m
            texts = [[str(v) for v in row] for row in cm_p]
        fig_cm = go.Figure(go.Heatmap(
            z=cm_p, x=class_names, y=class_names,
            text=texts, texttemplate="%{text}",
            textfont=dict(size=8, color='#e2e8f0'),
            colorscale=[[0.0,'#0d1117'],[0.3,'#0e2a33'],[0.6,'#164e5a'],[1.0,'#2dd4bf']],
            showscale=True,
            colorbar=dict(tickfont=dict(color='#94a3b8'), outlinewidth=0),
        ))
        fig_cm.update_layout(**PT)
        fig_cm.update_layout(
            title_text=f"Confusion Matrix — {sel_cm}",
            xaxis=dict(**_AXIS, title="Predicted", tickangle=-35),
            yaxis=dict(**_AXIS, title="Actual"),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # ── Feature Importance ──────────────────────────────────────────────────
    tree_mods = {k: v for k, v in trained_models.items() if hasattr(v, 'feature_importances_')}
    if tree_mods:
        sec("Feature Importance", "#f59e0b")
        fi_name = st.selectbox("Select Model", list(tree_mods.keys()), key="fi")
        imp     = tree_mods[fi_name].feature_importances_
        fi_df   = pd.DataFrame({"Feature": X.columns, "Importance": imp}).sort_values("Importance")
        fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation='h',
            color="Importance",
            color_continuous_scale=[[0,'#0e2a33'],[0.5,'#164e5a'],[1,'#2dd4bf']],
            text=fi_df["Importance"].apply(lambda v: f"{v:.4f}"))
        fig_fi.update_coloraxes(showscale=False)
        fig_fi.update_traces(textposition='outside', textfont_color='#94a3b8')
        fig_fi.update_layout(yaxis_title="", **PT)
        st.plotly_chart(fig_fi, use_container_width=True)

    # ── Class Distribution ──────────────────────────────────────────────────
    sec("Class Distribution", "#4ade80")
    dist    = pd.DataFrame({"Crop": class_names, "Count": np.bincount(y)}).sort_values("Count", ascending=False)
    ext_pal = (VP * ((len(class_names) // len(VP)) + 2))[:len(class_names)]
    fig_dist = px.bar(dist, x="Crop", y="Count", color="Crop",
        color_discrete_sequence=ext_pal, text="Count")
    fig_dist.update_traces(textposition='outside', textfont_color='#94a3b8')
    fig_dist.update_layout(showlegend=False, xaxis_tickangle=-40, **PT)
    st.plotly_chart(fig_dist, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 : CROP PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
with nav_tab2:

    sec("Crop Prediction Panel", "#2dd4bf")

    # Model selector + info
    pred_col1, pred_col2 = st.columns([2, 1])

    with pred_col2:
        sel_model = st.selectbox(
            "Choose Prediction Model",
            list(trained_models.keys()),
            index=0,
            key="pred_model"
        )
        sel_acc = res.loc[res["Model"] == sel_model, "Accuracy"].values[0]
        sel_f1  = res.loc[res["Model"] == sel_model, "F1 Score"].values[0]

        st.markdown(f"""
        <div style="background:var(--card);border:1px solid var(--border);border-radius:10px;padding:1rem 1.2rem;margin-top:.8rem;">
          <div style="font-size:.62rem;text-transform:uppercase;letter-spacing:.1em;color:var(--text-dim);font-family:'DM Mono',monospace;margin-bottom:.8rem;">Model Info</div>
          <div style="display:flex;justify-content:space-between;margin-bottom:.4rem;">
            <span style="font-size:.75rem;color:var(--text-sec);">Accuracy</span>
            <span style="font-size:.75rem;color:var(--teal);font-family:'DM Mono',monospace;">{sel_acc:.4f}</span>
          </div>
          <div style="background:rgba(255,255,255,.05);border-radius:4px;height:5px;margin-bottom:.8rem;overflow:hidden;">
            <div style="width:{sel_acc*100:.1f}%;height:100%;background:linear-gradient(90deg,#2dd4bf,#60a5fa);border-radius:4px;"></div>
          </div>
          <div style="display:flex;justify-content:space-between;">
            <span style="font-size:.75rem;color:var(--text-sec);">F1 Score</span>
            <span style="font-size:.75rem;color:var(--amber);font-family:'DM Mono',monospace;">{sel_f1:.4f}</span>
          </div>
          <div style="background:rgba(255,255,255,.05);border-radius:4px;height:5px;margin-top:.4rem;overflow:hidden;">
            <div style="width:{sel_f1*100:.1f}%;height:100%;background:linear-gradient(90deg,#f59e0b,#f87171);border-radius:4px;"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:var(--card);border:1px solid var(--border);border-radius:10px;padding:1rem 1.2rem;margin-top:.8rem;">
          <div style="font-size:.62rem;text-transform:uppercase;letter-spacing:.1em;color:var(--text-dim);font-family:'DM Mono',monospace;margin-bottom:.6rem;">Feature Guide</div>
          <div style="font-size:.72rem;color:var(--text-sec);line-height:1.7;">
            <b style="color:var(--teal);">N</b> — Nitrogen (0–140 kg/ha)<br>
            <b style="color:var(--teal);">P</b> — Phosphorus (5–145 kg/ha)<br>
            <b style="color:var(--teal);">K</b> — Potassium (5–205 kg/ha)<br>
            <b style="color:var(--blue);">Temp</b> — °C<br>
            <b style="color:var(--blue);">Humidity</b> — %<br>
            <b style="color:var(--blue);">pH</b> — Soil pH (3.5–9.9)<br>
            <b style="color:var(--amber);">Rainfall</b> — mm/year
          </div>
        </div>
        """, unsafe_allow_html=True)

    with pred_col1:
        st.markdown('<div class="input-panel">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:.7rem;text-transform:uppercase;letter-spacing:.12em;color:var(--text-dim);font-family:DM Mono,monospace;margin-bottom:1rem;">🌱 Soil & Climate Parameters</div>', unsafe_allow_html=True)

        FEATURE_CONFIG = {
            "N":        {"label": "Nitrogen (N)",        "min": 0.0,   "max": 140.0, "step": 1.0,  "fmt": "%.0f"},
            "P":        {"label": "Phosphorus (P)",      "min": 5.0,   "max": 145.0, "step": 1.0,  "fmt": "%.0f"},
            "K":        {"label": "Potassium (K)",       "min": 5.0,   "max": 205.0, "step": 1.0,  "fmt": "%.0f"},
            "temperature": {"label": "Temperature (°C)", "min": 8.0,   "max": 44.0,  "step": 0.1,  "fmt": "%.1f"},
            "humidity": {"label": "Humidity (%)",        "min": 14.0,  "max": 100.0, "step": 0.1,  "fmt": "%.1f"},
            "ph":       {"label": "Soil pH",             "min": 3.5,   "max": 9.9,   "step": 0.01, "fmt": "%.2f"},
            "rainfall": {"label": "Rainfall (mm/year)",  "min": 20.0,  "max": 300.0, "step": 0.1,  "fmt": "%.1f"},
        }

        feat_cols = list(X.columns)
        input_vals = {}

        row_a, row_b, row_c = st.columns(3), st.columns(3), st.columns(1)
        col_groups = [row_a, row_b, [row_c[0]]]
        all_cols = [row_a[0], row_a[1], row_a[2], row_b[0], row_b[1], row_b[2], row_c[0]]

        for ci, feat in enumerate(feat_cols):
            cfg  = FEATURE_CONFIG.get(feat, {"label": feat, "min": feat_stats[feat]["min"],
                                             "max": feat_stats[feat]["max"], "step": 0.01, "fmt": "%.2f"})
            mean = feat_stats[feat]["mean"]
            with all_cols[ci]:
                val = st.number_input(
                    cfg["label"],
                    min_value=cfg["min"],
                    max_value=cfg["max"],
                    value=round(mean, 2),
                    step=cfg["step"],
                    format=cfg["fmt"],
                    key=f"feat_{feat}"
                )
                input_vals[feat] = val

        st.markdown('</div>', unsafe_allow_html=True)

        # Preset buttons
        st.markdown('<div style="margin:.6rem 0 .3rem;font-size:.65rem;color:var(--text-dim);text-transform:uppercase;letter-spacing:.1em;">Quick Presets</div>', unsafe_allow_html=True)
        presets = {
            "🌾 Rice":   {"N":80,"P":40,"K":40,"temperature":23,"humidity":82,"ph":6.5,"rainfall":200},
            "🌽 Maize":  {"N":80,"P":40,"K":20,"temperature":22,"humidity":65,"ph":6.0,"rainfall":80},
            "🍌 Banana": {"N":100,"P":75,"K":50,"temperature":27,"humidity":80,"ph":6.5,"rainfall":100},
            "☕ Coffee": {"N":100,"P":30,"K":30,"temperature":25,"humidity":70,"ph":6.5,"rainfall":150},
        }
        pb1, pb2, pb3, pb4 = st.columns(4)
        for btn, (label, pvals) in zip([pb1,pb2,pb3,pb4], presets.items()):
            if btn.button(label, use_container_width=True):
                for k, v in pvals.items():
                    st.session_state[f"feat_{k}"] = float(v)
                st.rerun()

        st.write("")
        predict_btn = st.button("🔍 Predict Crop", use_container_width=True, type="primary")

    # ── Prediction Result ───────────────────────────────────────────────────
    if predict_btn:
        model = trained_models[sel_model]
        input_arr = np.array([[input_vals[f] for f in feat_cols]])
        input_scaled = scaler.transform(input_arr)

        pred_idx  = model.predict(input_scaled)[0]
        pred_crop = le.inverse_transform([pred_idx])[0]
        emoji     = CROP_EMOJI.get(pred_crop.lower(), "🌿")

        # Confidence scores
        if hasattr(model, "predict_proba"):
            probs     = model.predict_proba(input_scaled)[0]
            conf      = probs[pred_idx]
            top_k_idx = np.argsort(probs)[::-1][:5]
            top_k     = [(le.inverse_transform([i])[0], probs[i]) for i in top_k_idx]
        else:
            conf  = 1.0
            top_k = [(pred_crop, 1.0)]

        st.markdown(f"""
        <div class="pred-result">
          <div class="crop-icon">{emoji}</div>
          <div class="crop-name">{pred_crop.title()}</div>
          <div class="crop-sub">Recommended Crop · {sel_model}</div>
          <div class="conf-bar-wrap">
            <div class="conf-bar" style="width:{conf*100:.1f}%"></div>
          </div>
          <div class="conf-txt">Confidence: {conf*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        if len(top_k) > 1:
            st.markdown('<div style="margin-top:1rem;font-size:.65rem;color:var(--text-dim);text-transform:uppercase;letter-spacing:.1em;">Top-5 Alternatives</div>', unsafe_allow_html=True)
            pills_html = ""
            for crop, prob in top_k:
                em = CROP_EMOJI.get(crop.lower(), "🌿")
                pills_html += f'<span class="alt-pill">{em} {crop.title()} <b>{prob*100:.1f}%</b></span>'
            st.markdown(f'<div style="margin-top:.5rem;">{pills_html}</div>', unsafe_allow_html=True)

        # Probability bar chart
        if hasattr(model, "predict_proba") and len(top_k) > 1:
            st.markdown('<br>', unsafe_allow_html=True)
            top10_idx   = np.argsort(probs)[::-1][:10]
            top10_crops = [le.inverse_transform([i])[0].title() for i in top10_idx]
            top10_probs = [probs[i] * 100 for i in top10_idx]

            fig_prob = go.Figure()
            for ci, (cn, cp) in enumerate(zip(top10_crops, top10_probs)):
                clr = VP[0] if ci == 0 else VP[2]
                fig_prob.add_trace(go.Bar(
                    x=[cn], y=[cp],
                    marker_color=clr, opacity=0.85 if ci == 0 else 0.5,
                    text=f"{cp:.1f}%", textposition='outside',
                    textfont=dict(color='#94a3b8', size=10),
                    name=cn, showlegend=False,
                ))
            fig_prob.update_layout(title_text="Probability Distribution — Top 10 Crops",
                                   yaxis_title="Probability (%)", yaxis_range=[0, max(top10_probs)*1.25],
                                   **PT)
            st.plotly_chart(fig_prob, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 : DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with nav_tab3:

    sec("Dataset Overview", "#2dd4bf")
    st.dataframe(data.head(50), use_container_width=True, hide_index=True)

    sec("Statistical Summary", "#60a5fa")
    st.dataframe(data.describe().round(3), use_container_width=True)

    sec("Feature Correlations", "#f59e0b")
    corr = X.corr()
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns, y=corr.columns,
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}",
        textfont=dict(size=10, color='#e2e8f0'),
        colorscale=[
            [0.0, '#164e5a'], [0.5, '#0d1117'], [1.0, '#2dd4bf']
        ],
        zmid=0,
        colorbar=dict(tickfont=dict(color='#94a3b8'), outlinewidth=0),
    ))
    fig_corr.update_layout(title_text="Feature Correlation Matrix", **PT)
    st.plotly_chart(fig_corr, use_container_width=True)

    sec("Feature Distributions by Crop", "#4ade80")
    feat_sel = st.selectbox("Feature", list(X.columns), key="feat_dist")
    fig_kde = px.box(data, x="label", y=feat_sel, color="label",
        color_discrete_sequence=(VP * 5)[:len(class_names)],
        title=f"{feat_sel} Distribution by Crop")
    fig_kde.update_layout(showlegend=False, xaxis_tickangle=-40, **PT)
    st.plotly_chart(fig_kde, use_container_width=True)

    sec("Scatter — Feature vs Feature", "#a78bfa")
    sc1, sc2 = st.columns(2)
    fx = sc1.selectbox("X Axis", list(X.columns), index=0, key="scx")
    fy = sc2.selectbox("Y Axis", list(X.columns), index=1, key="scy")
    fig_sc = px.scatter(data, x=fx, y=fy, color="label",
        color_discrete_sequence=(VP * 5)[:len(class_names)],
        opacity=0.7, title=f"{fx} vs {fy}")
    sf(fig_sc)
    st.plotly_chart(fig_sc, use_container_width=True)


# ── FOOTER ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  CropAI Dashboard &nbsp;·&nbsp; Scikit-Learn &nbsp;·&nbsp;
  XGBoost &nbsp;·&nbsp; Streamlit &nbsp;·&nbsp; Plotly
</div>
""", unsafe_allow_html=True)