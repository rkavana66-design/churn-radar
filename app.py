"""
Predictive Analytics in Telecommunication using Naive Bayes Classifier
Big Data & Tools mini-project: Streamlit + Apache Spark (PySpark) with a scikit-learn fallback.
Run locally:  streamlit run app.py
"""
import os
import sys
import time
import shutil

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score

# --------------------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Telecom Command Center | Naive Bayes Churn",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------------------
# Custom CSS: dark command-center theme, glassmorphism, floating animations
# --------------------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=DM+Sans:wght@400;500;600&display=swap');

:root{
  --bg:#090d16; --cyan:#00f5d4; --blue:#3b82f6; --text:#e6edf7; --muted:#8a99b3;
  --glass:rgba(17,25,43,.58); --line:rgba(0,245,212,.22);
}

/* ---- Chrome removal ---- */
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"]{display:none !important; visibility:hidden !important;}
.block-container{padding-top:1.2rem !important; padding-bottom:3rem; max-width:1280px;}

/* ---- Base ---- */
.stApp{
  background:
    radial-gradient(circle at 10% 6%, rgba(59,130,246,.20), transparent 38%),
    radial-gradient(circle at 92% 0%, rgba(0,245,212,.13), transparent 34%),
    radial-gradient(circle at 50% 100%, rgba(59,130,246,.10), transparent 40%),
    #090d16;
  color:var(--text);
  font-family:'DM Sans',sans-serif;
}
h1,h2,h3,h4{font-family:'Space Grotesk',sans-serif !important; color:var(--text) !important; letter-spacing:-.01em;}
p,li,label,span,div{font-family:'DM Sans',sans-serif;}
[data-testid="stWidgetLabel"] p{color:#b8c6de !important; font-weight:500;}
.stCaption, [data-testid="stCaptionContainer"]{color:var(--muted) !important;}

/* ---- Keyframes ---- */
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-9px)}}
@keyframes glowPulse{0%,100%{box-shadow:0 0 10px rgba(0,245,212,.35),0 0 0 rgba(0,245,212,0)}50%{box-shadow:0 0 22px rgba(0,245,212,.75),0 0 40px rgba(59,130,246,.25)}}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(var(--glow),.55)}50%{box-shadow:0 0 0 14px rgba(var(--glow),0)}}
@keyframes sheen{0%{background-position:0% 50%}100%{background-position:200% 50%}}

/* ---- Hero ---- */
.hero{padding:10px 4px 6px 4px; margin-bottom:6px;}
.hero-top{display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; margin-bottom:14px;}
.hero-badge{
  display:inline-block; padding:8px 18px; border-radius:999px; font-size:.82rem; font-weight:600; color:var(--cyan);
  background:rgba(0,245,212,.08); border:1px solid rgba(0,245,212,.45); backdrop-filter:blur(12px);
  animation:float 5s ease-in-out infinite, glowPulse 3.5s ease-in-out infinite;
}
.status-pill{
  display:inline-flex; align-items:center; gap:8px; padding:8px 16px; border-radius:999px; font-size:.82rem; font-weight:600;
  backdrop-filter:blur(12px); animation:glowPulse 2.4s ease-in-out infinite;
}
.status-spark{color:#00f5d4; background:rgba(0,245,212,.10); border:1px solid rgba(0,245,212,.6);}
.status-fallback{color:#fbbf24; background:rgba(251,191,36,.10); border:1px solid rgba(251,191,36,.6); --c:251,191,36;}
.hero h1{font-size:2.6rem; line-height:1.1; margin:0 0 6px 0; font-weight:700;
  background:linear-gradient(90deg,#e6edf7 0%,#00f5d4 45%,#3b82f6 100%); background-size:200% auto;
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; animation:sheen 8s linear infinite;}
.hero p{color:var(--muted); font-size:1.02rem; max-width:760px; margin:0;}

/* ---- KPI strip ---- */
.kpi-grid{display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin:18px 0 20px 0;}
.kpi-grid.mini{grid-template-columns:repeat(4,1fr); margin:8px 0 14px 0;}
.kpi-card{
  background:var(--glass); backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px);
  border:1px solid var(--line); border-radius:18px; padding:18px 20px;
  box-shadow:0 8px 30px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.04);
  animation:float 6s ease-in-out infinite; transition:box-shadow .3s ease, border-color .3s ease;
}
.kpi-card:hover{animation-play-state:paused; border-color:var(--cyan); box-shadow:0 0 28px rgba(0,245,212,.35), 0 8px 30px rgba(0,0,0,.4);}
.kpi-label{color:var(--muted); font-size:.82rem; font-weight:500; margin-bottom:6px;}
.kpi-value{font-family:'Space Grotesk',sans-serif; font-size:2.1rem; font-weight:700; color:var(--cyan); line-height:1.1; text-shadow:0 0 18px rgba(0,245,212,.35);}
.kpi-card.blue .kpi-value{color:#7db1ff; text-shadow:0 0 18px rgba(59,130,246,.45);}
.kpi-sub{color:#6f7e99; font-size:.76rem; margin-top:6px;}
.mini .kpi-value{font-size:1.6rem}

/* ---- Glass panels ---- */
.glass{
  background:var(--glass); backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px);
  border:1px solid var(--line); border-radius:18px; padding:20px 22px; margin-bottom:14px;
  box-shadow:0 8px 30px rgba(0,0,0,.30);
}
div[data-testid="stVerticalBlockBorderWrapper"]{
  background:var(--glass) !important; backdrop-filter:blur(12px); border:1px solid var(--line) !important;
  border-radius:18px !important; box-shadow:0 8px 30px rgba(0,0,0,.30);
}

/* ---- Risk banner ---- */
.risk-banner{padding:16px 20px; border-radius:14px; font-weight:600; font-size:1.05rem; border:1px solid currentColor; animation:pulse 1.8s ease-in-out infinite; margin-bottom:14px;}
.risk-High{color:#ff4d6d; background:rgba(255,77,109,.12); --glow:255,77,109;}
.risk-Medium{color:#fbbf24; background:rgba(251,191,36,.11); --glow:251,191,36;}
.risk-Low{color:#34d399; background:rgba(52,211,153,.10); --glow:52,211,153;}

/* ---- Action plan ---- */
.plan-item{display:flex; gap:14px; align-items:flex-start; padding:12px 14px; margin-bottom:10px; border-radius:12px;
  background:rgba(9,13,22,.55); border:1px solid rgba(59,130,246,.25); transition:border-color .25s, transform .25s;}
.plan-item:hover{border-color:var(--cyan); transform:translateX(4px);}
.plan-tag{flex:0 0 auto; padding:3px 10px; border-radius:999px; font-size:.72rem; font-weight:600; color:#06121f;
  background:linear-gradient(90deg,#00f5d4,#3b82f6);}
.plan-title{font-weight:600; color:var(--text); font-size:.95rem}
.plan-detail{color:var(--muted); font-size:.86rem; margin-top:2px}

/* ---- Architecture cards ---- */
.step-grid{display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin:10px 0 18px 0;}
.step-card{background:var(--glass); backdrop-filter:blur(12px); border:1px solid var(--line); border-radius:16px; padding:18px; transition:transform .3s, box-shadow .3s, border-color .3s;}
.step-card:hover{transform:translateY(-5px); box-shadow:0 0 26px rgba(59,130,246,.35); border-color:var(--blue);}
.step-num{font-family:'Space Grotesk',sans-serif; font-size:1.5rem; font-weight:700; color:var(--cyan);}
.step-card h4{margin:2px 0 6px 0; font-size:1.02rem;}
.step-card p{color:var(--muted); font-size:.88rem; margin:0;}
.phase{display:inline-block; font-size:.7rem; font-weight:600; padding:2px 9px; border-radius:999px; margin-bottom:6px;}
.phase-map{color:#00f5d4; border:1px solid rgba(0,245,212,.5); background:rgba(0,245,212,.08);}
.phase-reduce{color:#7db1ff; border:1px solid rgba(59,130,246,.55); background:rgba(59,130,246,.10);}
.phase-driver{color:#c4b5fd; border:1px solid rgba(167,139,250,.5); background:rgba(167,139,250,.08);}

.flow{display:grid; grid-template-columns:1fr auto 1.35fr auto 1.1fr; gap:10px; align-items:stretch; margin:8px 0 6px 0;}
.node{background:rgba(9,13,22,.6); border:1px solid rgba(59,130,246,.35); border-radius:12px; padding:10px 14px; font-size:.86rem; display:flex; flex-direction:column; justify-content:center;}
.node span{color:var(--muted); font-size:.76rem; margin-top:2px}
.node.map{border-color:rgba(0,245,212,.5)}
.node.reduce{border-color:rgba(59,130,246,.7); grid-column:5; grid-row:1 / span 3; text-align:center; align-items:center; background:rgba(59,130,246,.08); box-shadow:0 0 24px rgba(59,130,246,.25);}
.arrow{color:var(--cyan); align-self:center; font-size:1.2rem}

/* ---- Widgets ---- */
.stButton>button, .stDownloadButton>button{
  color:var(--text); font-weight:600; border-radius:12px; padding:.6rem 1.1rem; border:1.5px solid transparent;
  background:linear-gradient(#0d1424,#0d1424) padding-box, linear-gradient(90deg,#00f5d4,#3b82f6) border-box;
  transition:transform .25s ease, box-shadow .25s ease, color .25s ease;
}
.stButton>button:hover, .stDownloadButton>button:hover{
  transform:translateY(-2px); color:#06121f; border-color:transparent;
  background:linear-gradient(90deg,#00f5d4,#3b82f6) padding-box, linear-gradient(90deg,#00f5d4,#3b82f6) border-box;
  box-shadow:0 6px 26px rgba(0,245,212,.45);
}
.stButton>button:hover p, .stDownloadButton>button:hover p{color:#06121f !important;}

div[data-baseweb="tab-list"]{gap:6px; background:rgba(17,25,43,.55); padding:6px; border-radius:14px; border:1px solid var(--line); backdrop-filter:blur(12px);}
button[data-baseweb="tab"]{border-radius:10px !important; padding:10px 18px !important; background:transparent !important; transition:background .25s, color .25s, box-shadow .25s;}
button[data-baseweb="tab"] p{color:var(--muted); font-weight:600;}
button[data-baseweb="tab"]:hover{background:rgba(59,130,246,.14) !important;}
button[data-baseweb="tab"]:hover p{color:var(--text);}
button[data-baseweb="tab"][aria-selected="true"]{background:rgba(0,245,212,.10) !important; box-shadow:inset 0 0 0 1px rgba(0,245,212,.5);}
button[data-baseweb="tab"][aria-selected="true"] p{color:var(--cyan);}
div[data-baseweb="tab-highlight"]{background:linear-gradient(90deg,#00f5d4,#3b82f6) !important; height:3px !important; border-radius:3px;}
div[data-baseweb="tab-border"]{display:none;}

div[data-baseweb="select"]>div{background:rgba(9,13,22,.7) !important; border:1px solid rgba(59,130,246,.4) !important; border-radius:10px !important; transition:border-color .25s, box-shadow .25s;}
div[data-baseweb="select"]>div:hover{border-color:var(--cyan) !important; box-shadow:0 0 14px rgba(0,245,212,.25);}
ul[role="listbox"]{background:#0d1424 !important;}

div[data-testid="stSlider"] div[role="slider"]{background:var(--cyan) !important; box-shadow:0 0 14px rgba(0,245,212,.8); transition:transform .2s;}
div[data-testid="stSlider"] div[role="slider"]:hover{transform:scale(1.25);}
div[data-testid="stSliderThumbValue"], div[data-testid="stSliderThumbValue"] *{color:var(--cyan) !important; font-weight:600;}

[data-testid="stFileUploaderDropzone"]{
  background:rgba(17,25,43,.55) !important; border:1.5px dashed rgba(0,245,212,.5) !important; border-radius:16px !important;
  backdrop-filter:blur(12px); transition:all .3s ease;
}
[data-testid="stFileUploaderDropzone"]:hover{
  border:1.5px solid transparent !important;
  background:linear-gradient(#0d1424,#0d1424) padding-box, linear-gradient(90deg,#00f5d4,#3b82f6) border-box !important;
  box-shadow:0 0 28px rgba(0,245,212,.3);
}
[data-testid="stFileUploaderDropzone"] button{border-radius:10px;}
[data-testid="stDataFrame"]{border:1px solid var(--line); border-radius:12px; overflow:hidden;}
[data-testid="stRadio"] label{color:var(--text) !important;}
code, pre{font-family:'JetBrains Mono','Fira Code',monospace !important;}

@media (max-width:900px){
  .kpi-grid,.kpi-grid.mini{grid-template-columns:repeat(2,1fr);}
  .step-grid{grid-template-columns:1fr;}
  .flow{grid-template-columns:1fr; }
  .node.reduce{grid-column:auto; grid-row:auto;}
  .arrow{transform:rotate(90deg); justify-self:center;}
  .hero h1{font-size:1.9rem;}
}
@media (prefers-reduced-motion:reduce){
  .kpi-card,.hero-badge,.status-pill,.risk-banner,.hero h1{animation:none !important;}
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# Domain constants + synthetic CDR generator
# --------------------------------------------------------------------------------------
CONTRACTS = ["Month-to-Month", "One Year", "Two Year"]
INTERNET = ["Fiber Optic", "DSL", "None"]
HIGH_THRESHOLD, MED_THRESHOLD = 0.60, 0.30

CONTRACT_ALIASES = {
    "month-to-month": "Month-to-Month", "month to month": "Month-to-Month", "monthly": "Month-to-Month", "m2m": "Month-to-Month",
    "one year": "One Year", "1 year": "One Year", "one-year": "One Year", "1yr": "One Year",
    "two year": "Two Year", "2 year": "Two Year", "two-year": "Two Year", "2yr": "Two Year",
}
INTERNET_ALIASES = {
    "fiber optic": "Fiber Optic", "fiber": "Fiber Optic", "fibre": "Fiber Optic", "fibre optic": "Fiber Optic",
    "dsl": "DSL", "none": "None", "no": "None", "no internet": "None",
}
REQUIRED_COLS = ["contract_type", "internet_service", "tenure_months", "monthly_charges", "service_calls"]


def generate_cdr(n: int, seed: int, labelled: bool = False, with_ids: bool = False) -> pd.DataFrame:
    """Simulate realistic telecom CDR-derived subscriber records with a latent churn process."""
    rng = np.random.default_rng(seed)
    contract = rng.choice(3, n, p=[0.55, 0.25, 0.20])
    internet = rng.choice(3, n, p=[0.44, 0.34, 0.22])
    tenure = np.clip(rng.gamma(2.0, 14.0, n), 1, 72).astype(int)
    base = np.array([85.0, 55.0, 25.0])[internet]
    spread = np.array([15.0, 12.0, 6.0])[internet]
    charges = np.clip(rng.normal(base, spread), 18, 120).round(2)
    calls = np.clip(rng.poisson(np.where(contract == 0, 1.6, 0.9)), 0, 10)
    logit = (
        -1.1
        + np.array([1.5, -0.4, -1.7])[contract]
        + np.array([0.75, 0.0, -0.7])[internet]
        - 0.035 * tenure
        + 0.014 * (charges - 65)
        + 0.5 * calls
        + rng.normal(0, 0.5, n)
    )
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({
        "contract_type": np.array(CONTRACTS)[contract],
        "internet_service": np.array(INTERNET)[internet],
        "tenure_months": tenure,
        "monthly_charges": charges,
        "service_calls": calls,
    })
    if with_ids:
        df.insert(0, "subscriber_id", [f"SUB-{i:06d}" for i in range(1, n + 1)])
    if labelled:
        df["churn"] = churn
    return df


def encode(df: pd.DataFrame) -> np.ndarray:
    """One-hot / binned non-negative feature matrix (20 columns), valid for multinomial Naive Bayes."""
    c = df["contract_type"].map({k: i for i, k in enumerate(CONTRACTS)}).to_numpy().astype(int)
    i = df["internet_service"].map({k: j for j, k in enumerate(INTERNET)}).to_numpy().astype(int)
    t = np.digitize(df["tenure_months"].to_numpy(dtype=float), [6, 12, 24, 48])      # 5 bins
    m = np.digitize(df["monthly_charges"].to_numpy(dtype=float), [40, 60, 80, 100])  # 5 bins
    s = np.digitize(df["service_calls"].to_numpy(dtype=float), [1, 3, 5])             # 4 bins
    return np.hstack([np.eye(3)[c], np.eye(3)[i], np.eye(5)[t], np.eye(5)[m], np.eye(4)[s]])


# --------------------------------------------------------------------------------------
# ML engines: Apache Spark (primary) and scikit-learn (fallback)
# --------------------------------------------------------------------------------------
class SparkEngine:
    name = "Apache Spark"

    def __init__(self, X, y):
        from pyspark.sql import SparkSession
        from pyspark.ml.feature import VectorAssembler
        from pyspark.ml.classification import NaiveBayes

        os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
        os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
        self.spark = (
            SparkSession.builder.master("local[*]")
            .appName("TelecomChurnNaiveBayes")
            .config("spark.driver.memory", "1g")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.driver.host", "127.0.0.1")
            .getOrCreate()
        )
        self.spark.sparkContext.setLogLevel("ERROR")
        self.version = self.spark.version
        self.cols = [f"f{k}" for k in range(X.shape[1])]
        self.assembler = VectorAssembler(inputCols=self.cols, outputCol="features")
        train = self.assembler.transform(self._frame(X, y))
        nb = NaiveBayes(smoothing=1.0, modelType="multinomial", featuresCol="features", labelCol="label")
        self.model = nb.fit(train)
        self.pi = np.array(self.model.pi.toArray())
        self.theta = np.array(self.model.theta.toArray())

    def _frame(self, X, y=None):
        ddl = ", ".join(f"{c} double" for c in self.cols) + (", label double" if y is not None else "")
        if y is None:
            rows = [tuple(r) for r in X.astype(float).tolist()]
        else:
            rows = [tuple(r) + (float(l),) for r, l in zip(X.astype(float).tolist(), y)]
        return self.spark.createDataFrame(rows, schema=ddl)

    def score(self, X):
        """Distributed batch inference: returns P(churn) for each row."""
        from pyspark.ml.functions import vector_to_array

        df = self.assembler.transform(self._frame(X))
        out = self.model.transform(df).select(vector_to_array("probability").getItem(1).alias("p")).collect()
        return np.array([r["p"] for r in out], dtype=float)

    def single(self, x):
        """Low-latency scoring for one subscriber using the parameters exported from the Spark model."""
        s = self.pi + self.theta @ x
        s = s - s.max()
        e = np.exp(s)
        return float(e[1] / e.sum())


class SklearnEngine:
    name = "scikit-learn"
    version = None

    def __init__(self, X, y):
        from sklearn.naive_bayes import GaussianNB

        self.clf = GaussianNB().fit(X, y)

    def score(self, X):
        return self.clf.predict_proba(X)[:, 1]

    def single(self, x):
        return float(self.clf.predict_proba(x.reshape(1, -1))[0, 1])


@st.cache_resource(show_spinner="Booting analytics engine and training Naive Bayes on 5,000 CDR records...")
def build_engine():
    pdf = generate_cdr(5000, seed=42, labelled=True)
    X, y = encode(pdf), pdf["churn"].to_numpy()
    idx = np.random.default_rng(0).permutation(len(X))
    cut = int(0.8 * len(X))
    tr, te = idx[:cut], idx[cut:]
    note = ""
    try:
        if shutil.which("java") is None:
            raise RuntimeError("Java runtime not detected on PATH")
        engine = SparkEngine(X[tr], y[tr])
    except Exception as exc:  # Java missing, Spark failed to launch, etc.
        note = f"{type(exc).__name__}: {str(exc)[:160]}"
        engine = SklearnEngine(X[tr], y[tr])
    pred = (engine.score(X[te]) >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y[te], pred),
        "precision": precision_score(y[te], pred, zero_division=0),
        "recall": recall_score(y[te], pred, zero_division=0),
        "churn_rate": float(y.mean()),
        "n_train": len(tr),
        "n_test": len(te),
    }
    return engine, metrics, note


engine, METRICS, ENGINE_NOTE = build_engine()
SPARK_ACTIVE = engine.name == "Apache Spark"

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def risk_level(p: float) -> str:
    return "High" if p >= HIGH_THRESHOLD else "Medium" if p >= MED_THRESHOLD else "Low"


def style_fig(fig, height=380):
    fig.update_layout(
        template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#cbd5e1"), margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def gauge_chart(prob: float):
    pct = prob * 100
    color = "#34d399" if pct < 30 else "#fbbf24" if pct < 60 else "#ff4d6d"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        number={"suffix": "%", "font": {"size": 54, "color": color, "family": "Space Grotesk"}, "valueformat": ".1f"},
        delta={"reference": METRICS["churn_rate"] * 100, "valueformat": ".1f", "suffix": " pts vs base rate",
               "increasing": {"color": "#ff4d6d"}, "decreasing": {"color": "#34d399"}},
        title={"text": "Churn probability", "font": {"size": 16, "color": "#8a99b3"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#475569", "tickfont": {"color": "#8a99b3"}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "rgba(52,211,153,.16)"},
                {"range": [30, 60], "color": "rgba(251,191,36,.16)"},
                {"range": [60, 100], "color": "rgba(255,77,109,.18)"},
            ],
            "threshold": {"line": {"color": "#00f5d4", "width": 4}, "thickness": 0.8, "value": pct},
        },
    ))
    style_fig(fig, 340)
    fig.update_layout(transition={"duration": 900, "easing": "cubic-in-out"}, margin=dict(l=30, r=30, t=60, b=10))
    return fig


def retention_plan(contract, internet, tenure, charges, calls, level):
    base = {
        "High": ("Now", "Assign to a retention specialist", "Call within 24 hours with a personalised save offer based on the account history."),
        "Medium": ("This week", "Send proactive outreach", "Share a usage review and a plan-fit recommendation by email and SMS."),
        "Low": ("Quarterly", "Keep in the nurture journey", "Include in loyalty campaigns and test an upsell to a higher tier."),
    }[level]
    plan = [base]
    if contract == "Month-to-Month" and level != "Low":
        plan.append(("Offer", "Move to a 12-month contract", "Offer a 10-15% discount or a device upgrade in exchange for a term commitment."))
    if calls >= 3:
        plan.append(("Service", "Resolve open service issues first", "Three or more service calls: escalate to tier-2 support and add a goodwill credit before any sales offer."))
    if internet == "Fiber Optic" and charges >= 90 and level != "Low":
        plan.append(("Pricing", "Review the bill for value", "Offer a bundle (mobile or TV add-on) or a loyalty credit to offset the high monthly charge."))
    if tenure < 12:
        plan.append(("Onboarding", "Run a first-year check-in", "Early-tenure subscribers are the most fragile: confirm coverage and speed satisfaction."))
    if contract == "Two Year" and level == "Low":
        plan.append(("Growth", "Promote add-on services", "A stable, committed subscriber is the best candidate for a premium or family plan."))
    return plan


def normalize_batch(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}. Expected: {', '.join(REQUIRED_COLS)}.")
    df["contract_type"] = df["contract_type"].astype(str).str.strip().str.lower().map(CONTRACT_ALIASES)
    df["internet_service"] = df["internet_service"].astype(str).str.strip().str.lower().map(INTERNET_ALIASES)
    for col, allowed in (("contract_type", CONTRACTS), ("internet_service", INTERNET)):
        if df[col].isna().any():
            raise ValueError(f"'{col}' has {int(df[col].isna().sum())} unrecognised value(s). Allowed: {', '.join(allowed)}.")
    for col in ("tenure_months", "monthly_charges", "service_calls"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            raise ValueError(f"'{col}' has {int(df[col].isna().sum())} non-numeric or empty value(s).")
    return df


def score_batch(df: pd.DataFrame, source: str) -> dict:
    clean = normalize_batch(df)
    X = encode(clean)
    t0 = time.perf_counter()
    p = engine.score(X)
    latency_ms = (time.perf_counter() - t0) * 1000
    out = clean.copy()
    out["churn_probability"] = np.round(p, 4)
    out["risk_level"] = np.select([p >= HIGH_THRESHOLD, p >= MED_THRESHOLD], ["High", "Medium"], default="Low")
    return {"df": out, "latency_ms": latency_ms, "source": source}


def recommended_action(row_df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.select(
            [
                row_df["service_calls"] >= 4,
                (row_df["contract_type"] == "Month-to-Month") & (row_df["internet_service"] == "Fiber Optic"),
                row_df["tenure_months"] < 12,
            ],
            [
                "Priority callback + service credit",
                "Offer 12-month contract with fiber loyalty discount",
                "First-year check-in + welcome bundle",
            ],
            default="Personalised loyalty offer",
        ),
        index=row_df.index,
    )


def kpi_card(label, value, sub, delay=0.0, blue=False):
    cls = "kpi-card blue" if blue else "kpi-card"
    return (f'<div class="{cls}" style="animation-delay:{delay}s"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>')


# --------------------------------------------------------------------------------------
# Hero + KPI strip
# --------------------------------------------------------------------------------------
if SPARK_ACTIVE:
    pill = '<span class="status-pill status-spark">⚡ Apache Spark Engine Active</span>'
else:
    pill = '<span class="status-pill status-fallback">🛡️ scikit-learn Fallback Engine Active</span>'

st.markdown(
    '<div class="hero"><div class="hero-top"><span class="hero-badge">📡 Big Data &amp; Tools Mini-Project</span>'
    + pill
    + '</div><h1>Predictive Analytics in Telecommunication</h1>'
    '<p>Naive Bayes churn prediction on Call Detail Records: score one subscriber live or a whole CDR batch in milliseconds.</p></div>',
    unsafe_allow_html=True,
)
if not SPARK_ACTIVE and ENGINE_NOTE:
    st.caption(f"Spark could not start ({ENGINE_NOTE}). Running the scikit-learn GaussianNB fallback instead.")

kpi_slot = st.empty()  # filled at the end so it reflects any batch scored in this run

# --------------------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🎯 Live Subscriber Risk Simulator",
    "🗄️ Big Data Batch CDR Scoring",
    "🧩 Distributed Spark & MapReduce Architecture",
])

# ---------------------------- Tab 1: Live simulator -----------------------------------
with tab1:
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("#### Subscriber profile")
            contract = st.selectbox("Contract Type", CONTRACTS, index=0)
            internet = st.selectbox("Internet Service", INTERNET, index=0)
            tenure = st.slider("Tenure (Months)", 1, 72, 5)
            charges = st.slider("Monthly Charges ($)", 18, 120, 95, format="$%d")
            calls = st.slider("Customer Service Calls", 0, 10, 4)

    row = pd.DataFrame([{
        "contract_type": contract, "internet_service": internet, "tenure_months": tenure,
        "monthly_charges": charges, "service_calls": calls,
    }])
    prob = engine.single(encode(row)[0])
    level = risk_level(prob)
    banner = {
        "High": "🚨 HIGH RISK: immediate retention intervention recommended",
        "Medium": "⚠️ MEDIUM RISK: proactive outreach recommended",
        "Low": "✅ LOW RISK: nurture and upsell",
    }[level]

    with right:
        st.markdown(f'<div class="risk-banner risk-{level}">{banner} ({prob * 100:.1f}%)</div>', unsafe_allow_html=True)
        st.plotly_chart(gauge_chart(prob), use_container_width=True, config={"displayModeBar": False})
        at_risk = charges * 12 * prob
        st.markdown(
            f'<div class="kpi-grid mini" style="grid-template-columns:repeat(2,1fr)">'
            + kpi_card("Annual revenue at risk", f"${at_risk:,.0f}", f"${charges * 12:,.0f} annual value x churn probability")
            + kpi_card("Risk tier", level, f"High &ge; {HIGH_THRESHOLD:.0%}, Medium &ge; {MED_THRESHOLD:.0%}", 0.6, blue=True)
            + "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### Automated CRM retention action plan")
    items = "".join(
        f'<div class="plan-item"><span class="plan-tag">{tag}</span><div><div class="plan-title">{title}</div>'
        f'<div class="plan-detail">{detail}</div></div></div>'
        for tag, title, detail in retention_plan(contract, internet, tenure, charges, calls, level)
    )
    st.markdown(items, unsafe_allow_html=True)
    st.caption(
        "Single-subscriber scores use the class priors and Laplace-smoothed likelihoods exported from the trained "
        f"{engine.name} model, so sliders respond instantly. Batch scoring in the next tab runs on the full engine."
    )

# ---------------------------- Tab 2: Batch scoring ------------------------------------
with tab2:
    st.markdown("#### Bulk CDR inference")
    up_col, demo_col = st.columns([2.2, 1], gap="large")
    with up_col:
        uploaded = st.file_uploader(
            "Drag and drop a CDR CSV",
            type=["csv"],
            help="Columns: " + ", ".join(REQUIRED_COLS) + " (subscriber_id is optional).",
        )
        st.caption("Required columns: " + ", ".join(REQUIRED_COLS))
    with demo_col:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        demo_clicked = st.button("⚡ Load 10,000 Telecom CDR Batch", use_container_width=True)

    try:
        if demo_clicked:
            demo = generate_cdr(10000, seed=2026, with_ids=True)
            with st.spinner("Running bulk inference..."):
                st.session_state["batch"] = score_batch(demo, "Demo batch (10,000 simulated CDR records)")
        elif uploaded is not None:
            key = (uploaded.name, uploaded.size)
            if st.session_state.get("upload_key") != key:
                raw = pd.read_csv(uploaded, keep_default_na=False, na_values=[""])
                with st.spinner("Running bulk inference..."):
                    st.session_state["batch"] = score_batch(raw, f"Uploaded file: {uploaded.name}")
                st.session_state["upload_key"] = key
    except Exception as exc:
        st.error(f"Could not score this file. {exc}")

    batch = st.session_state.get("batch")
    if batch is None:
        st.markdown(
            '<div class="glass">No batch scored yet. Upload a CSV or load the 10,000-record demo batch to see '
            'execution latency, risk segmentation and the high-risk retention list.</div>',
            unsafe_allow_html=True,
        )
    else:
        out = batch["df"]
        n = len(out)
        hi = out[out["churn_probability"] >= HIGH_THRESHOLD]
        st.caption(f"Source: {batch['source']} | Engine: {engine.name}")
        st.markdown(
            '<div class="kpi-grid mini">'
            + kpi_card("Records scored", f"{n:,}", "rows in this batch")
            + kpi_card("Execution latency", f"{batch['latency_ms']:,.0f} ms", "DataFrame build + inference + collect", 0.4, blue=True)
            + kpi_card("Throughput", f"{n / max(batch['latency_ms'] / 1000, 1e-6):,.0f}/s", "records per second", 0.8)
            + kpi_card("High-risk subscribers", f"{len(hi):,}", f"{len(hi) / n:.1%} of the batch", 1.2, blue=True)
            + "</div>",
            unsafe_allow_html=True,
        )

        agg = (out.groupby(["contract_type", "internet_service"])
               .agg(subscribers=("churn_probability", "size"), avg_churn_pct=("churn_probability", lambda s: s.mean() * 100))
               .reset_index())
        view = st.radio("Chart view", ["Sunburst", "Bar chart"], horizontal=True, label_visibility="collapsed")
        scale = [[0, "#34d399"], [0.5, "#fbbf24"], [1, "#ff4d6d"]]
        if view == "Sunburst":
            fig = px.sunburst(
                agg, path=["contract_type", "internet_service"], values="subscribers", color="avg_churn_pct",
                color_continuous_scale=scale, title="Churn risk by contract and internet type (colour = average churn %)",
            )
            fig.update_traces(hovertemplate="<b>%{label}</b><br>Subscribers: %{value:,}<br>Avg churn: %{color:.1f}%<extra></extra>")
            fig.update_coloraxes(colorbar_title="Avg churn %")
        else:
            fig = px.bar(
                agg, x="contract_type", y="avg_churn_pct", color="internet_service", barmode="group",
                category_orders={"contract_type": CONTRACTS, "internet_service": INTERNET},
                color_discrete_map={"Fiber Optic": "#00f5d4", "DSL": "#3b82f6", "None": "#a78bfa"},
                labels={"avg_churn_pct": "Average churn probability (%)", "contract_type": "Contract", "internet_service": "Internet"},
                title="Average churn probability by contract and internet type",
            )
        st.plotly_chart(style_fig(fig, 460), use_container_width=True, config={"displayModeBar": False})

        st.markdown("#### Scored records")
        show = out.copy()
        show["churn_probability"] = (show["churn_probability"] * 100).round(1)
        st.dataframe(
            show.sort_values("churn_probability", ascending=False).head(15).rename(columns={"churn_probability": "churn_%"}),
            use_container_width=True, hide_index=True,
        )

        export = hi.sort_values("churn_probability", ascending=False).copy()
        export["recommended_action"] = recommended_action(export)
        st.download_button(
            "⬇️ Export High-Risk Retention List (.CSV)",
            data=export.to_csv(index=False).encode("utf-8"),
            file_name="high_risk_retention_list.csv",
            mime="text/csv",
            disabled=export.empty,
        )
        st.caption(f"{len(export):,} subscribers with churn probability of {HIGH_THRESHOLD:.0%} or higher, sorted by risk.")

# ---------------------------- Tab 3: Architecture -------------------------------------
PIPELINE_CODE = '''from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.functions import vector_to_array

spark = (SparkSession.builder
         .master("local[*]")                      # on a cluster: yarn / spark://host:7077 / k8s://
         .appName("TelecomChurnNaiveBayes")
         .config("spark.driver.memory", "1g")
         .getOrCreate())

# 1. Ingest CDR features (one-hot contract, internet, binned tenure / charges / calls)
cdr = spark.read.parquet("hdfs:///telecom/cdr_features/")     # or spark.createDataFrame(...)

# 2. Assemble the feature vector
assembler = VectorAssembler(inputCols=[f"f{i}" for i in range(20)], outputCol="features")
train, test = assembler.transform(cdr).randomSplit([0.8, 0.2], seed=42)

# 3. Fit: one pass of Map (count) + Reduce (sum) with Laplace smoothing
nb = NaiveBayes(smoothing=1.0, modelType="multinomial",
                featuresCol="features", labelCol="label")
model = nb.fit(train)

# 4. Embarrassingly parallel scoring: every partition applies the broadcast model
scored = model.transform(test).select(
    vector_to_array("probability").getItem(1).alias("churn_probability"))
scored.show(5)'''

with tab3:
    st.markdown("#### How Naive Bayes scales across Spark worker nodes")
    st.markdown(
        '<div class="glass">Naive Bayes only needs <b>counts</b>: how often each feature value appears with each class. '
        'Counting is associative and commutative, so every worker can count its own partition independently and the '
        'partial counts are simply added together. The data is read once, there is no iterative optimisation loop, '
        'and runtime grows linearly with the number of records and falls as workers are added.</div>',
        unsafe_allow_html=True,
    )

    flow = '<div class="flow">'
    for w, rows in ((1, "1.7k"), (2, "1.7k"), (3, "1.6k")):
        flow += (f'<div class="node">Worker {w}<span>partition {w} of CDR records (~{rows})</span></div><div class="arrow">➜</div>'
                 f'<div class="node map">Map: emit ((feature, value, class), 1)<span>map-side combine into local counts</span></div>')
        if w == 1:
            flow += '<div class="arrow">➜</div><div class="node reduce"><b>Reduce</b><span>reduceByKey sums counts per (feature, value, class) in a single shuffle</span><br><span>Then: priors and smoothed log-likelihoods, broadcast to all executors</span></div>'
        else:
            flow += '<div class="arrow">➜</div>'
    flow += "</div>"
    # the grid places the reduce node in column 5 spanning all rows; add spacing placeholders for rows 2 and 3
    st.markdown(f'<div class="glass">{flow}</div>', unsafe_allow_html=True)

    steps = [
        ("1", "driver", "Driver", "Ingest and partition", "CDR features are read from HDFS, S3 or Parquet and split into partitions, one task per partition on each executor core."),
        ("2", "map", "Map phase", "Emit feature counts", "Each task scans its partition once and emits ((feature, value, class), 1) for every active feature, plus a (class, 1) record for the priors."),
        ("3", "map", "Map phase", "Combine locally", "A map-side combiner adds counts inside each partition first, so only a few hundred small key-count pairs leave each worker, not raw records."),
        ("4", "reduce", "Reduce phase", "Sum across workers", "One shuffle groups identical keys and adds the partial counts. This is the only network exchange in training, and it is not repeated."),
        ("5", "driver", "Driver", "Build the model", "Priors log P(C) and Laplace-smoothed likelihoods log((N_ci + 1) / (N_c + d)) are computed from the tiny reduced table."),
        ("6", "map", "Inference", "Broadcast and score", "The model is broadcast once. Each partition scores its rows independently with an argmax over log-posteriors: no shuffle at all."),
    ]
    cards = "".join(
        f'<div class="step-card"><div class="step-num">{n}</div><span class="phase phase-{ph}">{tag}</span>'
        f"<h4>{title}</h4><p>{body}</p></div>"
        for n, ph, tag, title, body in steps
    )
    st.markdown(f'<div class="step-grid">{cards}</div>', unsafe_allow_html=True)

    m1, m2 = st.columns(2, gap="large")
    with m1:
        st.markdown("##### Model")
        st.latex(r"\hat{y}=\arg\max_{c}\;\log P(c)+\sum_{i=1}^{d}x_i\,\log\theta_{ci}")
        st.latex(r"\theta_{ci}=\frac{N_{ci}+\alpha}{N_c+\alpha d},\quad \alpha=1\ \text{(Laplace smoothing)}")
        st.caption("N_ci is the count of feature i in class c. Smoothing keeps unseen feature/class pairs from zeroing the posterior.")
    with m2:
        workers = np.array([1, 2, 4, 8, 16, 32])
        one_pass = 100 / workers + 2
        iterative = 30 * (100 / workers + 2)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=workers, y=one_pass, mode="lines+markers", name="Naive Bayes (1 pass)",
                                 line=dict(color="#00f5d4", width=3)))
        fig.add_trace(go.Scatter(x=workers, y=iterative, mode="lines+markers", name="Iterative learner (30 passes)",
                                 line=dict(color="#3b82f6", width=3, dash="dot")))
        fig.update_layout(title="Illustrative cost model: relative training time vs. workers",
                          xaxis_title="Worker nodes", yaxis_title="Relative time (log scale)", yaxis_type="log",
                          xaxis=dict(type="log", tickvals=workers, ticktext=[str(w) for w in workers]),
                          legend=dict(orientation="h", y=-0.28))
        st.plotly_chart(style_fig(fig, 330), use_container_width=True, config={"displayModeBar": False})
        st.caption("Schematic model, not a benchmark: cost = passes over the data x (work / workers + shuffle overhead).")

    st.markdown("#### PySpark pipeline")
    st.code(PIPELINE_CODE, language="python")

# --------------------------------------------------------------------------------------
# KPI strip (rendered last so it includes batches scored during this run)
# --------------------------------------------------------------------------------------
extra = len(st.session_state["batch"]["df"]) if st.session_state.get("batch") is not None else 0
total_records = 5000 + extra
kpi_slot.markdown(
    '<div class="kpi-grid">'
    + kpi_card("Total CDR Records Analyzed", f"{total_records:,}+", f"5,000 training corpus + {extra:,} batch records", 0.0)
    + kpi_card("Model Accuracy", f"{METRICS['accuracy'] * 100:.1f}%", f"hold-out set of {METRICS['n_test']:,} records", 0.5, blue=True)
    + kpi_card("Precision (churn)", f"{METRICS['precision'] * 100:.1f}%", "flagged subscribers who really churn", 1.0)
    + kpi_card("Recall (churn)", f"{METRICS['recall'] * 100:.1f}%", "churners the model catches", 1.5, blue=True)
    + "</div>",
    unsafe_allow_html=True,
)
