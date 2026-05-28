import sys
import os
import subprocess

required_packages = ['streamlit', 'numpy', 'pandas', 'joblib', 'scikit-learn']
for package in required_packages:
    try:
        __import__(package if package != 'scikit-learn' else 'sklearn')
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

if os.environ.get('STREAMLIT_RUNNING') != '1':
    os.environ['STREAMLIT_RUNNING'] = '1'
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
    sys.exit(stcli.main())

import streamlit as st
import streamlit.components.v1 as components
import json
import joblib
import numpy as np

st.set_page_config(
    page_title="폐암 환자 군집 분석",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, .stApp, .stApp * {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.stApp, [data-testid="stAppViewContainer"] { background: #0a0b0d !important; }
[data-testid="stHeader"], [data-testid="stMain"], [data-testid="stMainBlockContainer"],
.main, .block-container { background: transparent !important; }
.main .block-container {
    padding-top: 1.2rem; padding-bottom: 1rem;
    max-width: 1280px;
}
[data-testid="stAlert"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
}
[data-testid="stAlert"] * { color: rgba(255,255,255,0.85) !important; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }

.hero-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(24px) saturate(140%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 1.2rem 1.6rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}
.hero-card h1 {
    color: #fff !important;
    font-size: 1.45rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.03em;
}
.hero-card p {
    color: rgba(255,255,255,0.5) !important;
    font-size: 0.78rem;
    margin: 0.25rem 0 0 0 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-card">
    <h1>폐암 환자 군집 분석</h1>
    <p>Lung Cancer Patient Clustering · K-Means Unsupervised Learning</p>
</div>
""", unsafe_allow_html=True)

# ===== 모델 로드 & 중심점/스케일 파라미터 추출 =====
@st.cache_resource
def load_params():
    try:
        obj1 = joblib.load("lungkmeans.pkl")
    except FileNotFoundError:
        return None, "lungkmeans.pkl 파일을 찾을 수 없습니다."
    try:
        obj2 = joblib.load("lungscaler.pkl")
    except FileNotFoundError:
        obj2 = None

    if hasattr(obj1, 'cluster_centers_'):
        kmeans, scaler = obj1, obj2
    elif obj2 is not None and hasattr(obj2, 'cluster_centers_'):
        kmeans, scaler = obj2, obj1
    else:
        return None, "KMeans 모델을 인식할 수 없습니다."

    centers = kmeans.cluster_centers_.tolist()  # 스케일링된 공간의 중심점

    if scaler is not None and hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
        mean = scaler.mean_.tolist()
        scale = scaler.scale_.tolist()
    else:
        # 스케일러가 없으면 identity
        mean = [0.0] * len(centers[0])
        scale = [1.0] * len(centers[0])

    return {
        "centers": centers,
        "mean": mean,
        "scale": scale,
        "n_clusters": int(getattr(kmeans, "n_clusters", len(centers))),
    }, None

params, load_err = load_params()

if load_err or params is None:
    st.error((load_err or "모델 파일을 불러올 수 없습니다.") +
             " 노트북에서 KMeans 모델을 lungkmeans.pkl, StandardScaler를 lungscaler.pkl로 같은 폴더에 저장해 주세요.")
    st.stop()

# K=4 기준 군집 의미 (노트북 결과 분석)
CLUSTER_INFO_LIST = [
    {"name": "중간군",        "tone": "warn",    "desc": "위험 인자가 일부 누적된 중간 수준의 환자군입니다."},
    {"name": "건강군",        "tone": "safe",    "desc": "위험 인자가 낮아 상대적으로 건강한 환자군입니다."},
    {"name": "고위험군",      "tone": "danger2", "desc": "위험 인자가 가장 높은 폐암 고위험 환자군입니다."},
    {"name": "폐암 위험군",   "tone": "danger",  "desc": "흡연·음주가 누적된 위험 환자군입니다."},
]

# JS로 안전하게 데이터 전달
data_payload = {
    "params": params,
    "clusterInfo": CLUSTER_INFO_LIST,
    "vars": [
        {"key": "smoke", "label": "흡연",   "min": 0,  "max": 40, "default": 10, "decimals": 0},
        {"key": "alc",   "label": "음주량", "min": 0,  "max": 10, "default": 3,  "decimals": 0},
        {"key": "age",   "label": "나이",   "min": 10, "max": 90, "default": 45, "decimals": 0},
    ]
}
payload_json = json.dumps(data_payload)

html = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { box-sizing: border-box; font-family: 'Inter', sans-serif; }
body { margin: 0; padding: 0; background: transparent; color: #fff; }

.workspace {
    display: grid;
    grid-template-columns: 1.05fr 1fr;
    gap: 14px;
    width: 100%;
    height: 540px;
}

.panel {
    position: relative;
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(24px) saturate(140%);
    -webkit-backdrop-filter: blur(24px) saturate(140%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 22px 22px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}
.panel::before {
    content: '';
    position: absolute;
    top: 0; left: 14%; right: 14%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.22), transparent);
    pointer-events: none;
}
.panel-label {
    color: rgba(255,255,255,0.42);
    font-size: 0.64rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.2em;
}
.panel-hint {
    color: rgba(255,255,255,0.4);
    font-size: 0.74rem;
    margin: 6px 0 18px 0;
}

.faders {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    flex: 1;
    min-height: 0;
}
.fader-col {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    user-select: none;
    min-height: 0;
}
.fader-label {
    color: rgba(255,255,255,0.65);
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-weight: 600;
}
.fader-track-wrap {
    position: relative;
    width: 56px;
    flex: 1;
    min-height: 0;
    display: flex;
    justify-content: center;
    align-items: center;
    cursor: pointer;
    touch-action: none;
}
.fader-track {
    position: relative;
    width: 6px;
    height: 100%;
    background: rgba(255,255,255,0.06);
    border-radius: 6px;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.5);
    overflow: hidden;
}
.fader-fill {
    position: absolute;
    left: 0; right: 0; bottom: 0;
    background: linear-gradient(0deg, rgba(255,255,255,0.85), rgba(255,255,255,0.35));
    border-radius: 6px;
    height: 50%;
}
.fader-handle {
    position: absolute;
    width: 42px;
    height: 20px;
    left: 50%;
    transform: translate(-50%, -50%);
    top: 50%;
    background: linear-gradient(180deg, #f0f0f0, #c0c0c0 45%, #808080 46%, #a8a8a8 100%);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 4px;
    box-shadow: 0 3px 8px rgba(0,0,0,0.55), inset 0 1px 0 rgba(255,255,255,0.6);
    cursor: ns-resize;
    pointer-events: none;
}
.fader-handle::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 4px; right: 4px;
    height: 1px;
    background: rgba(0,0,0,0.55);
    transform: translateY(-50%);
}
.fader-col.dragging .fader-fill {
    background: linear-gradient(0deg, rgba(140,220,180,0.9), rgba(140,220,180,0.4));
}
.fader-value {
    color: #fff;
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: -0.02em;
}
.fader-range {
    color: rgba(255,255,255,0.28);
    font-size: 0.58rem;
    letter-spacing: 0.06em;
}

.result-headline { text-align: center; flex: 1; display: flex; flex-direction: column; justify-content: center; padding: 0 12px; }
.result-cluster {
    color: rgba(255,255,255,0.45);
    font-size: 0.74rem;
    margin: 0 0 10px 0;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    font-weight: 700;
}
.result-name {
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.045em;
    line-height: 1.05;
    margin: 0;
    color: #fff;
    transition: color 0.4s ease;
}
.result-name.safe { color: #6bcf9f; }
.result-name.warn { color: #ffd166; }
.result-name.danger { color: #ff8a80; }
.result-name.danger2 { color: #ff6b6b; }
.result-desc {
    color: rgba(255,255,255,0.6);
    font-size: 0.85rem;
    line-height: 1.55;
    margin: 14px auto 0 auto;
    max-width: 340px;
}

.cluster-dots {
    display: flex;
    justify-content: center;
    gap: 14px;
    margin-top: 20px;
}
.cluster-dot {
    width: 11px; height: 11px;
    border-radius: 50%;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    transition: all 0.3s ease;
}
.cluster-dot.active { transform: scale(1.3); }
.cluster-dot.active.safe { background: #6bcf9f; border-color: #6bcf9f; box-shadow: 0 0 14px rgba(107,207,159,0.7); }
.cluster-dot.active.warn { background: #ffd166; border-color: #ffd166; box-shadow: 0 0 14px rgba(255,209,102,0.7); }
.cluster-dot.active.danger { background: #ff8a80; border-color: #ff8a80; box-shadow: 0 0 14px rgba(255,138,128,0.7); }
.cluster-dot.active.danger2 { background: #ff6b6b; border-color: #ff6b6b; box-shadow: 0 0 14px rgba(255,107,107,0.7); }

.result-stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    padding-top: 18px;
    margin-top: 16px;
    border-top: 1px solid rgba(255,255,255,0.06);
}
.stat-block { text-align: center; }
.stat-block .v { color: #fff; font-size: 1.05rem; font-weight: 700; letter-spacing: -0.02em; }
.stat-block .l {
    color: rgba(255,255,255,0.38);
    font-size: 0.58rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-top: 4px;
    font-weight: 600;
}
</style>
</head>
<body>
<div class="workspace">
    <div class="panel">
        <div class="panel-label">Input</div>
        <div class="panel-hint">핸들을 위/아래로 드래그하면 결과가 실시간으로 갱신됩니다</div>
        <div class="faders" id="faders"></div>
    </div>
    <div class="panel">
        <div class="panel-label">Cluster Result</div>
        <div class="result-headline">
            <div class="result-cluster" id="r-cluster">Cluster —</div>
            <h2 class="result-name" id="r-name">대기 중</h2>
            <p class="result-desc" id="r-desc">페이더를 조정하세요</p>
            <div class="cluster-dots" id="dots"></div>
        </div>
        <div class="result-stats">
            <div class="stat-block"><div class="v" id="s-smoke">10</div><div class="l">흡연</div></div>
            <div class="stat-block"><div class="v" id="s-alc">3</div><div class="l">음주량</div></div>
            <div class="stat-block"><div class="v" id="s-age">45</div><div class="l">나이</div></div>
        </div>
    </div>
</div>

<script>
var DATA = __PAYLOAD__;
var VARS = DATA.vars;
var PARAMS = DATA.params;
var CLUSTER_INFO = DATA.clusterInfo;
var values = {};
VARS.forEach(function(v) { values[v.key] = v.default; });

// 페이더 DOM 생성
var fadersContainer = document.getElementById('faders');
VARS.forEach(function(v) {
    var col = document.createElement('div');
    col.className = 'fader-col';
    col.dataset.key = v.key;
    col.dataset.min = v.min;
    col.dataset.max = v.max;
    col.dataset.decimals = v.decimals;
    col.innerHTML =
        '<div class="fader-label">' + v.label + '</div>' +
        '<div class="fader-track-wrap">' +
            '<div class="fader-track"><div class="fader-fill"></div></div>' +
            '<div class="fader-handle"></div>' +
        '</div>' +
        '<div class="fader-value">' + v.default + '</div>' +
        '<div class="fader-range">' + v.min + ' \u2014 ' + v.max + '</div>';
    fadersContainer.appendChild(col);
});

// 군집 점 생성
var dotsContainer = document.getElementById('dots');
for (var i = 0; i < PARAMS.n_clusters; i++) {
    var dot = document.createElement('div');
    dot.className = 'cluster-dot';
    dot.dataset.idx = i;
    dotsContainer.appendChild(dot);
}

function fmt(val, decimals) {
    if (decimals === 0) return String(Math.round(val));
    return val.toFixed(decimals);
}

// 클라이언트 사이드 K-Means 예측
function predict() {
    // 입력 순서: [흡연, 음주량, 나이] - 노트북과 동일
    var x = [values.smoke, values.alc, values.age];
    // 스케일링
    var scaled = [];
    for (var i = 0; i < x.length; i++) {
        scaled.push((x[i] - PARAMS.mean[i]) / PARAMS.scale[i]);
    }
    // 각 중심점과의 거리 계산
    var bestIdx = 0;
    var bestDist = Infinity;
    for (var c = 0; c < PARAMS.centers.length; c++) {
        var center = PARAMS.centers[c];
        var d = 0;
        for (var j = 0; j < center.length; j++) {
            var diff = scaled[j] - center[j];
            d += diff * diff;
        }
        if (d < bestDist) {
            bestDist = d;
            bestIdx = c;
        }
    }
    return bestIdx;
}

function updateResult() {
    var clusterId = predict();
    var info = CLUSTER_INFO[clusterId] || { name: "Cluster " + clusterId, tone: "warn", desc: "분류된 군집입니다." };

    document.getElementById('r-cluster').textContent = 'Cluster · ' + clusterId;
    var nameEl = document.getElementById('r-name');
    nameEl.textContent = info.name;
    nameEl.className = 'result-name ' + info.tone;
    document.getElementById('r-desc').textContent = info.desc;

    var dots = document.querySelectorAll('.cluster-dot');
    dots.forEach(function(d, i) {
        d.className = 'cluster-dot';
        if (i === clusterId) {
            d.classList.add('active', info.tone);
        }
    });

    document.getElementById('s-smoke').textContent = Math.round(values.smoke);
    document.getElementById('s-alc').textContent = Math.round(values.alc);
    document.getElementById('s-age').textContent = Math.round(values.age);
}

// 페이더 인터랙션
document.querySelectorAll('.fader-col').forEach(function(col) {
    var min = parseFloat(col.dataset.min);
    var max = parseFloat(col.dataset.max);
    var decimals = parseInt(col.dataset.decimals);
    var key = col.dataset.key;
    var range = max - min;
    var trackWrap = col.querySelector('.fader-track-wrap');
    var track = col.querySelector('.fader-track');
    var fill = col.querySelector('.fader-fill');
    var handle = col.querySelector('.fader-handle');
    var valueEl = col.querySelector('.fader-value');
    var currentValue = values[key];

    function render() {
        var pct = (currentValue - min) / range;
        var trackRect = track.getBoundingClientRect();
        var h = trackRect.height;
        fill.style.height = (pct * 100) + '%';
        handle.style.top = ((1 - pct) * h) + 'px';
        valueEl.textContent = fmt(currentValue, decimals);
    }

    function setFromY(clientY) {
        var trackRect = track.getBoundingClientRect();
        var ratio = 1 - Math.max(0, Math.min(1, (clientY - trackRect.top) / trackRect.height));
        currentValue = Math.max(min, Math.min(max, min + ratio * range));
        values[key] = currentValue;
        render();
        updateResult();
    }

    setTimeout(render, 50);
    window.addEventListener('resize', render);

    var dragging = false;

    trackWrap.addEventListener('pointerdown', function(e) {
        e.preventDefault();
        dragging = true;
        col.classList.add('dragging');
        setFromY(e.clientY);
        try { trackWrap.setPointerCapture(e.pointerId); } catch(err) {}
    });

    trackWrap.addEventListener('pointermove', function(e) {
        if (!dragging) return;
        setFromY(e.clientY);
    });

    function stopDrag(e) {
        if (!dragging) return;
        dragging = false;
        col.classList.remove('dragging');
        try { trackWrap.releasePointerCapture(e.pointerId); } catch(err) {}
    }
    trackWrap.addEventListener('pointerup', stopDrag);
    trackWrap.addEventListener('pointercancel', stopDrag);

    trackWrap.addEventListener('wheel', function(e) {
        e.preventDefault();
        var step = range / 100;
        var dir = e.deltaY < 0 ? 1 : -1;
        currentValue = Math.max(min, Math.min(max, currentValue + dir * step));
        values[key] = currentValue;
        render();
        updateResult();
    }, { passive: false });
});

// 초기 예측
setTimeout(updateResult, 100);
</script>
</body>
</html>
"""

html = html.replace("__PAYLOAD__", payload_json)
components.html(html, height=580, scrolling=False)
