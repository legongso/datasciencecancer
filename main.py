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

# ===== Streamlit Main UI Style =====
st.markdown("""
<style>
@font-face {
    font-family: 'GmarketSans';
    src: url('https://fastly.jsdelivr.net/gh/projectnoonnu/noonfonts_2001@1.1/GmarketSansLight.woff') format('woff');
    font-weight: 300;
}
@font-face {
    font-family: 'GmarketSans';
    src: url('https://fastly.jsdelivr.net/gh/projectnoonnu/noonfonts_2001@1.1/GmarketSansMedium.woff') format('woff');
    font-weight: 500;
}
@font-face {
    font-family: 'GmarketSans';
    src: url('https://fastly.jsdelivr.net/gh/projectnoonnu/noonfonts_2001@1.1/GmarketSansBold.woff') format('woff');
    font-weight: 700;
}

html, body, .stApp, .stApp * {
    font-family: 'GmarketSans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.stApp, [data-testid="stAppViewContainer"] { background: #0a0b0d !important; }
[data-testid="stHeader"], [data-testid="stMain"], [data-testid="stMainBlockContainer"],
.main, .block-container { background: transparent !important; }
.main .block-container {
    padding-top: 1.2rem; padding-bottom: 1rem;
    max-width: 1440px;
}
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
    <h1>폐암 환자 군집 분석 시스템</h1>
    <p>Dynamic Dimension Multi-Viewer · K-Means Decision Boundary Analysis</p>
</div>
""", unsafe_allow_html=True)

# ===== 모델 및 차원 파라미터 로드 =====
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

    centers = kmeans.cluster_centers_.tolist()

    if scaler is not None and hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
        mean = scaler.mean_.tolist()
        scale = scaler.scale_.tolist()
    else:
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
    st.error((load_err or "모델 파일을 불러올 수 없습니다.") + " 파일을 확인해주세요.")
    st.stop()

CLUSTER_INFO_LIST = [
    {"name": "중간군",        "tone": "danger",  "desc": "위험 인자가 일부 누적된 중간 수준의 환자군입니다."},
    {"name": "건강군",        "tone": "safe",    "desc": "위험 인자가 낮아 상대적으로 건강한 환자군입니다."},
    {"name": "고위험군",      "tone": "danger2", "desc": "위험 인자가 가장 높은 폐암 고위험 환자군입니다."},
    {"name": "폐암 위험군",   "tone": "warn",    "desc": "흡연·음주가 누적된 위험 환자군입니다."},
]

# 가상 입체 데이터셋 (smoke, alc, age, cluster_id)
dataset_points = [
    [2, 1, 25, 1], [0, 0, 28, 1], [1, 2, 30, 1], [0, 1, 34, 1], [4, 2, 33, 1], [2, 1, 45, 1], [4, 2, 40, 1],
    [10, 4, 18, 0], [12, 3, 22, 0], [20, 5, 25, 0], [10, 4, 28, 0], [12, 5, 34, 0], [15, 4, 37, 0], [12, 5, 42, 0], [25, 3, 51, 0],
    [34, 8, 26, 3], [25, 9, 34, 3], [20, 7, 40, 3], [30, 8, 43, 3], [30, 9, 44, 3], [35, 6, 48, 3], [28, 8, 55, 3],
    [15, 4, 58, 2], [5, 3, 62, 2], [20, 5, 62, 2], [25, 6, 62, 2], [20, 4, 69, 2], [10, 2, 73, 2], [20, 5, 77, 2], [15, 3, 75, 2]
]

data_payload = {
    "params": params,
    "clusterInfo": CLUSTER_INFO_LIST,
    "points": dataset_points,
    "vars": [
        {"key": "smoke", "label": "흡연량",   "min": 0,  "max": 40, "default": 10, "decimals": 0, "idx": 0},
        {"key": "alc",   "label": "음주량",   "min": 0,  "max": 10, "default": 3,  "decimals": 0, "idx": 1},
        {"key": "age",   "label": "나이",     "min": 15, "max": 80, "default": 45, "decimals": 0, "idx": 2},
    ]
}
payload_json = json.dumps(data_payload)

# ===== 인터랙티브 UI 메인 임베딩 스크립트 =====
html = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<style>
@font-face { font-family: 'GmarketSans'; src: url('https://fastly.jsdelivr.net/gh/projectnoonnu/noonfonts_2001@1.1/GmarketSansLight.woff') format('woff'); }
@font-face { font-family: 'GmarketSans'; src: url('https://fastly.jsdelivr.net/gh/projectnoonnu/noonfonts_2001@1.1/GmarketSansMedium.woff') format('woff'); }
@font-face { font-family: 'GmarketSans'; src: url('https://fastly.jsdelivr.net/gh/projectnoonnu/noonfonts_2001@1.1/GmarketSansBold.woff') format('woff'); }

* { box-sizing: border-box; font-family: 'GmarketSans', sans-serif; }
body { margin: 0; padding: 0; background: transparent; color: #fff; overflow: hidden; }

.workspace { display: grid; grid-template-columns: 280px 1fr 300px; gap: 14px; width: 100%; height: 570px; }

.panel {
    position: relative; background: rgba(255,255,255,0.04); backdrop-filter: blur(24px) saturate(140%);
    border: 1px solid rgba(255,255,255,0.08); border-radius: 20px; padding: 20px; display: flex; flex-direction: column;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}
.panel-label { color: rgba(255,255,255,0.42); font-size: 0.64rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.2em; }

.top-toolbar { display: flex; align-items: center; justify-content: space-between; margin: 6px 0 12px 0; gap: 10px; }
.btn-group { display: flex; background: rgba(255,255,255,0.06); border-radius: 8px; padding: 2px; border: 1px solid rgba(255,255,255,0.05); }
.toggle-btn { background: transparent; border: none; color: rgba(255,255,255,0.4); padding: 6px 14px; font-size: 0.74rem; font-weight: 600; cursor: pointer; border-radius: 6px; transition: all 0.25s ease; }
.toggle-btn.active { background: rgba(255,255,255,0.12); color: #fff; box-shadow: 0 2px 8px rgba(0,0,0,0.2); }
.selectors { display: flex; gap: 8px; align-items: center; }
.selectors select { background: #15161a; border: 1px solid rgba(255,255,255,0.12); color: rgba(255,255,255,0.85); border-radius: 6px; padding: 4px 8px; font-size: 0.72rem; outline: none; cursor: pointer; }

.faders { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; flex: 1; margin-top: 15px; }
.fader-col { display: flex; flex-direction: column; align-items: center; gap: 12px; user-select: none; }
.fader-label { color: rgba(255,255,255,0.65); font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.12em; font-weight: 600; }
.fader-track-wrap { position: relative; width: 46px; flex: 1; display: flex; justify-content: center; align-items: center; cursor: pointer; touch-action: none; }
.fader-track { position: relative; width: 6px; height: 100%; background: rgba(255,255,255,0.06); border-radius: 6px; }
.fader-fill { position: absolute; left: 0; right: 0; bottom: 0; background: linear-gradient(0deg, rgba(255,255,255,0.85), rgba(255,255,255,0.35)); border-radius: 6px; }
.fader-handle { position: absolute; width: 36px; height: 18px; left: 50%; transform: translate(-50%, -50%); background: linear-gradient(180deg, #f0f0f0, #c0c0c0 45%, #808080 46%, #a8a8a8 100%); border: 1px solid rgba(255,255,255,0.3); border-radius: 4px; box-shadow: 0 3px 8px rgba(0,0,0,0.55); pointer-events: none; }
.fader-value { color: #fff; font-size: 1.25rem; font-weight: 800; }
.fader-range { color: rgba(255,255,255,0.28); font-size: 0.58rem; }

.chart-container { flex: 1; position: relative; width: 100%; min-height: 0; border-radius: 12px; overflow: hidden; }
#canvas2D, #threeCanvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: none; }
#canvas2D { display: block; }

.result-headline { text-align: center; flex: 1; display: flex; flex-direction: column; justify-content: center; }
.result-cluster { color: rgba(255,255,255,0.45); font-size: 0.74rem; margin-bottom: 8px; letter-spacing: 0.15em; font-weight: 700; }
.result-name { font-size: 2.3rem; font-weight: 800; margin: 0; }
.result-name.safe { color: #6bcf9f; } .result-name.warn { color: #ffd166; } .result-name.danger { color: #b388ff; } .result-name.danger2 { color: #90caf9; } 
.result-desc { color: rgba(255,255,255,0.55); font-size: 0.82rem; line-height: 1.5; margin: 12px auto 0 auto; max-width: 250px; }

.cluster-dots { display: flex; justify-content: center; gap: 12px; margin-top: 15px; }
.cluster-dot { width: 10px; height: 10px; border-radius: 50%; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.15); transition: all 0.3s ease; }
.cluster-dot.active.safe { background: #6bcf9f; box-shadow: 0 0 12px #6bcf9f; }
.cluster-dot.active.warn { background: #ffd166; box-shadow: 0 0 12px #ffd166; }
.cluster-dot.active.danger { background: #b388ff; box-shadow: 0 0 12px #b388ff; }
.cluster-dot.active.danger2 { background: #90caf9; box-shadow: 0 0 12px #90caf9; }

.result-stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; padding-top: 15px; margin-top: 15px; border-top: 1px solid rgba(255,255,255,0.06); }
.stat-block { text-align: center; }
.stat-block .v { color: #fff; font-size: 1.05rem; font-weight: 700; }
.stat-block .l { color: rgba(255,255,255,0.35); font-size: 0.58rem; margin-top: 4px; font-weight: 600; }
</style>
</head>
<body>
<div class="workspace">
    <div class="panel">
        <div class="panel-label">Input Parameters</div>
        <div class="faders" id="faders"></div>
    </div>
    
    <div class="panel" style="padding-top: 16px;">
        <div class="top-toolbar">
            <div class="btn-group">
                <button class="toggle-btn active" id="btn2D" onclick="switchMode('2D')">2D Map</button>
                <button class="toggle-btn" id="btn3D" onclick="switchMode('3D')">3D Space</button>
            </div>
            <div class="selectors" id="dimSelectors">
                <label style="font-size:0.7rem; color:rgba(255,255,255,0.4);">X축</label>
                <select id="selX" onchange="onAxisChange()"></select>
                <label style="font-size:0.7rem; color:rgba(255,255,255,0.4); margin-left:4px;">Y축</label>
                <select id="selY" onchange="onAxisChange()"></select>
            </div>
        </div>
        
        <div class="chart-container" id="viewContainer">
            <canvas id="canvas2D"></canvas>
            <canvas id="threeCanvas"></canvas>
        </div>
    </div>
    
    <div class="panel">
        <div class="panel-label">Cluster Analysis</div>
        <div class="result-headline">
            <div class="result-cluster" id="r-cluster">Cluster —</div>
            <h2 class="result-name" id="r-name">-</h2>
            <p class="result-desc" id="r-desc">파라미터를 연동 중입니다.</p>
            <div class="cluster-dots" id="dots"></div>
        </div>
        <div class="result-stats">
            <div class="stat-block"><div class="v" id="s-smoke">0</div><div class="l">흡연량</div></div>
            <div class="stat-block"><div class="v" id="s-alc">0</div><div class="l">음주량</div></div>
            <div class="stat-block"><div class="v" id="s-age">0</div><div class="l">나이</div></div>
        </div>
    </div>
</div>

<script>
var DATA = __PAYLOAD__;
var VARS = DATA.vars;
var PARAMS = DATA.params;
var CLUSTER_INFO = DATA.clusterInfo;
var POINTS = DATA.points;
var values = {};
VARS.forEach(function(v) { values[v.key] = v.default; });

var currentMode = '2D';
var axisXKey = 'age';
var axisYKey = 'smoke';

var TONE_COLORS = { "safe": "#6bcf9f", "warn": "#ffd166", "danger": "#b388ff", "danger2": "#90caf9" };
var TONE_HEX_THREE = { "safe": 0x6bcf9f, "warn": 0xffd166, "danger": 0xb388ff, "danger2": 0x90caf9 };

// 누락되었던 숫자 포맷팅 함수 복구!
function fmt(val, decimals) {
    if (decimals === 0) return String(Math.round(val));
    return val.toFixed(decimals);
}

// 드롭다운 구조 생성
var selX = document.getElementById('selX');
var selY = document.getElementById('selY');
VARS.forEach(function(v) {
    var opX = document.createElement('option'); opX.value = v.key; opX.textContent = v.label;
    var opY = document.createElement('option'); opY.value = v.key; opY.textContent = v.label;
    if(v.key === axisXKey) opX.selected = true;
    if(v.key === axisYKey) opY.selected = true;
    selX.appendChild(opX); selY.appendChild(opY);
});

function onAxisChange() { axisXKey = selX.value; axisYKey = selY.value; draw2D(); }

function switchMode(mode) {
    currentMode = mode;
    document.getElementById('btn2D').classList.toggle('active', mode === '2D');
    document.getElementById('btn3D').classList.toggle('active', mode === '3D');
    document.getElementById('dimSelectors').style.visibility = (mode === '2D') ? 'visible' : 'hidden';
    document.getElementById('canvas2D').style.display = (mode === '2D') ? 'block' : 'none';
    document.getElementById('threeCanvas').style.display = (mode === '3D') ? 'block' : 'none';
    
    if(mode === '2D') { draw2D(); } else { resize3D(); updateResult(); }
}

var fadersContainer = document.getElementById('faders');
VARS.forEach(function(v) {
    var col = document.createElement('div');
    col.className = 'fader-col';
    col.dataset.key = v.key; col.dataset.min = v.min; col.dataset.max = v.max; col.dataset.decimals = v.decimals;
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

var dotsContainer = document.getElementById('dots');
for (var i = 0; i < PARAMS.n_clusters; i++) {
    var dot = document.createElement('div');
    dot.className = 'cluster-dot';
    dotsContainer.appendChild(dot);
}

function predictCustom(smokeVal, alcVal, ageVal) {
    var x = [smokeVal, alcVal, ageVal];
    var scaled = [];
    for (var i = 0; i < x.length; i++) { scaled.push((x[i] - PARAMS.mean[i]) / PARAMS.scale[i]); }
    var bestIdx = 0, bestDist = Infinity;
    for (var c = 0; c < PARAMS.centers.length; c++) {
        var center = PARAMS.centers[c], d = 0;
        for (var j = 0; j < center.length; j++) { var diff = scaled[j] - center[j]; d += diff * diff; }
        if (d < bestDist) { bestDist = d; bestIdx = c; }
    }
    return bestIdx;
}
function predictCurrent() { return predictCustom(values.smoke, values.alc, values.age); }

var canvas2D = document.getElementById('canvas2D');
var ctx2D = canvas2D.getContext('2d');

function draw2D() {
    if(currentMode !== '2D') return;
    var w = canvas2D.width = canvas2D.parentElement.clientWidth;
    var height = canvas2D.height = canvas2D.parentElement.clientHeight;
    
    var pad = { top: 25, right: 25, bottom: 40, left: 45 };
    var chartW = w - pad.left - pad.right;
    var chartH = height - pad.top - pad.bottom;
    
    var vX = VARS.find(function(v){return v.key === axisXKey;});
    var vY = VARS.find(function(v){return v.key === axisYKey;});
    
    function getXPos(val) { return pad.left + ((val - vX.min) / (vX.max - vX.min)) * chartW; }
    function getYPos(val) { return pad.top + chartH - ((val - vY.min) / (vY.max - vY.min)) * chartH; }
    function getXVal(px) { return vX.min + ((px - pad.left) / chartW) * (vX.max - vX.min); }
    function getYVal(py) { return vY.min + (1 - (py - pad.top) / chartH) * (vY.max - vY.min); }

    var step = 4;
    for (var y = pad.top; y < pad.top + chartH; y += step) {
        for (var x = pad.left; x < pad.left + chartW; x += step) {
            var simX = getXVal(x); var simY = getYVal(y);
            var sVal = (axisXKey==='smoke')? simX : ((axisYKey==='smoke')? simY : values.smoke);
            var alVal = (axisXKey==='alc')? simX : ((axisYKey==='alc')? simY : values.alc);
            var agVal = (axisXKey==='age')? simX : ((axisYKey==='age')? simY : values.age);
            var cid = predictCustom(sVal, alVal, agVal);
            
            ctx2D.fillStyle = TONE_COLORS[CLUSTER_INFO[cid].tone];
            ctx2D.globalAlpha = 0.12;
            ctx2D.fillRect(x, y, step, step);
        }
    }
    ctx2D.globalAlpha = 1.0;

    ctx2D.strokeStyle = 'rgba(255,255,255,0.06)'; ctx2D.lineWidth = 1; ctx2D.fillStyle = 'rgba(255,255,255,0.3)'; ctx2D.font = '10px GmarketSans';
    ctx2D.textAlign = 'center'; ctx2D.fillText(vX.min, pad.left, pad.top + chartH + 15); ctx2D.fillText(vX.max, pad.left + chartW, pad.top + chartH + 15);
    ctx2D.textAlign = 'right'; ctx2D.fillText(Math.round(vY.min), pad.left - 8, pad.top + chartH); ctx2D.fillText(Math.round(vY.max), pad.left - 8, pad.top + 10);
    ctx2D.fillStyle = 'rgba(255,255,255,0.5)'; ctx2D.textAlign = 'center'; ctx2D.fillText(vX.label, pad.left + chartW/2, pad.top + chartH + 30);

    POINTS.forEach(function(pt) {
        var pXVal = (axisXKey==='smoke')? pt[0] : ((axisXKey==='alc')? pt[1] : pt[2]);
        var pYVal = (axisYKey==='smoke')? pt[0] : ((axisYKey==='alc')? pt[1] : pt[2]);
        var cx = getXPos(pXVal); var cy = getYPos(pYVal);
        if(cx >= pad.left && cx <= pad.left+chartW && cy >= pad.top && cy <= pad.top+chartH) {
            ctx2D.beginPath(); ctx2D.arc(cx, cy, 4.5, 0, 2*Math.PI);
            ctx2D.fillStyle = TONE_COLORS[CLUSTER_INFO[pt[3]].tone];
            ctx2D.globalAlpha = 0.6; ctx2D.fill(); ctx2D.globalAlpha = 1.0;
        }
    });

    var curX = getXPos(values[axisXKey]); var curY = getYPos(values[axisYKey]);
    var sz = 8;
    ctx2D.strokeStyle = '#ffffff'; ctx2D.lineWidth = 5; ctx2D.lineCap = 'round';
    ctx2D.beginPath(); ctx2D.moveTo(curX-sz, curY-sz); ctx2D.lineTo(curX+sz, curY+sz); ctx2D.moveTo(curX+sz, curY-sz); ctx2D.lineTo(curX-sz, curY+sz); ctx2D.stroke();
    ctx2D.strokeStyle = '#000000'; ctx2D.lineWidth = 2;
    ctx2D.beginPath(); ctx2D.moveTo(curX-sz, curY-sz); ctx2D.lineTo(curX+sz, curY+sz); ctx2D.moveTo(curX+sz, curY-sz); ctx2D.lineTo(curX-sz, curY+sz); ctx2D.stroke();
}

// 3D WebGL
var container3D = document.getElementById('viewContainer');
var canvas3D = document.getElementById('threeCanvas');
var scene = new THREE.Scene();
var camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
var renderer = new THREE.WebGLRenderer({ canvas: canvas3D, alpha: true, antialias: true });
var controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = 0.07;
camera.position.set(22, 18, 22);

scene.add(new THREE.AxesHelper(15));
var grid = new THREE.GridHelper(30, 20, 0x444444, 0x222222); grid.position.y = -6; scene.add(grid);

function map3X(v) { return ((v - 0)/(40 - 0))*20 - 10; } // smoke
function map3Y(v) { return ((v - 0)/(10 - 0))*20 - 5; }  // alc
function map3Z(v) { return ((v - 15)/(80 - 15))*20 - 10; } // age

POINTS.forEach(function(pt) {
    var m = new THREE.Mesh(new THREE.SphereGeometry(0.35, 12, 12), new THREE.MeshBasicMaterial({
        color: TONE_HEX_THREE[CLUSTER_INFO[pt[3]].tone], transparent: true, opacity: 0.65
    }));
    m.position.set(map3X(pt[0]), map3Y(pt[1]), map3Z(pt[2]));
    scene.add(m);
});

var crossGroup = new THREE.Group();
var cMat = new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 3 });
var cGeo1 = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-1, -1, 0), new THREE.Vector3(1, 1, 0)]);
var cGeo2 = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(1, -1, 0), new THREE.Vector3(-1, 1, 0)]);
crossGroup.add(new THREE.Line(cGeo1, cMat)); crossGroup.add(new THREE.Line(cGeo2, cMat));
scene.add(crossGroup);

function resize3D() {
    var w = container3D.clientWidth; var h = container3D.clientHeight;
    camera.aspect = w / h; camera.updateProjectionMatrix();
    renderer.setSize(w, h);
}

function updateResult() {
    var cid = predictCurrent();
    var info = CLUSTER_INFO[cid];
    
    document.getElementById('r-cluster').textContent = 'Cluster · ' + cid;
    var nameEl = document.getElementById('r-name');
    nameEl.textContent = info.name; nameEl.className = 'result-name ' + info.tone;
    document.getElementById('r-desc').textContent = info.desc;
    
    var dots = document.querySelectorAll('.cluster-dot');
    dots.forEach(function(d, idx) {
        d.className = 'cluster-dot' + (idx === cid ? ' active ' + info.tone : '');
    });
    
    document.getElementById('s-smoke').textContent = Math.round(values.smoke);
    document.getElementById('s-alc').textContent = Math.round(values.alc);
    document.getElementById('s-age').textContent = Math.round(values.age);

    if(currentMode === '2D') {
        draw2D();
    } else {
        crossGroup.position.set(map3X(values.smoke), map3Y(values.alc), map3Z(values.age));
    }
}

// 마우스 드래그 이벤트 수정(e 전달 및 fmt 적용)
document.querySelectorAll('.fader-col').forEach(function(col) {
    var min = parseFloat(col.dataset.min), max = parseFloat(col.dataset.max);
    var key = col.dataset.key, range = max - min;
    var decimals = parseInt(col.dataset.decimals);
    var trackWrap = col.querySelector('.fader-track-wrap'), track = col.querySelector('.fader-track');
    var fill = col.querySelector('.fader-fill'), handle = col.querySelector('.fader-handle'), valueEl = col.querySelector('.fader-value');
    var curVal = values[key];

    function render() {
        var pct = (curVal - min) / range;
        fill.style.height = (pct * 100) + '%';
        handle.style.top = ((1 - pct) * track.getBoundingClientRect().height) + 'px';
        valueEl.textContent = fmt(curVal, decimals);
    }
    function setFromY(clientY) {
        var rect = track.getBoundingClientRect();
        var ratio = 1 - Math.max(0, Math.min(1, (clientY - rect.top) / rect.height));
        curVal = Math.max(min, Math.min(max, min + ratio * range));
        values[key] = curVal; render(); updateResult();
    }
    var dragging = false;
    trackWrap.addEventListener('pointerdown', function(e) { e.preventDefault(); dragging = true; setFromY(e.clientY); try { trackWrap.setPointerCapture(e.pointerId); }catch(v){} });
    trackWrap.addEventListener('pointermove', function(e) { if(dragging) setFromY(e.clientY); });
    
    // 여기서 e를 누락했었습니다!
    function stopDrag(e) { 
        if(!dragging) return;
        dragging = false; 
        try{trackWrap.releasePointerCapture(e.pointerId);}catch(v){} 
    }
    trackWrap.addEventListener('pointerup', stopDrag); 
    trackWrap.addEventListener('pointercancel', stopDrag);
    
    setTimeout(render, 50);
});

window.addEventListener('resize', function() { draw2D(); resize3D(); });

function animate() {
    requestAnimationFrame(animate);
    if(currentMode === '3D') {
        controls.update();
        crossGroup.rotation.y += 0.015;
        renderer.render(scene, camera);
    }
}

setTimeout(function() {
    resize3D();
    updateResult();
    animate();
}, 120);
</script>
</body>
</html>
"""

html = html.replace("__PAYLOAD__", payload_json)
components.html(html, height=610, scrolling=False)
