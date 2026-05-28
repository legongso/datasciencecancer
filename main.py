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

# ===== Streamlit Main UI Style (지마켓 산스 적용) =====
st.markdown("""
<style>
@font-face {
    font-family: 'GmarketSans';
    src: url('https://fastly.jsdelivr.net/gh/projectnoonnu/noonfonts_2001@1.1/GmarketSansLight.woff') format('woff');
    font-weight: 300;
    font-style: normal;
}
@font-face {
    font-family: 'GmarketSans';
    src: url('https://fastly.jsdelivr.net/gh/projectnoonnu/noonfonts_2001@1.1/GmarketSansMedium.woff') format('woff');
    font-weight: 500;
    font-style: normal;
}
@font-face {
    font-family: 'GmarketSans';
    src: url('https://fastly.jsdelivr.net/gh/projectnoonnu/noonfonts_2001@1.1/GmarketSansBold.woff') format('woff');
    font-weight: 700;
    font-style: normal;
}

html, body, .stApp, .stApp * {
    font-family: 'GmarketSans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.stApp, [data-testid="stAppViewContainer"] { background: #0a0b0d !important; }
[data-testid="stHeader"], [data-testid="stMain"], [data-testid="stMainBlockContainer"],
.main, .block-container { background: transparent !important; }
.main .block-container {
    padding-top: 1.2rem; padding-bottom: 1rem;
    max-width: 1400px;
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
    <h1>폐암 환자 군집 분석 (3D 인터랙티브 Map)</h1>
    <p>Lung Cancer Patient Clustering · K-Means 3D WebGL Visualization</p>
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
    st.error((load_err or "모델 파일을 불러올 수 없습니다.") +
             " 노트북에서 KMeans 모델을 lungkmeans.pkl, StandardScaler를 lungscaler.pkl로 같은 폴더에 저장해 주세요.")
    st.stop()

# K=4 기준 군집 의미 및 색상 매핑
CLUSTER_INFO_LIST = [
    {"name": "중간군",        "tone": "danger",  "desc": "위험 인자가 일부 누적된 중간 수준의 환자군입니다."},
    {"name": "건강군",        "tone": "safe",    "desc": "위험 인자가 낮아 상대적으로 건강한 환자군입니다."},
    {"name": "고위험군",      "tone": "danger2", "desc": "위험 인자가 가장 높은 폐암 고위험 환자군입니다."},
    {"name": "폐암 위험군",   "tone": "warn",    "desc": "흡연·음주가 누적된 위험 환자군입니다."},
]

# 3D 공간을 채울 가상 데이터셋 (나이, 흡연량, 음주량, 군집 ID)
dataset_points = [
    # 건강군 (초록 계열) - 나이 낮음, 흡연 낮음, 음주 낮음
    [25, 2, 1, 1], [28, 0, 0, 1], [30, 1, 2, 1], [34, 0, 1, 1], [33, 4, 2, 1], [45, 2, 1, 1], [40, 4, 2, 1],
    # 중간군 (보라 계열) - 전반적으로 중간 수치에 분포
    [18, 10, 4, 0], [22, 12, 3, 0], [25, 20, 5, 0], [28, 10, 4, 0], [34, 12, 5, 0], [37, 15, 4, 0], [42, 12, 5, 0], [51, 25, 3, 0],
    # 폐암 위험군 (노랑 계열) - 흡연량과 음주량이 매우 높은 군집
    [26, 34, 8, 3], [34, 25, 9, 3], [40, 20, 7, 3], [43, 30, 8, 3], [44, 30, 9, 3], [48, 35, 6, 3], [55, 28, 8, 3],
    # 고위험군 (청회색 계열) - 나이가 많고 고위험인 군집
    [58, 15, 4, 2], [62, 5, 3, 2], [62, 20, 5, 2], [62, 25, 6, 2], [69, 20, 4, 2], [73, 10, 2, 2], [77, 20, 5, 2], [75, 15, 3, 2]
]

data_payload = {
    "params": params,
    "clusterInfo": CLUSTER_INFO_LIST,
    "points": dataset_points,
    "vars": [
        {"key": "smoke", "label": "흡연량",   "min": 0,  "max": 40, "default": 10, "decimals": 0},
        {"key": "alc",   "label": "음주량",   "min": 0,  "max": 10, "default": 3,  "decimals": 0},
        {"key": "age",   "label": "나이",     "min": 15, "max": 80, "default": 45, "decimals": 0},
    ]
}
payload_json = json.dumps(data_payload)

# ===== HTML 컴포넌트 (Three.js 기반 3D Scatter Plot 구현) =====
html = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
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

* { box-sizing: border-box; font-family: 'GmarketSans', sans-serif; }
body { margin: 0; padding: 0; background: transparent; color: #fff; overflow: hidden; }

.workspace {
    display: grid;
    grid-template-columns: 280px 1fr 300px;
    gap: 14px;
    width: 100%;
    height: 560px;
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
    gap: 12px;
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
    letter-spacing: 0.12em;
    font-weight: 600;
}
.fader-track-wrap {
    position: relative;
    width: 46px;
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
    width: 36px;
    height: 18px;
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
    font-size: 1.25rem;
    font-weight: 800;
    letter-spacing: -0.02em;
}
.fader-range {
    color: rgba(255,255,255,0.28);
    font-size: 0.58rem;
    letter-spacing: 0.06em;
}

/* 3D 캔버스 영역 */
.chart-container {
    flex: 1;
    position: relative;
    width: 100%;
    min-height: 0;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 12px;
}
#threeCanvas { width: 100%; height: 100%; display: block; }

/* 차트 내부 축 가이드 텍스트 설명 */
.axis-legend {
    position: absolute;
    bottom: 10px; left: 15px;
    font-size: 0.68rem;
    color: rgba(255,255,255,0.35);
    line-height: 1.4;
    pointer-events: none;
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
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.045em;
    line-height: 1.05;
    margin: 0;
    color: #fff;
    transition: color 0.4s ease;
}
.result-name.safe { color: #6bcf9f; }
.result-name.warn { color: #ffd166; }
.result-name.danger { color: #b388ff; } 
.result-name.danger2 { color: #90caf9; } 

.result-desc {
    color: rgba(255,255,255,0.6);
    font-size: 0.85rem;
    line-height: 1.55;
    margin: 14px auto 0 auto;
    max-width: 260px;
}

.cluster-dots { display: flex; justify-content: center; gap: 14px; margin-top: 20px; }
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
.cluster-dot.active.danger { background: #b388ff; border-color: #b388ff; box-shadow: 0 0 14px rgba(179,136,255,0.7); }
.cluster-dot.active.danger2 { background: #90caf9; border-color: #90caf9; box-shadow: 0 0 14px rgba(144,202,249,0.7); }

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
        <div class="panel-label">Input Parameters</div>
        <div class="panel-hint">핸들을 조절해 변수를 실시간 변경해보세요</div>
        <div class="faders" id="faders"></div>
    </div>
    
    <div class="panel">
        <div class="panel-label">3D Cluster Space Map</div>
        <div class="panel-hint">마우스 드래그: 회전 | 휠: 확대/축소</div>
        <div class="chart-container" id="canvasContainer">
            <canvas id="threeCanvas"></canvas>
            <div class="axis-legend">
                <span style="color:#ff6b6b">■ X축: 나이 (Red)</span><br>
                <span style="color:#51cf66">■ Y축: 흡연량 (Green)</span><br>
                <span style="color:#339af0">■ Z축: 음주량 (Blue)</span>
            </div>
        </div>
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
            <div class="stat-block"><div class="v" id="s-smoke">10</div><div class="l">흡연량</div></div>
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
var POINTS = DATA.points;
var values = {};
VARS.forEach(function(v) { values[v.key] = v.default; });

var TONE_COLORS = {
    "safe": 0x6bcf9f,
    "warn": 0xffd166,
    "danger": 0xb388ff,
    "danger2": 0x90caf9
};

// 페이더 UI 생성
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

function predict() {
    var x = [values.smoke, values.alc, values.age];
    var scaled = [];
    for (var i = 0; i < x.length; i++) {
        scaled.push((x[i] - PARAMS.mean[i]) / PARAMS.scale[i]);
    }
    var bestIdx = 0;
    var bestDist = Infinity;
    for (var c = 0; c < PARAMS.centers.length; c++) {
        var center = PARAMS.centers[c];
        var d = 0;
        for (var j = 0; j < center.length; j++) {
            var diff = scaled[j] - center[j];
            d += diff * diff;
        }
        if (d < bestDist) { bestDist = d; bestIdx = c; }
    }
    return bestIdx;
}

// ===== Three.js 3D 엔진 초기화 및 공간 구현 =====
var container = document.getElementById('canvasContainer');
var canvas = document.getElementById('threeCanvas');

var scene = new THREE.Scene();
var camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
var renderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true, antialias: true });
renderer.setSize(container.clientWidth, container.clientHeight);

var controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;

// 카메라 초기 시점 위치 배치
camera.position.set(25, 20, 25);
controls.target.set(0, 0, 0);
controls.update();

// 축 가이드 라인 배치 생성 (기본 크기: 15)
var axesHelper = new THREE.AxesHelper(15);
scene.add(axesHelper);

// 공간 내부 그리드(Grid) 격자 바닥 배치
var gridHelper = new THREE.GridHelper(30, 30, 0x444444, 0x222222);
gridHelper.position.y = -5;
scene.add(gridHelper);

// 데이터 범위 바운더리 매핑용 정규화 헬퍼 함수
// 나이(15~80) -> (-10~10), 흡연량(0~40) -> (-5~15), 음주량(0~10) -> (-10~10)
function mapX(val) { return ((val - 15) / (80 - 15)) * 20 - 10; }
function mapY(val) { return ((val - 0) / (40 - 0)) * 20 - 5; }
function mapZ(val) { return ((val - 0) / (10 - 0)) * 20 - 10; }

// 데이터셋 점(구체 객체형태) 생성 및 3D 공간 추가
POINTS.forEach(function(pt) {
    var geo = new THREE.SphereGeometry(0.4, 16, 16);
    var info = CLUSTER_INFO[pt[3]];
    var colHex = TONE_COLORS[info.tone] || 0xffffff;
    var mat = new THREE.MeshBasicMaterial({ color: colHex, transparent: true, opacity: 0.7 });
    var mesh = new THREE.Mesh(geo, mat);
    
    mesh.position.set(mapX(pt[0]), mapY(pt[1]), mapZ(pt[2]));
    scene.add(mesh);
});

// 실시간 트래킹 입력값 'X' 표시용 3D 오브젝트 제작
var crossGroup = new THREE.Group();
var crossMat = new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 3 });
var size = 1.2;

// X 모양을 이룰 기하학 구조선 생성
var g1 = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-size, -size, 0), new THREE.Vector3(size, size, 0)]);
var g2 = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(size, -size, 0), new THREE.Vector3(-size, size, 0)]);
var g3 = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, -size, -size), new THREE.Vector3(0, size, size)]);

crossGroup.add(new THREE.Line(g1, crossMat));
crossGroup.add(new THREE.Line(g2, crossMat));
crossGroup.add(new THREE.Line(g3, crossMat));
scene.add(crossGroup);

function updateTargetPosition() {
    crossGroup.position.set(mapX(values.age), mapY(values.smoke), mapZ(values.alc));
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
        if (i === clusterId) d.classList.add('active', info.tone);
    });

    document.getElementById('s-smoke').textContent = Math.round(values.smoke);
    document.getElementById('s-alc').textContent = Math.round(values.alc);
    document.getElementById('s-age').textContent = Math.round(values.age);

    updateTargetPosition();
}

// 페이더 로직 연동
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
        fill.style.height = (pct * 100) + '%';
        handle.style.top = ((1 - pct) * track.getBoundingClientRect().height) + 'px';
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

    var dragging = false;
    trackWrap.addEventListener('pointerdown', function(e) {
        e.preventDefault(); dragging = true;
        col.classList.add('dragging'); setFromY(e.clientY);
        try { trackWrap.setPointerCapture(e.pointerId); } catch(err) {}
    });
    trackWrap.addEventListener('pointermove', function(e) { if (dragging) setFromY(e.clientY); });
    function stopDrag(e) {
        if (!dragging) return; dragging = false; col.classList.remove('dragging');
        try { trackWrap.releasePointerCapture(e.pointerId); } catch(err) {}
    }
    trackWrap.addEventListener('pointerup', stopDrag);
    trackWrap.addEventListener('pointercancel', stopDrag);
});

// 화면 크기 반응형 갱신 루프
window.addEventListener('resize', function() {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
});

// Three.js 애니메이션 프레임 렌더링 루프 (지속 호출)
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    
    // 타겟 'X' 표시를 살짝 회전시켜 시각적인 포인팅 효과 부여
    crossGroup.rotation.y += 0.01;
    
    renderer.render(scene, camera);
}

setTimeout(function() {
    updateResult();
    animate();
}, 100);
</script>
</body>
</html>
"""

html = html.replace("__PAYLOAD__", payload_json)
components.html(html, height=600, scrolling=False)
