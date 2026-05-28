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
import pandas as pd
import joblib

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
    -webkit-font-smoothing: antialiased;
}
.stApp, [data-testid="stAppViewContainer"] { background: #0a0b0d !important; }
[data-testid="stHeader"], [data-testid="stMain"], [data-testid="stMainBlockContainer"],
.main, .block-container { background: transparent !important; }
.main .block-container {
    padding-top: 1.2rem; padding-bottom: 1rem;
    max-width: 1280px;
}
[data-testid="stNumberInput"] { display: none !important; }
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

/* 메인 PREDICT 버튼 */
.stButton > button {
    background: rgba(255,255,255,0.09) !important;
    backdrop-filter: blur(20px);
    color: #fff !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    padding: 0.95rem 2rem !important;
    border-radius: 14px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    width: 100% !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    transition: all 0.25s ease !important;
    margin-top: 0.8rem;
}
.stButton > button:hover {
    background: rgba(255,255,255,0.18) !important;
    border-color: rgba(255,255,255,0.35) !important;
    transform: translateY(-1px);
}
</style>
""", unsafe_allow_html=True)

# Hero
st.markdown("""
<div class="hero-card">
    <h1>폐암 환자 군집 분석</h1>
    <p>Lung Cancer Patient Clustering · K-Means Unsupervised Learning</p>
</div>
""", unsafe_allow_html=True)

# 입력 변수 (노트북 컬럼 순서: 흡연, 음주량, 나이)
VARS = [
    {"key": "smoke", "label": "흡연",   "min": 0,  "max": 40, "default": 10, "decimals": 0},
    {"key": "alc",   "label": "음주량", "min": 0,  "max": 10, "default": 3,  "decimals": 0},
    {"key": "age",   "label": "나이",   "min": 10, "max": 90, "default": 45, "decimals": 0},
]

@st.cache_resource
def load_model():
    try:
        obj1 = joblib.load("lungkmeans.pkl")
    except FileNotFoundError:
        return None, None, "lungkmeans.pkl 파일을 찾을 수 없습니다."
    try:
        obj2 = joblib.load("lungscaler.pkl")
    except FileNotFoundError:
        obj2 = None
    if hasattr(obj1, 'predict'):
        return obj1, obj2, None
    elif obj2 is not None and hasattr(obj2, 'predict'):
        return obj2, obj1, None
    return obj1, obj2, None

model, scaler, load_err = load_model()

CLUSTER_INFO = {
    0: {"name": "중간군",        "tone": "warn",    "desc": "위험 인자가 일부 누적된 중간 수준의 환자군입니다."},
    1: {"name": "건강군",        "tone": "safe",    "desc": "위험 인자가 낮아 상대적으로 건강한 환자군입니다."},
    2: {"name": "고위험군",      "tone": "danger2", "desc": "위험 인자가 가장 높은 폐암 고위험 환자군입니다."},
    3: {"name": "폐암 위험군",   "tone": "danger",  "desc": "흡연·음주가 누적된 위험 환자군입니다."},
}

# 폼 상태용 number_input (숨김 처리됨)
with st.form("cluster_form"):
    smoke = st.number_input("흡연",   min_value=0,  max_value=80,  value=10, step=1, key="smoke_in")
    alc   = st.number_input("음주량", min_value=0,  max_value=20,  value=3,  step=1, key="alc_in")
    age   = st.number_input("나이",   min_value=1,  max_value=120, value=45, step=1, key="age_in")

    # 페이더 UI를 form 내부에 배치하기 위한 placeholder
    fader_placeholder = st.empty()

    submitted = st.form_submit_button("군집 분석하기")

# 예측 (제출되었을 때만)
cluster_id = -1
info = {"name": "대기 중", "tone": "neutral", "desc": "페이더를 조정한 뒤 군집 분석 버튼을 눌러주세요."}
n_clusters = 4

if submitted and model is not None:
    input_df = pd.DataFrame([[smoke, alc, age]], columns=['흡연', '음주량', '나이'])
    if scaler is not None and hasattr(scaler, 'transform'):
        X_proc = scaler.transform(input_df)
    else:
        X_proc = input_df.values
    cluster_id = int(model.predict(X_proc)[0])
    n_clusters = int(getattr(model, 'n_clusters', 4))
    info = CLUSTER_INFO.get(cluster_id, {
        "name": "Cluster " + str(cluster_id),
        "tone": "warn",
        "desc": "분류된 군집입니다."
    })

# UI CSS
fader_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { box-sizing: border-box; font-family: 'Inter', sans-serif; }
body { margin: 0; padding: 0; background: transparent; color: #fff; }

.workspace {
    display: grid;
    grid-template-columns: 1.05fr 1fr;
    gap: 14px;
    width: 100%;
    height: 500px;
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
    box-shadow:
        0 3px 8px rgba(0,0,0,0.55),
        inset 0 1px 0 rgba(255,255,255,0.6);
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
}
.result-name.safe { color: #6bcf9f; }
.result-name.warn { color: #ffd166; }
.result-name.danger { color: #ff8a80; }
.result-name.danger2 { color: #ff6b6b; }
.result-name.neutral { color: rgba(255,255,255,0.5); }
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
"""

# 페이더 컬럼
faders_inner = ''
defaults = {"smoke": smoke, "alc": alc, "age": age}
for v in VARS:
    cur_val = defaults[v['key']]
    rng_text = "{0:g} \u2014 {1:g}".format(v['min'], v['max'])
    val_str = ("{0:." + str(v['decimals']) + "f}").format(float(cur_val))
    faders_inner += (
        '<div class="fader-col"'
        ' data-key="' + v['key'] + '"'
        ' data-min="' + str(v['min']) + '"'
        ' data-max="' + str(v['max']) + '"'
        ' data-decimals="' + str(v['decimals']) + '"'
        ' data-label="' + v['label'] + '"'
        ' data-current="' + str(float(cur_val)) + '">'
        '<div class="fader-label">' + v['label'] + '</div>'
        '<div class="fader-track-wrap">'
        '<div class="fader-track"><div class="fader-fill"></div></div>'
        '<div class="fader-handle"></div>'
        '</div>'
        '<div class="fader-value">' + val_str + '</div>'
        '<div class="fader-range">' + rng_text + '</div>'
        '</div>'
    )

# 결과 카드
dots_html = ''
for i in range(4):
    cls = 'cluster-dot'
    if i == cluster_id:
        cls += ' active ' + info["tone"]
    style = '' if i < n_clusters else ' style="display:none;"'
    dots_html += '<div class="' + cls + '"' + style + '></div>'

cluster_label = "Cluster · " + str(cluster_id) if cluster_id >= 0 else "Cluster —"

body_html = (
    '<div class="workspace">'
    '<div class="panel">'
    '<div class="panel-label">Input</div>'
    '<div class="panel-hint">핸들을 위/아래로 드래그하여 환자 데이터를 입력하세요</div>'
    '<div class="faders">' + faders_inner + '</div>'
    '</div>'
    '<div class="panel">'
    '<div class="panel-label">Cluster Result</div>'
    '<div class="result-headline">'
    '<div class="result-cluster">' + cluster_label + '</div>'
    '<h2 class="result-name ' + info["tone"] + '">' + info["name"] + '</h2>'
    '<p class="result-desc">' + info["desc"] + '</p>'
    '<div class="cluster-dots">' + dots_html + '</div>'
    '</div>'
    '<div class="result-stats">'
    '<div class="stat-block"><div class="v">' + str(int(smoke)) + '</div><div class="l">흡연</div></div>'
    '<div class="stat-block"><div class="v">' + str(int(alc))   + '</div><div class="l">음주량</div></div>'
    '<div class="stat-block"><div class="v">' + str(int(age))   + '</div><div class="l">나이</div></div>'
    '</div>'
    '</div>'
    '</div>'
)

fader_js = """
<script>
(function() {
    var doc = window.parent.document;

    function setInputValue(label, value) {
        var inputs = doc.querySelectorAll('input');
        for (var i = 0; i < inputs.length; i++) {
            var inp = inputs[i];
            if (inp.getAttribute('aria-label') === label) {
                var setter = Object.getOwnPropertyDescriptor(window.parent.HTMLInputElement.prototype, 'value').set;
                setter.call(inp, String(value));
                inp.dispatchEvent(new Event('input', { bubbles: true }));
                inp.dispatchEvent(new Event('change', { bubbles: true }));
                return true;
            }
        }
        return false;
    }

    function fmt(val, decimals) {
        var d = parseInt(decimals);
        if (d === 0) return String(Math.round(val));
        return val.toFixed(d);
    }

    document.querySelectorAll('.fader-col').forEach(function(col) {
        var min = parseFloat(col.dataset.min);
        var max = parseFloat(col.dataset.max);
        var decimals = parseInt(col.dataset.decimals);
        var label = col.dataset.label;
        var range = max - min;
        var trackWrap = col.querySelector('.fader-track-wrap');
        var track = col.querySelector('.fader-track');
        var fill = col.querySelector('.fader-fill');
        var handle = col.querySelector('.fader-handle');
        var valueEl = col.querySelector('.fader-value');
        var currentValue = parseFloat(col.dataset.current);

        function render() {
            var pct = (currentValue - min) / range;
            var trackRect = track.getBoundingClientRect();
            var h = trackRect.height;
            fill.style.height = (pct * 100) + '%';
            handle.style.top = ((1 - pct) * h) + 'px';
            var displayVal = decimals === 0 ? Math.round(currentValue) : parseFloat(currentValue.toFixed(decimals));
            valueEl.textContent = fmt(displayVal, decimals);
        }

        function setFromY(clientY) {
            var trackRect = track.getBoundingClientRect();
            var ratio = 1 - Math.max(0, Math.min(1, (clientY - trackRect.top) / trackRect.height));
            currentValue = Math.max(min, Math.min(max, min + ratio * range));
            render();
        }

        function commit() {
            var displayVal = decimals === 0 ? Math.round(currentValue) : parseFloat(currentValue.toFixed(decimals));
            setInputValue(label, displayVal);
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
            commit();
            try { trackWrap.releasePointerCapture(e.pointerId); } catch(err) {}
        }
        trackWrap.addEventListener('pointerup', stopDrag);
        trackWrap.addEventListener('pointercancel', stopDrag);

        trackWrap.addEventListener('wheel', function(e) {
            e.preventDefault();
            var step = range / 100;
            var dir = e.deltaY < 0 ? 1 : -1;
            currentValue = Math.max(min, Math.min(max, currentValue + dir * step));
            render();
            commit();
        }, { passive: false });
    });
})();
</script>
"""

# 폼 안의 placeholder에 페이더+결과 UI 렌더링
with fader_placeholder.container():
    components.html(fader_css + body_html + fader_js, height=540, scrolling=False)

if load_err:
    st.error(load_err + " 노트북에서 KMeans 모델을 lungkmeans.pkl, StandardScaler를 lungscaler.pkl로 같은 폴더에 저장해 주세요.")
elif scaler is None:
    st.warning("lungscaler.pkl을 찾을 수 없어 raw 입력으로 예측했습니다. 정확도가 떨어질 수 있습니다.")
