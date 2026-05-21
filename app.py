import os
import re
import json
import math
import difflib
from collections import Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="GasCode Catcher v1.7", layout="wide")

# ------------------------------------------------------------
# GasCode Catcher v1.7
# 교사용 검토 보조 도구
# - AI 사용 여부를 확정하지 않음
# - 원문 내부 지표 + HuggingFace 재생성 비교 지표를 함께 제공
# ------------------------------------------------------------

LLM_SIGNATURES = [
    "따라서", "결과적으로", "전반적으로", "요약하면", "종합하면", "나아가",
    "이러한 점에서", "중요한 의미", "긍정적인 영향을", "교육적 가치",
    "올바른", "필수적", "바람직하다", "한계와 과제", "시사한다",
]

CONNECTORS = [
    "그리고", "하지만", "그러나", "또한", "따라서", "그러므로", "즉", "한편",
    "반면", "첫째", "둘째", "셋째", "마지막으로", "결과적으로", "나아가",
]

DEFAULT_REGEN_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# -----------------------------
# Basic NLP utilities
# -----------------------------

def clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    text = clean_text(text)
    # Korean/English punctuation + line breaks. Keeps short fragments because OCR/PDF extraction often breaks lines.
    parts = re.split(r"(?<=[.!?。！？다요죠음함됨임])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def tokenize(text: str) -> List[str]:
    # Korean-friendly simple tokenization: Korean syllables, Latin words, numbers.
    return re.findall(r"[가-힣A-Za-z0-9一-龥]+", text.lower())


def safe_entropy(items: List[Tuple[str, ...]]) -> float:
    if not items:
        return 0.0
    counts = Counter(items)
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    return float(-sum(p * math.log2(p) for p in probs if p > 0))


def cosine_counter(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return float(dot / (na * nb)) if na and nb else 0.0


def ngram_counter(tokens: List[str], n: int = 2) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(max(0, len(tokens)-n+1)))

# -----------------------------
# Metrics
# -----------------------------

def compute_metrics(text: str) -> Dict[str, float]:
    text = clean_text(text)
    sentences = split_sentences(text)
    tokens = tokenize(text)
    lengths = [len(tokenize(s)) for s in sentences]

    if len(lengths) >= 2:
        diffs = np.diff(lengths)
        trajectory_discontinuity = float(np.mean(np.abs(diffs)) / max(np.mean(lengths), 1))
        local_rhythm_std = float(np.std(diffs))
    else:
        trajectory_discontinuity = 0.0
        local_rhythm_std = 0.0

    if lengths:
        length_std = float(np.std(lengths))
        repeated_ratio = max(Counter(lengths).values()) / len(lengths) * 100
    else:
        length_std = 0.0
        repeated_ratio = 0.0

    connector_starts = sum(1 for s in sentences if any(s.startswith(c) for c in CONNECTORS))
    connector_ratio = connector_starts / max(len(sentences), 1) * 100

    sig_count = sum(text.count(sig) for sig in LLM_SIGNATURES)
    signature_density = sig_count / max(len(tokens), 1) * 100

    forward_bigrams = list(zip(tokens[:-1], tokens[1:]))
    backward_bigrams = list(zip(tokens[1:], tokens[:-1]))

    # Temporal shift instability: compare first/second half rhythm structures.
    if len(lengths) >= 8:
        half = len(lengths) // 2
        first = np.array(lengths[:half])
        second = np.array(lengths[half:half+len(first)])
        temporal_shift_instability = float(abs(np.mean(first) - np.mean(second)) + abs(np.std(first) - np.std(second)))
    else:
        temporal_shift_instability = 0.0

    # Organic Rhythm Score: high = too smooth/regular, low = more broken/nonlinear.
    organic_score = max(0.0, min(100.0, 100.0 - local_rhythm_std * 8.0))

    return {
        "문장 수": len(sentences),
        "토큰 수": len(tokens),
        "문장 호흡 변동성(표준편차)": round(length_std, 3),
        "Organic Rhythm Score": round(organic_score, 1),
        "LLM 상투어 밀도(%)": round(signature_density, 3),
        "접속어 시작 문장 비율(%)": round(connector_ratio, 2),
        "문장 길이 반복성(%)": round(repeated_ratio, 2),
        "Temporal Rewinding Entropy": round(safe_entropy(backward_bigrams), 3),
        "Forward Entropy": round(safe_entropy(forward_bigrams), 3),
        "Trajectory Discontinuity": round(trajectory_discontinuity, 3),
        "Temporal Shift Instability": round(temporal_shift_instability, 3),
    }


def score_metrics(metrics: Dict[str, float]) -> Tuple[float, List[str]]:
    score = 0.0
    reasons = []

    # 낮은 entropy: 짧거나 전개가 정형적이면 AI/양식문 가능성 증가. 단, 매우 짧은 글은 보정.
    token_count = metrics.get("토큰 수", 0)
    entropy = metrics.get("Temporal Rewinding Entropy", 0)
    organic = metrics.get("Organic Rhythm Score", 0)
    repeat = metrics.get("문장 길이 반복성(%)", 0)
    sig = metrics.get("LLM 상투어 밀도(%)", 0)
    conn = metrics.get("접속어 시작 문장 비율(%)", 0)
    traj = metrics.get("Trajectory Discontinuity", 0)

    if token_count < 150:
        score += 8
        reasons.append("짧은 글이라 지표 신뢰도가 낮아 보수적으로만 반영됨")
    else:
        if entropy < 9.5:
            score += 22
            reasons.append("Temporal Rewinding Entropy가 낮아 전개 조합이 비교적 정형적임")
        elif entropy < 10.5:
            score += 12
            reasons.append("Temporal Rewinding Entropy가 중간 이하임")

    if organic >= 70:
        score += 20
        reasons.append("문장 리듬이 지나치게 매끈하거나 균질함")
    elif organic >= 55:
        score += 10
        reasons.append("문장 리듬이 비교적 안정적임")

    if repeat >= 45:
        score += 18
        reasons.append("문장 길이 반복성이 높음")
    elif repeat >= 35:
        score += 9
        reasons.append("문장 길이 반복성이 다소 높음")

    if sig >= 2.0:
        score += 22
        reasons.append("LLM 상투 표현 밀도가 높음")
    elif sig >= 0.8:
        score += 12
        reasons.append("LLM 상투 표현이 일부 감지됨")

    if conn >= 20:
        score += 15
        reasons.append("접속어로 시작하는 문장 비율이 높음")
    elif conn >= 10:
        score += 8
        reasons.append("접속어 시작 문장이 다소 많음")

    if 0 < traj < 0.25:
        score += 12
        reasons.append("문장 간 길이 변화가 너무 완만함")
    elif traj > 1.8:
        score -= 8
        reasons.append("문장 간 궤적 변화가 커서 인간적/비선형 흐름 가능성이 있음")

    return round(max(0.0, min(score, 100.0)), 1), reasons

# -----------------------------
# HuggingFace regeneration comparison
# -----------------------------

def call_hf_regenerate(text: str, model: str, token: Optional[str], max_new_tokens: int = 450) -> str:
    if not token:
        raise RuntimeError("HuggingFace API Token이 없습니다. Streamlit secrets 또는 사이드바에 입력하세요.")

    endpoint = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}"}
    sample = text[:1600]
    prompt = (
        "다음 글과 같은 주제와 분량의 한국어 글을 새로 작성하세요. "
        "문장 구조와 어휘를 그대로 복사하지 말고, 자연스럽게 다시 쓰세요.\n\n"
        f"[원문]\n{sample}\n\n[새 글]"
    )
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.7,
            "return_full_text": False,
        },
        "options": {"wait_for_model": True},
    }
    r = requests.post(endpoint, headers=headers, json=payload, timeout=90)
    if r.status_code != 200:
        raise RuntimeError(f"HuggingFace 호출 실패: {r.status_code} / {r.text[:300]}")
    data = r.json()
    if isinstance(data, list) and data and "generated_text" in data[0]:
        return data[0]["generated_text"].strip()
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"].strip()
    return json.dumps(data, ensure_ascii=False)[:2000]


def compare_texts(original: str, regenerated: str) -> Dict[str, float]:
    o_tokens = tokenize(original)
    r_tokens = tokenize(regenerated)
    o_sents = split_sentences(original)
    r_sents = split_sentences(regenerated)

    o_bigram = ngram_counter(o_tokens, 2)
    r_bigram = ngram_counter(r_tokens, 2)
    o_trigram = ngram_counter(o_tokens, 3)
    r_trigram = ngram_counter(r_tokens, 3)

    char_ratio = difflib.SequenceMatcher(None, original[:5000], regenerated[:5000]).ratio()
    bigram_cos = cosine_counter(o_bigram, r_bigram)
    trigram_cos = cosine_counter(o_trigram, r_trigram)

    om = compute_metrics(original)
    rm = compute_metrics(regenerated)
    metric_keys = [
        "문장 호흡 변동성(표준편차)", "Organic Rhythm Score", "LLM 상투어 밀도(%)",
        "접속어 시작 문장 비율(%)", "문장 길이 반복성(%)", "Temporal Rewinding Entropy",
        "Trajectory Discontinuity", "Temporal Shift Instability"
    ]
    diffs = []
    for k in metric_keys:
        a = float(om.get(k, 0))
        b = float(rm.get(k, 0))
        denom = max(abs(a), abs(b), 1.0)
        diffs.append(abs(a-b) / denom)
    metric_similarity = max(0.0, 1.0 - float(np.mean(diffs)))

    return {
        "문자열 직접 유사도": round(char_ratio, 3),
        "2-gram 구조 유사도": round(bigram_cos, 3),
        "3-gram 구조 유사도": round(trigram_cos, 3),
        "지표 프로파일 유사도": round(metric_similarity, 3),
        "원문 문장 수": len(o_sents),
        "재생성문 문장 수": len(r_sents),
    }


def comparison_risk(compare: Dict[str, float]) -> Tuple[float, List[str]]:
    risk = 0.0
    reasons = []
    profile = compare.get("지표 프로파일 유사도", 0)
    bigram = compare.get("2-gram 구조 유사도", 0)
    trigram = compare.get("3-gram 구조 유사도", 0)
    direct = compare.get("문자열 직접 유사도", 0)

    if profile >= 0.82:
        risk += 30
        reasons.append("원문과 AI 재생성문의 지표 프로파일이 매우 유사함")
    elif profile >= 0.68:
        risk += 18
        reasons.append("원문과 AI 재생성문의 지표 프로파일이 다소 유사함")

    if bigram >= 0.45:
        risk += 25
        reasons.append("2-gram 구조 유사도가 높음")
    elif bigram >= 0.30:
        risk += 12
        reasons.append("2-gram 구조 유사도가 중간 이상임")

    if trigram >= 0.30:
        risk += 20
        reasons.append("3-gram 구조 유사도가 높음")
    elif trigram >= 0.18:
        risk += 10
        reasons.append("3-gram 구조 유사도가 중간 이상임")

    if direct >= 0.35:
        risk += 15
        reasons.append("문자열 직접 유사도가 높아 복사/재작성 가능성 검토 필요")

    return round(min(risk, 100), 1), reasons

# -----------------------------
# UI
# -----------------------------

st.title("GasCode Catcher v1.7")
st.caption("교사용 AI 글 검토 보조 리포트 · 확정 판정이 아니라 추가 검토 필요성 산출용")

with st.sidebar:
    st.header("설정")
    mode = st.radio("분석 모드", ["기본 지표 분석", "HuggingFace 재생성 비교"], index=0)
    model = st.text_input("HuggingFace 모델", DEFAULT_REGEN_MODEL)
    hf_token = st.text_input("HF API Token", value=os.getenv("HF_TOKEN", ""), type="password")
    max_new_tokens = st.slider("재생성 최대 토큰", 150, 900, 450, 50)
    st.divider()
    st.write("권장 해석")
    st.write("0~30: 낮음 / 30~60: 보조 검토 / 60+: 정밀 검토")

uploaded = st.file_uploader("TXT 파일 업로드", type=["txt"])
text_area = st.text_area("또는 텍스트 직접 입력", height=300)

text = ""
if uploaded is not None:
    text = uploaded.read().decode("utf-8", errors="ignore")
elif text_area.strip():
    text = text_area

if not text:
    st.info("분석할 글을 입력하거나 TXT 파일을 업로드하세요.")
    st.stop()

text = clean_text(text)
metrics = compute_metrics(text)
base_score, base_reasons = score_metrics(metrics)

st.subheader("1. 기본 판정")
col1, col2, col3 = st.columns(3)
col1.metric("추가 검토 필요성", f"{base_score}%")
col2.metric("문장 수", metrics["문장 수"])
col3.metric("토큰 수", metrics["토큰 수"])
st.progress(base_score / 100)

if base_score >= 60:
    st.warning("정밀 검토 권장: 정형적·균질적 문장 흐름 또는 LLM 상투 패턴이 비교적 강합니다.")
elif base_score >= 30:
    st.info("보조 검토 권장: 일부 지표에서 AI/양식문 특성이 감지됩니다.")
else:
    st.success("낮은 검토 필요성: 현재 지표만으로는 강한 AI 패턴이 보이지 않습니다.")

if base_reasons:
    st.write("판정 근거")
    for r in base_reasons:
        st.write(f"- {r}")

st.subheader("2. 세부 지표")
st.dataframe(pd.DataFrame([metrics]).T.rename(columns={0: "값"}), use_container_width=True)

if mode == "HuggingFace 재생성 비교":
    st.subheader("3. HuggingFace 재생성 비교")
    st.caption("원문과 같은 주제로 AI가 다시 쓴 글을 생성한 뒤, 원문과 지표 구조가 얼마나 닮았는지 비교합니다.")

    if st.button("AI 재생성 후 비교 실행"):
        try:
            with st.spinner("재생성 및 비교 중..."):
                regen = call_hf_regenerate(text, model=model, token=hf_token, max_new_tokens=max_new_tokens)
                comp = compare_texts(text, regen)
                comp_score, comp_reasons = comparison_risk(comp)
                final_score = round(min(100.0, base_score * 0.55 + comp_score * 0.45), 1)

            c1, c2, c3 = st.columns(3)
            c1.metric("기본 지표 점수", f"{base_score}%")
            c2.metric("재생성 비교 점수", f"{comp_score}%")
            c3.metric("통합 검토 점수", f"{final_score}%")
            st.progress(final_score / 100)

            st.write("재생성 비교 근거")
            for r in comp_reasons:
                st.write(f"- {r}")

            st.write("비교 지표")
            st.dataframe(pd.DataFrame([comp]).T.rename(columns={0: "값"}), use_container_width=True)

            with st.expander("AI 재생성문 보기"):
                st.write(regen)
        except Exception as e:
            st.error(str(e))
else:
    st.subheader("3. 해석 메모")
    st.write(
        "교수 논문처럼 형식이 강한 인간 글은 접속어·요약문·반복 구조 때문에 일부 AI형 지표가 올라갈 수 있습니다. "
        "반대로 학생 글이나 창작 글은 비문, 호흡 변화, 장면 이동, 불균질한 표현 때문에 인간형으로 낮게 나올 수 있습니다. "
        "따라서 이 도구는 확정 탐지기가 아니라 교사용 검토 우선순위 도구로 쓰는 것이 안전합니다."
    )

with st.expander("원문 미리보기"):
    st.write(text[:8000])
