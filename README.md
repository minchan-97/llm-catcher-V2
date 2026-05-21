[README.md](https://github.com/user-attachments/files/28101378/README.md)
# llm-catcher-V2
llm catcher
# GasCode Catcher v1.7

교사용 AI 글 검토 보조 Streamlit 앱입니다.  
AI 사용 여부를 확정하지 않고, 추가 검토 필요성을 수치화합니다.

## 주요 기능

- 문장 호흡 변동성
- Organic Rhythm Score
- LLM 상투어 밀도
- 접속어 시작 문장 비율
- 문장 길이 반복성
- Temporal Rewinding Entropy
- Trajectory Discontinuity
- Temporal Shift Instability
- HuggingFace 재생성문 비교

## 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud 배포

1. GitHub 저장소 생성
2. `app.py`, `requirements.txt`, `README.md` 업로드
3. Streamlit Community Cloud 접속
4. 저장소 연결
5. Main file path를 `app.py`로 설정
6. Deploy

## HuggingFace 비교 기능 사용법

사이드바에 HuggingFace API Token을 입력하거나 Streamlit secrets에 다음과 같이 등록합니다.

```toml
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxx"
```

기본 모델은 `mistralai/Mistral-7B-Instruct-v0.3`입니다. 한국어 품질이 더 좋은 모델로 교체할 수 있습니다.

## 주의

이 도구는 AI 사용 여부를 확정하는 증거가 아닙니다.  
교사용 검토 우선순위 산출용 보조 지표로 사용해야 합니다.
