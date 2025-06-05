import streamlit as st
import os
import json
import pandas as pd
import langid
import langcodes
import re
from transformers import pipeline
from langdetect import detect
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
# ─────────────────────────────────────────

# ── 감정 분석 & 요약 파이프라인(CPU 모드) ──
sentiment_model = "nlptown/bert-base-multilingual-uncased-sentiment"
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=sentiment_model,
    tokenizer=sentiment_model,
    truncation=True,
    device=-1             # CPU 모드 강제
)

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=-1             # CPU 모드 강제
)

# ── KeyBERT도 CPU 전용으로 초기화 ──
embed_model = SentenceTransformer(
    "all-MiniLM-L6-v2",  # or your preferred SBERT model
    device="cpu"         # ← 꼭 CPU 모드로 지정
)
kw_model = KeyBERT(model=embed_model)


# -------------------- 전처리 및 분석 함수 --------------------

def is_korean_internet_slang(text):
    if pd.isna(text):
        return False
    text = str(text).strip()
    return (
        bool(re.fullmatch(r'[ㄱ-ㅎㅏ-ㅣㅜㅠㅋㅎ\s]+', text))
        or any(kw in text for kw in ['ㄹㅇ', 'ㅇㅈ', 'ㅅㅂ', 'ㅋㅋㅋ', 'ㄷㄷ'])
    )

def clean_text(text):
    if pd.isna(text):
        return ''
    return re.sub(r'[^\w\sㄱ-힣]', '', str(text))

def detect_langid_custom(text):
    try:
        text = str(text).strip()
        if len(text) < 2:
            return 'unknown'
        if is_korean_internet_slang(text):
            return 'ko'
        cleaned = clean_text(text)
        lang, _ = langid.classify(cleaned)
        return lang
    except:
        return 'unknown'

def get_comments(url):
    json_file = 'YoutubeComments.json'
    os.system(f'youtube-comment-downloader --url "{url}" --output {json_file}')
    with open(json_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        data = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    os.remove(json_file)
    return pd.DataFrame(data)

def analyze_sentiment(df):
    # 언어 감지 후 빈 데이터 처리
    df['language'] = df['text'].apply(detect_langid_custom)
    dfs_by_lang = {}
    for lang in df['language'].unique():
        df_lang = df[df['language'] == lang].copy()
        if df_lang.empty:
            continue

        comments = df_lang['text'].dropna().astype(str).tolist()
        if not comments:
            continue

        try:
            results = sentiment_pipeline(comments)
        except Exception as e:
            # 모델 추론 중 오류가 나면 이 언어는 건너뛰기
            continue

        df_lang['sentiment'] = [r['label'] for r in results]
        df_lang['score'] = [r['score'] for r in results]
        df_lang['sentiment_score'] = df_lang['sentiment'].apply(lambda l: int(l.split()[0]))
        df_lang['sentiment_kor'] = df_lang['sentiment_score'].apply(
            lambda s: "부정" if s <= 2 else ("중립" if s == 3 else "긍정")
        )
        dfs_by_lang[lang] = df_lang

    if not dfs_by_lang:
        # 만약 하나도 남은 언어가 없다면 원본 df(언어·감정 없는 상태) 리턴
        return df.copy()
    return pd.concat(dfs_by_lang.values(), ignore_index=True)

def summarize_by_language(df):
    result = {}
    # 언어 코드별로 그룹핑
    grouped = df.groupby('language', dropna=True)
    for lang, group in grouped:
        comments = group['text'].dropna().astype(str).tolist()
        full_text = " ".join(comments)

        if len(full_text) < 50:
            summary = "(댓글이 너무 적어 요약 불가)"
            keywords = ""
        else:
            try:
                # 1,500자까지만 잘라서 요약
                summary = summarizer(full_text[:1500], max_length=150, min_length=30, do_sample=False)[0]['summary_text']
            except Exception as e:
                summary = "(요약 오류 발생)"
            # 키워드 추출
            try:
                keywords = ", ".join([kw[0] for kw in kw_model.extract_keywords(full_text)])
            except:
                keywords = ""
        result[lang] = {"summary": summary, "keywords": keywords}
    return result

# ------------------------ Streamlit 웹 UI ------------------------

def main():
    st.title("YouniversAI")
    url = st.text_input("Paste YouTube video URL here")

    if st.button("Go"):
        with st.spinner("🔎 댓글을 수집하고 분석 중입니다..."):
            df = get_comments(url)
            if df.empty or 'text' not in df.columns:
                st.warning("No comments found.")
                return

            # ── 감정 분석 및 언어 이름 추가 ──
            df = analyze_sentiment(df)

            def get_lang_name(code):
                try:
                    return langcodes.Language.get(code).display_name("en")
                except:
                    return "Unknown"

            df['lang_name'] = df['language'].apply(get_lang_name)
            st.session_state.df = df

            # ── 언어별 요약 결과를 세션에 캐시 저장 ──
            st.session_state.summary = summarize_by_language(df)

    if 'df' in st.session_state:
        df = st.session_state.df
        st.success("✅ 댓글 분석 완료!")
        st.write(f"총 댓글 수: {len(df)}")

        # 언어 선택 UI: 사람이 읽을 수 있는 이름
        lang_options = df['lang_name'].dropna().unique().tolist()
        selected_langs = st.multiselect("언어를 선택하세요", options=lang_options, default=lang_options)

        # 언어 이름 → 언어 코드 매핑
        def safe_display_name(code):
            try:
                return langcodes.Language.get(code).display_name("en")
            except:
                return None

        lang_name_to_code = {}
        for code in df['language'].dropna().unique():
            name = safe_display_name(code)
            if name:
                lang_name_to_code[name] = code

        selected_codes = [lang_name_to_code[name] for name in selected_langs if name in lang_name_to_code]
        filtered_df = df[df['language'].isin(selected_codes)]

        if not filtered_df.empty:
            st.write("📋 필터링된 댓글")
            st.dataframe(filtered_df[['lang_name', 'text', 'sentiment_kor']])

            # 언어별 요약·키워드 출력
            for lang_code in selected_codes:
                summary = st.session_state.summary.get(lang_code)
                lang_label = safe_display_name(lang_code) or "Unknown"
                st.markdown(f"### 💬 {lang_label} 요약")

                if summary:
                    st.markdown(f"**요약:** {summary.get('summary', '(요약 없음)')}")
                    st.markdown(f"**키워드:** {summary.get('keywords', '(키워드 없음)')}")
                else:
                    st.info("요약 결과가 없습니다.")
        else:
            st.warning("선택한 언어에 해당하는 댓글이 없습니다.")

if __name__ == "__main__":
    main()
