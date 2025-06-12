'''
#필요한 라이브러리들, 터미널에서 실행행
# YouTube 댓글 다운로드를 위한 라이브러리 설치
!pip install youtube-comment-downloader

# 언어 감지를 위한 라이브러리 설치 (langid: 텍스트의 언어를 감지해주는 라이브러리)
!pip install langid

# Streamlit 설치 (Streamlit은 간단한 웹 앱을 빠르게 만들 수 있게 해주는 라이브러리)
!pip install streamlit

# ngrok을 통해 로컬 서버를 외부에서도 접속 가능하게 해주는 라이브러리 설치 (streamlit 웹앱을 배포할 때 사용 가능)
!pip install pyngrok

# 언어 코드 관련 처리를 위한 라이브러리 설치 (langcodes: ISO 639 언어 코드 관리 및 변환)
!pip install langcodes

!pip install transformer
'''
#디자인 코드
design_code = '''
import streamlit as st

def apply_css():
    st.markdown("""
    <style>
    body {
      margin: 0;
      padding: 0;
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background-color: #ffffff;
      text-align: center;
      color: #333;
    }
    .center-box {
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      margin-top: 50px;
    }
    .center-box img {
      width: 40px;
      margin-bottom: 20px;
    }
    .center-box .title {
      font-size: 48px;
      color: #e60000;
      font-weight: bold;
      margin: 10px 0;
    }
    .center-box .subtitle {
      font-size: 16px;
      color: #888;
      margin-bottom: 30px;
    }
    .input-style input {
      padding: 12px 20px;
      width: 300px;
      border-radius: 10px;
      border: 1px solid #ccc;
      font-size: 14px;
    }
    .button-style button {
      padding: 12px 20px;
      background-color: #e60000;
      color: white;
      border: none;
      border-radius: 10px;
      font-weight: bold;
      cursor: pointer;
      font-size: 14px;
    }
    .button-style button:hover {
      background-color: #cc0000;
    }
    </style>
    """, unsafe_allow_html=True)

def show_header():
    st.markdown("""
    <div class="center-box">
      <img src="https://upload.wikimedia.org/wikipedia/commons/b/b8/YouTube_Logo_2017.svg" alt="YouTube logo" />
      <div class="title">YouniversAI</div>
      <div class="subtitle">Your smart assistant for multilingual YouTube comment insights.</div>
    </div>
    """, unsafe_allow_html=True)
'''
with open("design.py", "w", encoding="utf-8") as f:
    f.write(design_code)

#댓글수집
comments_collection_code = '''
import os
import json
import pandas as pd

def get_comments(url):
    json_file = 'YoutubeComments.json'
    os.system(f'youtube-comment-downloader --url "{url}" --output {json_file}')

    with open(json_file, 'r', encoding='utf-8') as f:
        content = f.read()
        try:
            json_data = json.loads(content)
        except json.JSONDecodeError:
            json_data = []
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    json_data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    df = pd.DataFrame(json_data)
    os.remove(json_file)
    return df
'''

# 파일로 저장
with open("logic.py", "w", encoding="utf-8") as f:
    f.write(comments_collection_code)

#댓글 분류류
classifier_code = '''
import langid
import langcodes

def classify_language(df):
    df['lang'] = df['text'].apply(lambda x: langid.classify(x)[0])
    df['언어'] = df['lang'].apply(
        lambda code: langcodes.Language.get(code).display_name() if code else code)
    return df

def classify_and_store(df, session_state):
    df = classify_language(df)
    session_state.df = df
    return df
'''

# 파일로 저장
with open("classifier.py", "w", encoding="utf-8") as f:
    f.write(classifier_code)

#요약 전용 파일
summarizer_code='''
import re
import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

# 요약 모델 초기화
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=-1  # CPU 모드
)

# 키워드 추출 모델 초기화
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
kw_model = KeyBERT(model=embed_model)

def summarize_by_language(df):
    result = {}
    # summarizer.py 수정
    grouped = df.groupby('lang', dropna=True)


    for lang, group in grouped:
        comments = group['text'].dropna().astype(str).tolist()
        full_text = " ".join(comments)

        if len(full_text) < 50:
            summary = "(댓글이 너무 적어 요약 불가)"
            keywords = ""
        else:
            try:
                summary = summarizer(full_text[:1500], max_length=150, min_length=30, do_sample=False)[0]['summary_text']
            except:
                summary = "(요약 오류 발생)"
            try:
                keywords = ", ".join([kw[0] for kw in kw_model.extract_keywords(full_text)])
            except:
                keywords = ""

        result[lang] = {
            "summary": summary,
            "keywords": keywords
        }

    return result
'''
# 파일로 저장
with open("summarizer.py", "w", encoding="utf-8") as f:
    f.write(summarizer_code)

#감정 분석 전용 파일일
sentiment_code='''
import langid
import pandas as pd
from transformers import pipeline

# 감정 분석 모델 초기화 (CPU 모드)
sentiment_model = "nlptown/bert-base-multilingual-uncased-sentiment"
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=sentiment_model,
    tokenizer=sentiment_model,
    truncation=True,
    device=-1
)

# 언어 감지 함수 (간단 버전)
def detect_langid(text):
    try:
        text = str(text).strip()
        if len(text) < 2:
            return 'unknown'
        return langid.classify(text)[0]
    except:
        return 'unknown'

# 감정 분석 함수
def analyze_sentiment(df):
    df['language'] = df['text'].apply(detect_langid)
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
        except Exception:
            continue

        df_lang['sentiment'] = [r['label'] for r in results]
        df_lang['score'] = [r['score'] for r in results]
        df_lang['sentiment_score'] = df_lang['sentiment'].apply(lambda l: int(l.split()[0]))
        df_lang['sentiment_kor'] = df_lang['sentiment_score'].apply(
            lambda s: "부정" if s <= 2 else ("중립" if s == 3 else "긍정")
        )

        dfs_by_lang[lang] = df_lang

    if not dfs_by_lang:
        return df.copy()

    return pd.concat(dfs_by_lang.values(), ignore_index=True)
'''
# 파일로 저장
with open("sentiment.py", "w", encoding="utf-8") as f:
    f.write(sentiment_code)




'''
#최종 실행 파일(감정분석+요약약)
!pip install keybert sentence-transformers transformers --quiet #터미널에서 실행행
'''
streamlit_code='''
import streamlit as st
import langcodes
from design import apply_css, show_header
from logic import get_comments
from classifier import classify_and_store
from summarizer import summarize_by_language
from sentiment import analyze_sentiment  # ✅ 감정 분석 추가

def main():
    apply_css()
    show_header()

    url = st.text_input("Paste YouTube video URL here")
    if st.button("Go"):
        st.video(url)  # ✅ 유튜브 영상 표시
        st.write(f"\U0001F4FA You entered: {url}")
        with st.spinner("\U0001F50E댓글을 분류중입니다... 잠시만 기다려주세요"):
            df = get_comments(url)
            if df.empty or 'text' not in df.columns:
                st.warning("No comments found or 'text' field missing.")
                return
            classify_and_store(df, st.session_state)

    if 'df' in st.session_state:
        df = st.session_state.df
        st.success("댓글 수집 및 언어 분류 완료!")
        st.write("총 댓글 수:", len(df))

        languages = df['lang'].unique().tolist()
        language_options = [langcodes.Language.get(code).display_name() for code in languages]

        selected_langs = st.multiselect("언어를 선택하세요", options=language_options, default=language_options)
        selected_lang_codes = [code for code in languages if langcodes.Language.get(code).display_name() in selected_langs]
        filtered_df = df[df['lang'].isin(selected_lang_codes)]

        if not filtered_df.empty:
            st.write(filtered_df[['언어', 'text']])

            # ✅ 감정 분석 버튼 추가
            if st.button("감정 분석하기"):
                with st.spinner("\U0001F914 감정을 분석 중입니다..."):
                    df_with_sentiment = analyze_sentiment(filtered_df)
                    st.session_state.df = df_with_sentiment  # 업데이트
                    st.success("감정 분석 완료!")
                    st.dataframe(df_with_sentiment[['언어', 'text', 'sentiment_kor']])

            # ✅ 요약 버튼 추가
            if st.button("요약하기"):
                with st.spinner("\u23F3 언어별 요약 중입니다..."):
                    st.session_state.summary = summarize_by_language(filtered_df)

                for lang_code in selected_lang_codes:
                    summary = st.session_state.summary.get(lang_code)
                    lang_label = langcodes.Language.get(lang_code).display_name()
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
'''
# 실제 파일로 저장
with open("streamlit_app.py", "w", encoding="utf-8") as f:
    f.write(streamlit_code)
