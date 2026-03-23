import streamlit as st
import pandas as pd
import numpy as np
import colorsys
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# --- [1] 설정 ---
OPENAI_API_KEY = "sk-xxxxxxxxxx"
client = OpenAI(api_key=OPENAI_API_KEY)

MODEL_NAME = 'intfloat/multilingual-e5-base'

DB_CONFIG = {
    "host": "aitest",
    "database": "edb",
    "user": "enterprisedb",
    "password": "enterprisedb",
    "port": "5444"
}

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)

embed_model = load_embedding_model()

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def get_e5_embedding(text, is_query=True):
    prefix = "query: " if is_query else "passage: "
    return embed_model.encode([prefix + text])[0].tolist()

# --- [2] DB 초기화 (rgb_vector를 vector(3)로 처리) ---

def init_database():
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("DROP TABLE IF EXISTS color_semantic_search") # 구조 변경을 위해 초기화 시 삭제
        cur.execute("""
            CREATE TABLE color_semantic_search (
                id SERIAL PRIMARY KEY,
                hex_name TEXT,
                color_name TEXT,
                rgb_vector vector(3),
                description TEXT,
                embedding vector(768)
            )
        """)

        st.info("256개 색상 데이터를 생성 중입니다...")
        color_batch = []

        for i in range(256):
            hue = i / 256.0
            r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, 0.7, 0.8)]
            hex_val = '#%02x%02x%02x' % (r, g, b)

            # 색상 이름 판별
            if hue < 0.05 or hue >= 0.95: c_name = "빨간색"
            elif hue < 0.15: c_name = "오렌지색"
            elif hue < 0.25: c_name = "노란색"
            elif hue < 0.45: c_name = "초록색"
            elif hue < 0.55: c_name = "청록색"
            elif hue < 0.75: c_name = "파란색"
            elif hue < 0.85: c_name = "보라색"
            else: c_name = "분홍색"

            desc = f"이 색상은 {c_name} 계열이며 코드는 {hex_val.upper()}입니다."

            # 텍스트 임베딩 생성
            vec_768 = get_e5_embedding(desc, is_query=False)
 
            # 데이터 준비 (rgb_vector도 리스트 형태로 전달)
            color_batch.append((hex_val.upper(), c_name, [r, g, b], desc, vec_768))

        # INSERT 시 두 벡터 컬럼 모두 캐스팅 적용
        insert_query = """
            INSERT INTO color_semantic_search (hex_name, color_name, rgb_vector, description, embedding)
            VALUES %s
        """
        execute_values(cur, insert_query, color_batch,
                       template="(%s, %s, %s::vector, %s, %s::vector)")

        conn.commit()
        st.success("데이터베이스가 vector(3) 구조로 초기화되었습니다!")

    except Exception as e:
        st.error(f"초기화 실패: {e}")
    finally:
        cur.close()
        conn.close()

# --- [3] 검색 및 답변 생성 ---

def perform_search(query):
    query_vec = get_e5_embedding(query)
    conn = get_db_connection()

    # rgb_vector를 출력할 때 텍스트로 변환해서 가져오면 파이썬에서 다루기 쉽습니다.
    search_query = """
        SELECT hex_name, color_name, rgb_vector::text, (1 - (embedding <=> %s::vector)) AS similarity
        FROM color_semantic_search
        ORDER BY embedding <=> %s::vector
        LIMIT 3;
    """
    df = pd.read_sql(search_query, conn, params=(query_vec, query_vec))
    conn.close()
    return df

def generate_ai_response(user_query, results):
    context = ""
    for _, row in results.iterrows():
        context += f"- 이름: {row['color_name']}, 코드: {row['hex_name']}, RGB: {row['rgb_vector']}\n"

    prompt = f"사용자 요청 '{user_query}'에 대해 다음 색상들을 추천하는 이유를 설명해줘.\n{context}"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# --- [4] UI ---

st.title("🎨 RGB Vector 기반 시맨틱 검색")

if st.sidebar.button("DB 초기화 (vector(3) 적용)"):
    init_database()

query = st.text_input("원하는 색상의 느낌을 입력하세요")

if query:
    results = perform_search(query)
    if not results.empty:
        # AI 답변 생성
        st.info(generate_ai_response(query, results))

        # 결과 시각화
        cols = st.columns(3)
        for i, (_, row) in enumerate(results.iterrows()):
            with cols[i]:
                st.markdown(f'<div style="background:{row["hex_name"]}; height:100px; border-radius:10px;"></div>', unsafe_allow_html=True)
                st.write(f"**{row['color_name']}** ({row['hex_name']})")
