### rgb_semantic_search 
PostgreSQL(EPAS) 의 pgvector 를 이용한 RGB 의미 검색 코드입니다. <br>
사용한 임베딜 모델은 intfloat/multilingual-e5-base 이며 <br>
의미 검색의 검증을 위해 OpenAI의 gpt-4o 를 사용하였습니다. <br>
웹 애플리케이션 프레임워크는 streamlit 을 사용하였습니다. 


#### 테스트 방법
0. python 환경 만들기
```
python3.12 -m venv venv

# 가상환경 활성화
source venv/bin/activate

# 버전 확인 (이제 python만 쳐도 3.12로 실행됩니다)
python --version

pip install google-generativeai streamlit pandas numpy sentence-transformers psycopg2-binary openai
```

1. git clone
```
git clone https://github.com/unixfree/rgb_semantic_search.git
```

2. 수행
```
streamlit run app.py
```




