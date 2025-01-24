import os
import google.generativeai as genai
import json
from get_namuwiki_docs import load_namuwiki_docs_selenium
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st

# with open("key.json", 'r') as file:
#     data = json.load(file)
    
# gemini_api_key = data.get("gemini-key")pip 

# TODO: 아래 YOUR-HUGGINGFACE-API-KEY랑 OUR-GEMINI-API-KEY에 자기꺼 넣기
if not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "you are Secret key"    
gemini_api_key = "you are Secret key"


genai.configure(api_key=gemini_api_key)

# gemini 모델 로드 
def load_model():
    with st.spinner("모델을 로딩하는 중..."):
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    print("Model loaded...")
    return gemini_model

# 임베딩 로드
def load_embedding():
    with st.spinner("임베딩을 로딩하는 중..."):
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    print("Embedding loaded...")
    return embedding

# Faiss vector DB 생성
def create_vectorstore(topic):     
    with st.spinner("나무위키에서 문서를 가져오는 중..."):
        # text = load_namuwiki_docs_selenium(topic)        
        # st.write(f"찾은 문서 예시:\n{text[:100]}")
     # text = db.load()

        text = """
위키독스
위키
블로그
 로그인
위키독스
위키독스는 온라인 책을 제작 공유하는 플랫폼 서비스입니다.


 위키독스 옵시디언 플러그인 오픈! (2024년 11월 30일)
 블로그 서비스 오픈 (2024년 11월)
 다크모드 기능 추가 (2024년 8월)

추천책
공개책
전자책
책제목 또는 지은이를 입력하세요
※ 위키독스에 공개된 책입니다. (5페이지 이상의 책들만 노출됩니다.)
1
점프 투 파이썬
- 박응용
- 2025년 01월 13일
- 
- e-book
2
딥 러닝을 이용한 자연어 처리 입문
- Bryce 외 1명
- 2025년 01월 18일
- 
- e-book
3
파이썬으로 배우는 알고리즘 트레이딩 (개정판-2쇄)
- 조대표 외 1명
- 2023년 05월 30일
- 
4
딥 러닝 파이토치 교과서 - 입문부터 파인튜닝까지
- Bryce 외 1명
- 2025년 01월 17일
- 
- e-book


5
점프 투 자바
- 박응용
- 2024년 12월 08일
- 
- e-book
6
<랭체인LangChain 노트> - LangChain 한국어 튜토리얼🇰🇷
- 테디노트
- 2025년 01월 16일
- 
7
주식 시장을 이기는 마법의 자동매매
- 엑슬론
- 2023년 12월 18일
- 
8
왕초보를 위한 Python: 쉽게 풀어 쓴 기초 문법과 실습
- 전뇌해커
- 2024년 10월 08일
- 
- e-book
9
[ 문과생도 할 수 있는 ] 파이썬 업무 자동화 일잘러 되기 + 챗GPT
- 메이허
- 2024년 12월 22일
- 
10
미운코딩새끼: 4시간만에 끝내는 파이썬 기초
- 김왼손과 집단지성들
- 2018년 05월 22일
- 
11
초보자를 위한 파이썬 300제
- 조대표 외 1명
- 2024년 06월 23일
- 
12
PyQt5 Tutorial - 파이썬으로 만드는 나만의 GUI 프로그램
- Dardao
- 2021년 05월 19일
- 


13
사장님 몰래 하는 파이썬 업무자동화(부제 : 들키면 일 많아짐)
- 정용범, 손상우 외 1명
- 2024년 08월 27일
- 
14
[Python 완전정복 시리즈] 2편 : Pandas DataFrame 완전정복
- 김태준
- 2022년 03월 14일
- 
15
파이썬을 이용한 비트코인 자동매매 (개정판)
- 조대표 외 1명
- 2023년 01월 13일
- 
16
점프 투 장고
- 박응용
- 2024년 12월 10일
- 
- e-book
17
Machine Learning 강의노트
- 박수진
- 2021년 09월 30일
- 
18
위키독스
- 박응용 외 1명
- 2025년 01월 22일
- 
19
C++ 이야기(A Story of C++)
- SEADOG
- 2024년 12월 02일
- 
20
Must Learning with R (개정판)
- DoublekPark 외 1명
- 2023년 02월 22일
- 
이전12345...76다음

블로그 소식
TG_Miniapp
 0
텔레그램 미니앱 스토어
1. 텔레그램 앱센터 http://t.me/tappsbot 2. 파인드 미니앱 https://www.findmini.app/ 카테고리별로 미니앱 분류 해주는 웹사이트 텔레그램봇 : …

3시간 전
 5  0  0
키모
 2
C# &, |와 &&, ||의 차이
1. & (비트 AND) - 모든 조건을 무조건 평가(첫 번째 조건이 false이여도 두번째 조건을 확인한다) …

10시간 전
 46  0  0
TG_Miniapp
 0
텔레그램 미니앱과 웹3 연동
사례 5 1. bums : http://app.bums.bot http://t.me/bums 2. w-coin : https://w-coin.io http://t.me/wcointapbot 3. tapswap : …

14시간 전
 9  0  0
IT기술공유
 3
Blog Image
숫자 맞추기 게임 만들기 - C# 프로그래밍
★ 숫자 맞추기 게임 숫자 맞추기 게임은 임의로 생성된 숫자를 정해진 횟수 내에 맞추는 게임입니다. …

1일 전
 31  1  0
IT기술공유
 3
Blog Image
로그인 창 만들기 - C# 프로그래밍
★ 로그인 인터넷 환경에서 서비스를 이용하려면 회원 가입과 로그인은 필수입니다. 이메일, 사진 저장소, 쇼핑몰, SNS(소셜 …

1일 전
 51  1  0
IT기술공유
 3
Blog Image
첫 윈폼(WinForm) 만들기 - C# 프로그래밍
★ 윈폼(WinForms) 윈폼(WinForms)은 윈도우즈 폼(Windows Forms)의 단축어 이며, 윈도우즈 기반 사용자 인터페이스(UI, User Interface) 애플리케이션을 …

1일 전
 36  1  0
버들강아지
 0
웹크롤링
https://wikidocs.net/blog/@systrader79/1347/?fbclid=IwY2xjawH9hE5leHRuA2FlbQIxMAABHd5O4JGpsfrZZfG95uVfURqcLlAdtbkgXK9v8Z8l8WQVtihJmfyEQbNyAaem026eSz8BOIqGnUtFt9HugQ

1일 전
 24  0  0
리빗대학
 1
Blog Image
[정보공유] 가시다_AWS EKS Hands-on Study 모집
AWS, 클라우드 관련 해서 유명하신 가시다님이 스터디를 모집중인것같습니다. CKA 자격증을 준비중이라서, 저는 참여가 어렵지만 기회되시는분은 …

1일 전
 26  0  0
전뇌해커
 1
Blog Image
OpenAI 플레이그라운드에서 매개변수 조정하기
※ 도서 증정 이벤트 안내 위키독스의 “출판사와 함께하는 도서 증정 이벤트”를 통해 ⟪OpenAI, 구글 Gemini, …

2일 전
 46  0  0
systrader79의 트레이딩 이야기
 7
Blog Image
자연어로 코딩없이 웹크롤링 하는 방법 - firecrawl
AI가 정말 세상을 엄청나게 변화시키고 있습니다. 여러분도 웹 크롤링이라는 용어를 들어보신 적이 있으실 겁니다. 우리가 …

2일 전
 633  1  2
즐거운용
 0
Blog Image
Springboot 프로젝트 생성/설정
여기에 간편하게 프로젝트를 만들수 있도록 돕는 사이트가 있습니다. https://start.spring.io/ 여기서 보셔야 하는것들에 대한 설명입니다. Project …

2일 전
 34  0  0
TG_Miniapp
 0
ChatGPT 미니앱
챗GPT와 연계된 텔레그램 미니앱 이다. http://t.me/JugemuAIBot 테스크투언(챗GPT), 탭투언, 톤투언이 있고, 타임투언은 없다.

3일 전
 30  0  0

©2008 • 위키독스 • 개인정보취급방침 • 서비스약관사업자등록번호 : 724-14-01849 | 대표자명: 박응용 | 통신판매신고: 2022-경기과천-0278 | 문의: pahkey@gmail.com
"""
        
    if text:        
        paragraphs = text.split("\n\n")[:-1] if "\n\n" in text else text.split("\n") # 조금 더 디테일하게 수정하기 
    else:
        paragraphs = []    
        
    # FAISS 벡터 스토어 생성
    with st.spinner("벡터 스토어를 생성하는 중..."): 
        # convert to Document object (required for LangChain)
        documents = [Document(page_content=doc, metadata={"source": f"doc{idx+1}"}) for idx, doc in enumerate(paragraphs)]
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)    
        
        vectorstore = FAISS.from_documents(documents=splits, embedding=st.session_state.embedding)
        
    return vectorstore

# RAG using prompt
def rag_chatbot(question):
    context_docs = st.session_state.vectorstore.similarity_search(question, k=2)
    # for i, doc in enumerate(context_docs):
    #     st.write(f"{i+1}번째 문서: {doc.page_content}")
        
    context_docs = "\n\n".join([f"{i+1}번째 문서:\n{doc.page_content}" for i, doc in enumerate(context_docs)])

    # prompt = f"Context: {context_docs}\nQuestion: {question}\nAnswer in a complete sentence:"
    prompt = f"문맥: {context_docs}\n질문: {question}\n답변:" 
    # response = gemini_model(prompt)
    
    response = st.session_state.model.generate_content(prompt)
    answer = response.candidates[0].content.parts[0].text

    print("출처 문서:", context_docs)
    return answer, context_docs


# Streamlit 세션에서 모델을 한 번만 로드하도록 설정
# 1. gemini model 
if "model" not in st.session_state:
    st.session_state.model = load_model()

# 2. embedding model
if "embedding" not in st.session_state:
    st.session_state.embedding = load_embedding()

# 세션의 대화 히스토리 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "topic" not in st.session_state:
    st.session_state.topic = ""


# 1. 이 주제로 Vectorstore 만들 문서 가져오기
topic = st.text_input('찾을 문서의 주제를 입력하세요. 예시) 흑백요리사: 요리 계급 전쟁(시즌 1)')

if st.button('문서 가져오기'):
    if topic:
        vectorstore = create_vectorstore(topic)
        st.session_state.vectorstore = vectorstore    
        st.session_state.topic = topic
    else:
        st.warning('주제를 입력해라', icon="⚠️")
    
if st.session_state.topic and st.session_state.vectorstore:    
    st.write(f"주제: '{st.session_state.topic}' 로 Vectorstore 준비완료")
    
    
# 2. 사용자 질문에 유사한 내용을 Vectorstore에서 RAG 기반으로 답변
user_query = st.text_input('질문을 입력하세요.')

if st.button('질문하기') and user_query:
    # 사용자의 질문을 히스토리에 추가
    st.session_state.chat_history.append(f"[user]: {user_query}")
    st.text(f'[You]: {user_query}')    

    # response = st.session_state.model.generate_content(user_querie)
    # model_response = response.candidates[0].content.parts[0].text
        
    # 모델 응답 RAG
    if st.session_state.vectorstore:    
        response, context_docs = rag_chatbot(user_query)        
        st.text(f'[Chatbot]: {response}')
        st.text(f'출처 문서:\n')        
        st.write(context_docs)
    else: 
        response = "vector store is not ready."
        st.text(f'[Chatbot]: {response}')
    
    # 모델 응답을 히스토리에 추가
    st.session_state.chat_history.append(f"[chatbot]: {response}")
    
    # 전체 히스토리 출력
    st.text("Chat History")
    st.text('--------------------------------------------')
    st.text("\n".join(st.session_state.chat_history))
