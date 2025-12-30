import streamlit as st
import requests
from PIL import Image
import io

# FastAPI 서버 주소 (Docker Compose 서비스 이름 사용)
# 로컬 테스트 시에는 localhost, 도커 내부에서는 서비스명(app) 사용
import os

API_URL = os.getenv("API_URL", "http://app:8000")

st.set_page_config(page_title="Image Search Engine", layout="wide")

st.title("🖼️ AI Image Search Engine")
st.markdown("Try uploading an image to find similar artworks from The Met Museum!")

# 1. 사이드바: 이미지 업로드
with st.sidebar:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # 업로드된 이미지 미리보기
        image = Image.open(uploaded_file)
        st.image(image, caption="Query Image", use_column_width=True)

# 2. 메인 화면: 검색 결과
if uploaded_file is not None:
    if st.button("🔍 Search Similar Images"):
        with st.spinner("Searching..."):
            try:
                # 파일 포인터를 리셋하고 FastAPI로 전송
                uploaded_file.seek(0)
                files = {
                    "file": (uploaded_file.name, uploaded_file, uploaded_file.type)
                }

                # FastAPI 호출
                response = requests.post(f"{API_URL}/search", files=files)

                if response.status_code == 200:
                    results = response.json()["results"]

                    st.success(f"Found {len(results)} similar images!")

                    # 결과를 5개씩 그리드로 보여주기 (지금은 5개 제한이므로 한 줄)
                    cols = st.columns(5)

                    for idx, res in enumerate(results):
                        with cols[idx]:
                            # 이미지 표시
                            st.image(res["image_url"], use_column_width=True)
                            # 메타데이터 표시
                            st.caption(f"**{res['title']}**")
                            st.text(f"Artist: {res['artist']}")
                            st.text(f"Sim: {res['similarity']:.4f}")

                else:
                    st.error(f"Error: {response.status_code} - {response.text}")

            except Exception as e:
                st.error(f"Connection Error: {e}")
                st.info("Make sure the backend API is running.")
