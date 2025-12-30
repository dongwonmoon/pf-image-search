FROM python:3.9-slim

WORKDIR /app

RUN pip install streamlit requests Pillow

COPY . .

# Streamlit 포트 개방
EXPOSE 8501

# 실행 (API_URL은 docker-compose에서 주입)
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]