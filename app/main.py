from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import psycopg2
import os
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 전역 변수로 모델 세션 저장
model_session = None

# 환경 변수 (Docker Compose에서 주입 예정)
DB_HOST = os.getenv("DB_HOST", "postgres")
DB_USER = os.getenv("DB_USER", "admin")
DB_PASS = os.getenv("DB_PASS", "password")
DB_NAME = os.getenv("DB_NAME", "im_search")
MODEL_PATH = "/app/models/mobilenet_v3_small.onnx"  # 컨테이너 내부 경로

# --- 전처리 함수 ---
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Resize (Shortest side 256)
    base_size = 256
    w, h = img.size
    if w < h:
        new_w = base_size
        new_h = int(h * (base_size / w))
    else:
        new_h = base_size
        new_w = int(w * (base_size / h))
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # Center Crop (224x224)
    crop_size = 224
    left = (new_w - crop_size) / 2
    top = (new_h - crop_size) / 2
    right = (new_w + crop_size) / 2
    bottom = (new_h + crop_size) / 2
    img = img.crop((left, top, right, bottom))

    # Normalize & Tensor
    img_data = np.array(img).astype(np.float32) / 255.0
    img_data = (img_data - MEAN) / STD
    img_data = img_data.transpose(2, 0, 1)
    img_data = np.expand_dims(img_data, 0)

    return img_data


# --- Lifespan: 서버 시작/종료 시 실행 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시: 모델 로드
    global model_session
    try:
        logger.info(f"Loading model from {MODEL_PATH}...")
        model_session = ort.InferenceSession(MODEL_PATH)
        logger.info("✅ Model loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")

    yield

    logger.info("Shutting down API server...")


# 앱 초기화
app = FastAPI(title="Image Search API", lifespan=lifespan)


# --- 엔드포인트: 헬스 체크 ---
@app.get("/")
def health_check():
    return {"status": "ok", "service": "image-search-api"}


# --- 엔드포인트: 이미지 검색 ---
@app.post("/search")
async def search_image(file: UploadFile = File(...)):
    if not model_session:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # 1. 이미지 읽기 및 전처리
    try:
        contents = await file.read()
        input_tensor = preprocess_image(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # 2. 모델 추론 (Inference)
    try:
        input_name = model_session.get_inputs()[0].name
        # ONNX Runtime은 동기 함수이므로, 대량 트래픽 시 비동기 처리(run_in_executor) 고려 필요
        # 현재는 간단하게 직접 호출
        result = model_session.run(None, {input_name: input_tensor})
        embedding = result[0][0].tolist()  # (576,)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # 3. DB 검색 (Cosine Similarity)
    # pgvector의 '<->' 연산자는 Euclidean Distance, '<=>'는 Cosine Distance
    # 보통 추천/검색에는 Cosine Distance('<=>')를 많이 씀. (0에 가까울수록 유사)
    results = []
    try:
        conn = psycopg2.connect(
            host=DB_HOST, user=DB_USER, password=DB_PASS, dbname=DB_NAME
        )
        cursor = conn.cursor()

        # 가장 유사한 5개 조회 (Cosine Distance 기준 오름차순)
        sql = """
            SELECT id, title, artist, image_url, (embedding <=> %s::vector) as distance
            FROM artworks
            WHERE embedding IS NOT NULL
            ORDER BY distance ASC
            LIMIT 5;
        """
        cursor.execute(sql, (embedding,))
        rows = cursor.fetchall()

        for row in rows:
            results.append(
                {
                    "id": row[0],
                    "title": row[1],
                    "artist": row[2],
                    "image_url": row[3],
                    "distance": float(row[4]),
                    "similarity": 1 - float(row[4]),  # 유사도(0~1)
                }
            )

        cursor.close()
        conn.close()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    return {"results": results}
