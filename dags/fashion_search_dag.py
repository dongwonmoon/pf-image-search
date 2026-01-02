from airflow import DAG
from airflow.decorators import task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime, timedelta
import requests
import logging
import time
import numpy as np
import onnxruntime as ort
from PIL import Image
from io import BytesIO
import os

PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY")

# 패션 관련 검색어 (다양한 스타일 확보)
SEARCH_TERMS = [
    "fashion",
    "street style",
    "model",
    "dress",
    "coat",
    "jacket",
    "sneakers",
    "high heels",
    "handbag",
    "jewelry",
    "portrait woman",
    "men fashion",
    "runway",
    "clothing",
    "apparel",
    "denim",
    "suit",
]

# 키워드 당 수집할 페이지 수 (1페이지당 200개)
# 17개 키워드 * 2페이지 * 200개 = 약 6,800장 (중복 제외 시 5,000장 예상)
PAGES_PER_TERM = 2

# 전처리 표준값
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(image_bytes):
    """이미지 전처리 (FastAPI와 동일 로직)"""
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")

        base_size = 256
        w, h = img.size
        if w < h:
            new_w = base_size
            new_h = int(h * (base_size / w))
        else:
            new_h = base_size
            new_w = int(w * (base_size / h))
        img = img.resize((new_w, new_h), Image.BILINEAR)

        crop_size = 224
        left = (new_w - crop_size) / 2
        top = (new_h - crop_size) / 2
        right = (new_w + crop_size) / 2
        bottom = (new_h + crop_size) / 2
        img = img.crop((left, top, right, bottom))

        img_data = np.array(img).astype(np.float32) / 255.0
        img_data = (img_data - MEAN) / STD
        img_data = img_data.transpose(2, 0, 1)
        img_data = np.expand_dims(img_data, 0)
        return img_data
    except Exception as e:
        return None


default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="fashion_search_dag",  # DAG ID 변경
    default_args=default_args,
    start_date=datetime(2023, 1, 1),
    schedule_interval="@once",
    catchup=False,
    dagrun_timeout=timedelta(hours=2),
) as dag:

    @task
    def fetch_fashion_data():
        """
        Pixabay API에서 패션 이미지를 검색하고 메타데이터를 저장
        """
        pg_hook = PostgresHook(postgres_conn_id="postgres_default")
        conn = pg_hook.get_conn()
        cursor = conn.cursor()

        total_stored = 0

        for term in SEARCH_TERMS:
            logging.info(f"Searching for: {term}")

            for page in range(1, PAGES_PER_TERM + 1):
                try:
                    url = f"https://pixabay.com/api/?key={PIXABAY_API_KEY}&q={term}&image_type=photo&per_page=200&page={page}"
                    res = requests.get(url, timeout=10)

                    if res.status_code != 200:
                        logging.error(f"Failed {term} page {page}: {res.status_code}")
                        continue

                    data = res.json()
                    hits = data.get("hits", [])

                    if not hits:
                        break  # 결과 없으면 다음 키워드로

                    saved_count = 0
                    for hit in hits:
                        # Pixabay 필드 매핑
                        # external_id -> id
                        # title -> tags (제목이 없어서 태그를 제목처럼 사용)
                        # artist -> user
                        # image_url -> webformatURL (중간 크기, 전송 빠름)

                        image_url = hit.get("webformatURL")
                        if not image_url:
                            continue

                        sql = """
                            INSERT INTO artworks (external_id, title, artist, image_url)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (external_id) DO NOTHING;
                        """
                        cursor.execute(
                            sql,
                            (
                                hit.get("id"),
                                hit.get("tags", "Unknown")[:500],
                                hit.get("user", "Unknown")[:500],
                                image_url,
                            ),
                        )
                        saved_count += 1

                    conn.commit()
                    total_stored += saved_count
                    logging.info(f"  - Page {page}: Stored {saved_count} images.")

                    # API 예절 (Rate Limit 방지)
                    time.sleep(1.0)

                except Exception as e:
                    logging.error(f"Error processing {term} page {page}: {e}")

        cursor.close()
        conn.close()
        logging.info(f"=== Total Metadata Stored: {total_stored} ===")

    @task
    def generate_embeddings():
        """
        저장된 이미지 중 임베딩이 없는 것을 찾아 벡터화
        """
        pg_hook = PostgresHook(postgres_conn_id="postgres_default")
        conn = pg_hook.get_conn()
        cursor = conn.cursor()

        # 임베딩 없는 것 조회
        cursor.execute("SELECT id, image_url FROM artworks WHERE embedding IS NULL;")
        rows = cursor.fetchall()
        total_rows = len(rows)

        if not rows:
            logging.info("No embeddings to generate.")
            return

        logging.info(f"Start generating embeddings for {total_rows} images...")

        # 헤더 추가 (Pixabay 이미지 서버 차단 방지용)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        model_path = "/opt/airflow/plugins/models/mobilenet_v3_small.onnx"
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name

        success_count = 0

        for i, (row_id, image_url) in enumerate(rows):
            if i % 50 == 0:
                logging.info(f"Embedding progress: {i}/{total_rows}")
                conn.commit()

            try:
                # 타임아웃 넉넉하게
                res = requests.get(image_url, headers=headers, timeout=10)
                if res.status_code != 200:
                    continue

                input_tensor = preprocess_image(res.content)
                if input_tensor is None:
                    continue

                result = session.run(None, {input_name: input_tensor})
                embedding_vector = result[0][0].tolist()

                cursor.execute(
                    "UPDATE artworks SET embedding = %s WHERE id = %s;",
                    (embedding_vector, row_id),
                )
                success_count += 1

            except Exception as e:
                # 너무 많은 에러 로그 방지
                if i % 100 == 0:
                    logging.error(f"Error ID {row_id}: {e}")
                continue

        conn.commit()
        cursor.close()
        conn.close()
        logging.info(f"Finished. Generated {success_count} embeddings.")

    # Flow
    fetch_fashion_data() >> generate_embeddings()
