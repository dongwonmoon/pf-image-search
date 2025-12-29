from airflow import DAG
from airflow.decorators import task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime, timedelta
import requests
import logging

import numpy as np
import onnxruntime as ort
from PIL import Image
from io import BytesIO

# 설정
SEARCH_TERM = "Sunflowers"
LIMIT = 20

# 전처리 표준 값 (ImageNet)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


def preprocess_image(image_bytes):
    """
    이미지 바이트 -> MobileNetV3 입력 형태 (1, 3, 224, 224) 변환
    """
    # 1. 이미지 로드 및 RGB 변환 (투명 배경 png 등 대응)
    img = Image.open(BytesIO(image_bytes)).convert("RGB")

    # 2. Resize: 짧은 변을 256으로 맞춤
    base_size = 256
    w, h = img.size
    if w < h:
        new_w = base_size
        new_h = int(h * (base_size / w))
    else:
        new_h = base_size
        new_w = int(w * (base_size / h))
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # 3. Center Crop: 중앙 224x224 자르기
    crop_size = 224
    left = (new_w - crop_size) / 2
    top = (new_h - crop_size) / 2
    right = (new_w + crop_size) / 2
    bottom = (new_h + crop_size) / 2
    img = img.crop((left, top, right, bottom))

    # 4. Normalize & Tensor 변환
    # 0~255 -> 0~1 (float32)
    img_data = np.array(img).astype(np.float32) / 255.0

    # (H, W, C) -> (H, W, C) - Mean / Std
    img_data = (img_data - MEAN) / STD

    # (H, W, C) -> (C, H, W) (Transpose)
    img_data = img_data.transpose(2, 0, 1)

    # (C, H, W) -> (1, C, H, W) (Batch 차원 추가)
    img_data = np.expand_dims(img_data, 0)

    return img_data


with DAG(
    dag_id="art_collection_dag",
    default_args=default_args,
    start_date=datetime(2023, 1, 1),
    schedule_interval="@once",
    catchup=False,
) as dag:

    @task
    def search_artworks_from_api():
        """
        1. The Met API에서 검색어로 작품 ID 목록을 가져옵니다.
        """
        url = (
            "https://collectionapi.metmuseum.org/public/collection/v1/search"
            f"?q={SEARCH_TERM}&hasImages=true"
        )
        response = requests.get(url)
        data = response.json()

        object_ids = data.get("objectIDs", [])[:LIMIT]
        logging.info(f"Found {len(object_ids)} artworks for '{SEARCH_TERM}'")

        return object_ids

    @task
    def fetch_and_store_metadata(object_ids):
        """
        2. 작품 ID를 순회하며 상세 정보를 가져와 DB에 저장합니다.
        """
        if not object_ids:
            logging.info("No object IDs received.")
            return

        pg_hook = PostgresHook(postgres_conn_id="postgres_default")
        conn = pg_hook.get_conn()
        cursor = conn.cursor()

        success_count = 0

        for obj_id in object_ids:
            try:
                detail_url = (
                    "https://collectionapi.metmuseum.org/public/collection/v1/objects/"
                    f"{obj_id}"
                )
                res = requests.get(detail_url)
                if res.status_code != 200:
                    continue

                art = res.json()
                image_url = art.get("primaryImage")

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
                        art.get("objectID"),
                        art.get("title", "Unknown"),
                        art.get("artistDisplayName", "Unknown"),
                        image_url,
                    ),
                )
                success_count += 1

            except Exception as e:
                logging.error(f"Error processing ID {obj_id}: {e}")

        conn.commit()
        cursor.close()
        conn.close()

        logging.info(f"Successfully stored {success_count} artworks.")

    @task
    def generate_embeddings():
        """
        3. DB에서 임베딩이 없는 데이터를 조회하여 벡터를 채워 넣습니다.
        """
        # 1. DB 연결 및 대상 조회
        pg_hook = PostgresHook(postgres_conn_id="postgres_default")
        conn = pg_hook.get_conn()
        cursor = conn.cursor()

        # 임베딩이 없는 데이터 조회
        cursor.execute(
            "SELECT id, image_url FROM artworks WHERE embedding IS NULL LIMIT 20;"
        )
        rows = cursor.fetchall()

        if not rows:
            logging.info("No images to process.")
            return

        # 2. ONNX 모델 로드 (경로는 실제 Docker 내부 경로 확인 필요)
        model_path = "/opt/airflow/plugins/models/mobilenet_v3_small.onnx"
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name

        success_count = 0

        for row_id, image_url in rows:
            try:
                # 3. 이미지 다운로드
                res = requests.get(image_url, timeout=10)
                if res.status_code != 200:
                    logging.warning(f"Failed to download {image_url}")
                    continue

                # 4. 전처리
                input_tensor = preprocess_image(res.content)

                # 5. ONNX 추론
                # result는 리스트 형태 [batch_output]
                result = session.run(None, {input_name: input_tensor})
                embedding_vector = result[0][
                    0
                ].tolist()  # (1, 576) -> (576,) list로 변환

                # 6. DB 업데이트 (pgvector는 리스트를 벡터로 인식함)
                update_sql = "UPDATE artworks SET embedding = %s WHERE id = %s;"
                cursor.execute(update_sql, (embedding_vector, row_id))
                success_count += 1

            except Exception as e:
                logging.error(f"Error processing image ID {row_id}: {e}")
                conn.rollback()  # 에러 발생 시 해당 트랜잭션 롤백 (선택 사항)
                continue

        conn.commit()
        cursor.close()
        conn.close()

        logging.info(f"Successfully updated embeddings for {success_count} images.")

    object_ids = search_artworks_from_api()
    fetch_and_store_metadata(object_ids)
