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

# --- 설정 변경 ---
# 1. 다양한 카테고리의 명화를 수집하기 위한 키워드 리스트
SEARCH_TERMS = [
    "Sunflowers",
    "Portrait",
    "Landscape",
    "Still Life",
    "Impressionism",
    "Vincent van Gogh",
    "Monet",
    "Oil Painting",
]
# 2. 각 키워드당 수집할 최대 개수 (8개 키워드 * 500개 = 약 4,000장 목표)
LIMIT_PER_TERM = 500

# 전처리 표준값
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(image_bytes):
    """이미지 전처리 (FastAPI와 로직 동일)"""
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
        logging.error(f"Preprocessing failed: {e}")
        return None


default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="art_collection_dag",
    default_args=default_args,
    start_date=datetime(2023, 1, 1),
    schedule_interval="@once",  # 대규모 수집이므로 수동 실행 권장
    catchup=False,
    # 태스크가 오래 걸려도 안 죽게 타임아웃 설정 (2시간)
    dagrun_timeout=timedelta(hours=2),
) as dag:

    @task
    def search_artworks_from_api():
        """
        여러 키워드로 검색하여 objectID 리스트를 수집 (중복 제거)
        """
        all_object_ids = set()  # 중복 방지용 Set

        for term in SEARCH_TERMS:
            try:
                url = (
                    "https://collectionapi.metmuseum.org/public/collection/v1/search"
                    f"?q={term}&hasImages=true&medium=Paintings"
                )
                response = requests.get(url, timeout=10)
                data = response.json()

                ids = data.get("objectIDs", [])
                if ids:
                    # 키워드 당 제한 개수만큼만 가져오기
                    selected_ids = ids[:LIMIT_PER_TERM]
                    all_object_ids.update(selected_ids)
                    logging.info(f"Term '{term}': Found {len(selected_ids)} items.")
            except Exception as e:
                logging.error(f"Failed search for '{term}': {e}")

        final_list = list(all_object_ids)
        logging.info(f"Total unique artworks to process: {len(final_list)}")
        return final_list

    @task
    def fetch_and_store_metadata(object_ids):
        if not object_ids:
            return

        pg_hook = PostgresHook(postgres_conn_id="postgres_default")
        conn = pg_hook.get_conn()
        cursor = conn.cursor()

        success_count = 0

        # 성능을 위해 이미 DB에 있는 ID는 건너뛰는 로직 추가 가능하지만,
        # 지금은 ON CONFLICT DO NOTHING 믿고 진행
        for i, obj_id in enumerate(object_ids):
            # 로그 너무 많이 남지 않게 100개마다 찍기
            if i % 100 == 0:
                logging.info(f"Fetching metadata progress: {i}/{len(object_ids)}")

            try:
                detail_url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{obj_id}"
                res = requests.get(detail_url, timeout=5)
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
                        art.get("title", "Unknown")[:500],  # 제목 너무 길면 자름
                        art.get("artistDisplayName", "Unknown")[:500],
                        image_url,
                    ),
                )
                success_count += 1
            except Exception as e:
                logging.error(f"Error fetching ID {obj_id}: {e}")

        conn.commit()
        cursor.close()
        conn.close()
        logging.info(f"Metadata stored count: {success_count}")

    @task
    def generate_embeddings():
        """
        DB에서 임베딩이 NULL인 항목들을 조회하여 벡터화 (배치 처리 느낌으로 전체 조회)
        """
        pg_hook = PostgresHook(postgres_conn_id="postgres_default")
        conn = pg_hook.get_conn()
        cursor = conn.cursor()

        # 처리되지 않은 모든 이미지 조회
        cursor.execute("SELECT id, image_url FROM artworks WHERE embedding IS NULL;")
        rows = cursor.fetchall()
        total_rows = len(rows)

        if not rows:
            logging.info("No embeddings to generate.")
            return

        logging.info(f"Start generating embeddings for {total_rows} images...")

        model_path = "/opt/airflow/plugins/models/mobilenet_v3_small.onnx"
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name

        success_count = 0

        for i, (row_id, image_url) in enumerate(rows):
            if i % 50 == 0:
                logging.info(f"Embedding progress: {i}/{total_rows}")
                conn.commit()  # 중간중간 커밋해서 에러 나도 일부는 저장되게 함

            try:
                res = requests.get(image_url, timeout=10)
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
                logging.error(f"Embedding error ID {row_id}: {e}")
                continue

        conn.commit()
        cursor.close()
        conn.close()
        logging.info(f"Finished. Successfully generated {success_count} embeddings.")

    # Flow
    ids = search_artworks_from_api()
    stored = fetch_and_store_metadata(ids)
    stored >> generate_embeddings()
