from airflow import DAG
from airflow.decorators import task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime, timedelta
import logging
import numpy as np
import onnxruntime as ort
from PIL import Image
from io import BytesIO
import asyncio
import aiohttp

# ---- 설정 ----
BATCH_SIZE = 50  # 한 번에 처리할 이미지 수
MODEL_PATH = "/opt/airflow/plugins/models/mobilenet_v3_small.onnx"

# 전처리 상수
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(image_bytes):
    """이미지 바이트 -> ONNX 입력 텐서 변환"""
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Resize & Center Crop (256 -> 224)
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

        # Normalize
        img_data = np.array(img).astype(np.float32) / 255.0
        img_data = (img_data - MEAN) / STD
        img_data = img_data.transpose(2, 0, 1)  # HWC -> CHW
        img_data = np.expand_dims(img_data, 0)  # Add batch dim
        return img_data
    except Exception:
        return None


async def download_images_async(urls):
    """aiohttp를 사용하여 이미지 병렬 다운로드"""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in urls:
            tasks.append(fetch_url(session, url))
        return await asyncio.gather(*tasks, return_exceptions=True)


async def fetch_url(session, url):
    """개별 URL 다운로드 (타임아웃 10초)"""
    headers = {"User-Agent": "Mozilla/5.0 (Airflow Consumer)"}
    async with session.get(url, headers=headers, timeout=10) as response:
        response.raise_for_status()
        return await response.read()


default_args = {
    "owner": "airflow",
    "retries": 0,  # 배치는 다음 턴에 재시도하면 되므로 즉시 실패 처리
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="consumer_embedding_async",
    default_args=default_args,
    start_date=datetime(2023, 1, 1),
    schedule_interval="*/2 * * * *",  # 2분마다 실행 (Producer 속도에 맞춰 더 자주 실행)
    catchup=False,
    max_active_runs=1,  # 중복 실행 방지
    tags=["consumer", "async", "inference"],
) as dag:

    @task
    def process_embeddings_async():
        # 1. DB에서 할 일(PENDING) 가져오기
        pg_hook = PostgresHook(postgres_conn_id="postgres_default")
        src_conn = pg_hook.get_conn()
        src_cursor = src_conn.cursor()

        # PENDING 상태인 것 중 오래된 순으로 가져옴 (FIFO)
        src_cursor.execute(
            f"""
            SELECT id, image_url 
            FROM artworks 
            WHERE processing_status = 'PENDING'
            ORDER BY id ASC
            LIMIT {BATCH_SIZE}
            FOR UPDATE SKIP LOCKED; 
        """
        )
        # FOR UPDATE SKIP LOCKED: 동시 실행 시 중복 작업 방지 (멀티 워커 확장 대비)

        rows = src_cursor.fetchall()

        if not rows:
            logging.info("No pending images. Sleeping.")
            src_cursor.close()
            src_conn.close()
            return

        logging.info(f"Fetched {len(rows)} images to process.")

        ids = [r[0] for r in rows]
        urls = [r[1] for r in rows]

        # DB 커넥션 잠시 반환 (다운로드 동안 DB 잡고 있지 않기 위함)
        src_conn.commit()  # LOCK 해제 (SKIP LOCKED를 썼으므로 이미 내 것임) or 상태를 'PROCESSING'으로 바꾸고 커밋해도 됨

        # 2. AsyncIO로 이미지 병렬 다운로드
        # Airflow Task는 기본적으로 동기식이므로 asyncio.run()으로 실행
        logging.info("Starting async download...")
        start_time = datetime.now()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        downloaded_data = loop.run_until_complete(download_images_async(urls))
        loop.close()

        download_duration = (datetime.now() - start_time).total_seconds()
        logging.info(f"Downloaded {len(rows)} images in {download_duration:.2f}s")

        # 3. 모델 로드 (ONNX)
        session = ort.InferenceSession(MODEL_PATH)
        input_name = session.get_inputs()[0].name

        # 4. 결과 처리 (Loop)
        updates_success = []
        updates_failed = []

        for i, data in enumerate(downloaded_data):
            row_id = ids[i]

            # 다운로드 실패 체크
            if isinstance(data, Exception):
                logging.warning(f"Download failed for ID {row_id}: {data}")
                updates_failed.append((row_id,))
                continue

            # 전처리 & 추론
            try:
                input_tensor = preprocess_image(data)
                if input_tensor is None:
                    raise ValueError("Preprocessing returned None (Invalid Image)")

                result = session.run(None, {input_name: input_tensor})
                vector = result[0][0].tolist()

                updates_success.append((vector, row_id))

            except Exception as e:
                logging.error(f"Inference failed for ID {row_id}: {e}")
                updates_failed.append((row_id,))

        # 5. DB 업데이트 (Batch Update)
        try:
            if updates_success:
                src_cursor.executemany(
                    """
                    UPDATE artworks 
                    SET embedding = %s::vector, processing_status = 'COMPLETED' 
                    WHERE id = %s
                """,
                    updates_success,
                )

            if updates_failed:
                src_cursor.executemany(
                    """
                    UPDATE artworks 
                    SET processing_status = 'FAILED' 
                    WHERE id = %s
                """,
                    updates_failed,
                )

            src_conn.commit()
            logging.info(
                f"Batch Result: Success={len(updates_success)}, Failed={len(updates_failed)}"
            )

        except Exception as e:
            logging.error(f"DB Update Failed: {e}")
            src_conn.rollback()
            raise e
        finally:
            src_cursor.close()
            src_conn.close()

    process_embeddings_async()
