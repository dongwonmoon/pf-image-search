from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime, timedelta
import requests
import logging

# 설정
SEARCH_TERM = "Sunflowers"  # 검색어: 해바라기
LIMIT = 20  # 한 번 실행 시 수집할 최대 개수 (테스트용)


def search_artworks_from_api(**context):
    """
    1. The Met API에서 검색어로 작품 ID 목록을 가져옵니다.
    """
    url = f"https://collectionapi.metmuseum.org/public/collection/v1/search?q={SEARCH_TERM}&hasImages=true"
    response = requests.get(url)
    data = response.json()

    object_ids = data.get("objectIDs", [])[:LIMIT]  # 테스트를 위해 개수 제한
    logging.info(f"Found {len(object_ids)} artworks for '{SEARCH_TERM}'")

    # 다음 Task로 ID 목록 전달 (XCom)
    context["ti"].xcom_push(key="object_ids", value=object_ids)


def fetch_and_store_metadata(**context):
    """
    2. 작품 ID를 순회하며 상세 정보를 가져와 DB에 저장합니다.
    (이미지 URL이 있는 경우만)
    """
    object_ids = context["ti"].xcom_pull(key="object_ids", task_ids="search_task")
    pg_hook = PostgresHook(postgres_conn_id="postgres_default")
    conn = pg_hook.get_conn()
    cursor = conn.cursor()

    success_count = 0

    for obj_id in object_ids:
        try:
            # 상세 정보 API 호출
            detail_url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{obj_id}"
            res = requests.get(detail_url)
            if res.status_code != 200:
                continue

            art = res.json()
            image_url = art.get("primaryImage")

            # 이미지가 없으면 스킵
            if not image_url:
                continue

            # DB에 저장 (UPSERT: 이미 있으면 업데이트, 없으면 삽입)
            # external_id 충돌 시 아무것도 안 함 (ON CONFLICT DO NOTHING)
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


# DAG 정의
default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="art_collection_dag",
    default_args=default_args,
    start_date=datetime(2023, 1, 1),
    schedule_interval="@once",  # 한 번만 실행 (나중엔 @daily로 변경 가능)
    catchup=False,
) as dag:

    task1 = PythonOperator(
        task_id="search_task",
        python_callable=search_artworks_from_api,
        provide_context=True,
    )

    task2 = PythonOperator(
        task_id="store_task",
        python_callable=fetch_and_store_metadata,
        provide_context=True,
    )

    task1 >> task2
