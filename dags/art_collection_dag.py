from airflow import DAG
from airflow.decorators import task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime, timedelta
import requests
import logging

# 설정
SEARCH_TERM = "Sunflowers"
LIMIT = 20


default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


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

    object_ids = search_artworks_from_api()
    fetch_and_store_metadata(object_ids)
