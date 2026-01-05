from airflow import DAG
from airflow.decorators import task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime, timedelta
import os
import json
import time
import logging
import requests

PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY")

# ---- 수집 파라미터 ----
PER_PAGE = 200
PAGES_PER_QUERY = 2
SLEEP_SEC = 1.0
QUERIES_PER_RUN = 50

# ---- 일반 이미지 전반 용 쿼리 구성 ----
BASE_TERMS = [
    "nature",
    "city",
    "people",
    "animal",
    "food",
    "travel",
    "sky",
    "beach",
    "mountain",
    "car",
    "building",
    "flower",
    "technology",
    "business",
    "sports",
    "music",
    "art",
    "night",
    "sunset",
    "forest",
]

ORDERS = ["latest", "popular"]
ORIENTATIONS = ["horizontal", "vertical"]
MIN_WIDTHS = [0, 1000]  # 0(제약 없음) + 1000(고해상도 쪽으로 분기)

IMAGE_TYPE = "photo"
SAFESEARCH = "true"

default_args = {
    "owner": "airflow",
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
}


def build_query_key(q: str, order: str, orientation: str, min_width: int) -> str:
    return f"q={q}|order={order}|orientation={orientation}|min_width={min_width}|image_type={IMAGE_TYPE}|safesearch={SAFESEARCH}"


def generate_query_rows():
    """
    pixabay_query_state에 넣을 쿼리 세트 생성.
    20(q) * 2(order) * 2(orientation) * 2(min_width) = 160개 쿼리
    """
    rows = []
    for q in BASE_TERMS:
        for order in ORDERS:
            for orientation in ORIENTATIONS:
                for min_width in MIN_WIDTHS:
                    query_key = build_query_key(q, order, orientation, min_width)
                    params = {
                        "q": q,
                        "order": order,
                        "orientation": orientation,
                        "min_width": min_width,
                        "image_type": IMAGE_TYPE,
                        "safesearch": SAFESEARCH,
                    }
                    rows.append((query_key, q, json.dumps(params)))
    return rows


with DAG(
    dag_id="pixabay_ingest_metadata",
    default_args=default_args,
    start_date=datetime(2023, 1, 1),
    schedule_interval="0 */6 * * *",  # 6시간 마다
    catchup=False,
    dagrun_timeout=timedelta(hours=2),
    tags=["pixabay", "ingest"],
) as dag:

    @task
    def seed_query_state_if_empty():
        """pixabay_query_state가 비어있으면 쿼리 세트를 한번만 시딩"""
        pg_hook = PostgresHook(postgres_conn_id="postgres_default")
        conn = pg_hook.get_conn()
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM pixabay_query_state;")
        count = cur.fetchone()[0]
        if count > 0:
            logging.info(f"pixabay_query_state already seeded: {count} rows")
            cur.close()
            conn.close()
            return

        rows = generate_query_rows()
        logging.info(f"Seeding pixabay_query_state: {len(rows)} rows")

        cur.executemany(
            """
            INSERT INTO pixabay_query_state (query_key, q, params, next_page, done)
            VALUES (%s, %s, %s::jsonb, 1, false)
            ON CONFLICT (query_key) DO NOTHING;
            """,
            rows,
        )
        conn.commit()
        cur.close()
        conn.close()

    @task
    def ingest_pixabay_metadata():
        """
        done=false인 쿼리 중 일부를 가져와 next_page부터 2페이지 수집하고,
        artworks에 UPSERT(중복은 external_id로 방지).
        """
        if not PIXABAY_API_KEY:
            raise RuntimeError("PIXABAY_API_KEY is not set in environment")

        pg_hook = PostgresHook(postgres_conn_id="postgres_default")
        conn = pg_hook.get_conn()
        cur = conn.cursor()

        # 이번 런에서 처리할 쿼리들 선택
        cur.execute(
            """
            SELECT query_key, params, next_page
            FROM pixabay_query_state
            WHERE done = false
            ORDER BY last_run_at NULLS FIRST, created_at ASC
            LIMIT %s;
            """,
            (QUERIES_PER_RUN,),
        )
        query_rows = cur.fetchall()

        if not query_rows:
            logging.info("No active queries (done=false).")
            cur.close()
            conn.close()
            return

        session = requests.Session()
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; airflow-ingest/1.0)",
        }

        total_inserted = 0
        total_hits = 0

        for query_key, params_jsonb, next_page in query_rows:
            params = params_jsonb  # psycopg2가 jsonb를 dict로 주는 경우도 있고 str로 주는 경우도 있음
            if isinstance(params, str):
                params = json.loads(params)

            q = params.get("q")
            order = params.get("order")
            orientation = params.get("orientation")
            min_width = params.get("min_width")

            logging.info(f"[Query] {query_key} (next_page={next_page})")

            empty_at_first_page = False
            pages_fetched = 0

            for p in range(next_page, next_page + PAGES_PER_QUERY):
                api_params = {
                    "key": PIXABAY_API_KEY,
                    "q": q,
                    "image_type": params.get("image_type", "photo"),
                    "safesearch": params.get("safesearch", "true"),
                    "order": order,
                    "orientation": orientation,
                    "min_width": min_width,
                    "per_page": PER_PAGE,
                    "page": p,
                }

                try:
                    res = session.get(
                        "https://pixabay.com/api/",
                        params=api_params,
                        headers=headers,
                        timeout=15,
                    )
                    if res.status_code == 429:
                        logging.warning(
                            "Rate limited (429). Stop ingest early for safety."
                        )
                        break
                    if res.status_code != 200:
                        logging.error(
                            f"Pixabay API error: {res.status_code} - query={query_key} page={p}"
                        )
                        continue

                    data = res.json()
                    hits = data.get("hits", [])
                    total_hits += len(hits)

                    if not hits:
                        if p == next_page:
                            empty_at_first_page = True
                        break

                    saved = 0
                    for hit in hits:
                        ext_id = hit.get("id")
                        web_url = hit.get("webformatURL")
                        if not (ext_id and web_url):
                            continue

                        thumb = hit.get("previewURL")
                        page_url = hit.get("pageURL")
                        tags = hit.get("tags")
                        user = hit.get("user")
                        w = hit.get("imageWidth")
                        h = hit.get("imageHeight")

                        cur.execute(
                            """
                            INSERT INTO artworks (
                                external_id, title, artist, image_url, thumbnail_url,
                                source, page_url, tags, image_width, image_height, raw_json,
                                processing_status
                            )
                            VALUES (
                                %s, %s, %s, %s, %s,
                                'pixabay', %s, %s, %s, %s, %s::jsonb,
                                'PENDING'
                            )
                            ON CONFLICT (external_id) DO NOTHING;
                            """,
                            (
                                ext_id,
                                (tags or "Unknown")[
                                    :500
                                ],  # title 컬럼은 일단 tags를 유지(원하면 나중에 분리)
                                (user or "Unknown")[:500],
                                web_url,
                                thumb,
                                page_url,
                                tags,
                                w,
                                h,
                                json.dumps(hit),
                            ),
                        )
                        saved += 1

                    conn.commit()
                    total_inserted += saved
                    pages_fetched += 1
                    logging.info(f"  - page={p}, hits={len(hits)}, inserted={saved}")

                    time.sleep(SLEEP_SEC)

                except Exception as e:
                    logging.error(f"Error: query={query_key} page={p} err={e}")

            # 쿼리 상태 업데이트
            if empty_at_first_page:
                cur.execute(
                    """
                    UPDATE pixabay_query_state
                    SET done=true, last_run_at=NOW()
                    WHERE query_key=%s;
                    """,
                    (query_key,),
                )
                logging.info(f"  - Mark done: {query_key}")
            else:
                # 페이지를 일부라도 가져왔으면 next_page 진행
                if pages_fetched > 0:
                    cur.execute(
                        """
                        UPDATE pixabay_query_state
                        SET next_page = next_page + %s,
                            last_run_at = NOW()
                        WHERE query_key=%s;
                        """,
                        (pages_fetched, query_key),
                    )
            conn.commit()

        cur.close()
        conn.close()

        logging.info(f"Done. total_hits={total_hits}, total_inserted={total_inserted}")

    seed_query_state_if_empty() >> ingest_pixabay_metadata()
