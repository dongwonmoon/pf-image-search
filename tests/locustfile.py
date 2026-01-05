from locust import HttpUser, task, between
import io
import base64

# 1x1 픽셀짜리 빨간색 점(이미지)의 Base64 문자열
# 이 문자열은 실제 PNG 파일의 바이너리 데이터입니다.
TINY_IMAGE_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pZnAAAAAElFTkSuQmCC"


def generate_dummy_image():
    # Base64 문자열을 디코딩해서 실제 바이너리(bytes)로 변환
    image_bytes = base64.b64decode(TINY_IMAGE_BASE64)
    return io.BytesIO(image_bytes)


class ImageSearchUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def search_image(self):
        # 1. 유효한 1x1 이미지 생성
        image_file = generate_dummy_image()

        # 2. FastAPI로 전송 (파일명은 .png로 설정)
        files = {"file": ("dummy.png", image_file, "image/png")}

        # 3. 요청 전송
        # catch_response=True를 쓰면 에러 발생 시 Locust UI에 빨간색으로 표시됨
        with self.client.post("/search", files=files, catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Got status {response.status_code}")
