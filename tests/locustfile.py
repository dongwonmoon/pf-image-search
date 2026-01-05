from locust import HttpUser, task, between
import os


class ImageSearchUser(HttpUser):
    # 실제 사용자처럼 행동 (1~3초 대기)
    wait_time = between(1, 3)

    def on_start(self):
        """테스트 시작 전 이미지를 메모리에 로드 (Disk I/O 병목 제거)"""
        # 현재 스크립트 위치 기준
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, "sample.jpg")

        try:
            with open(image_path, "rb") as f:
                self.image_data = f.read()
            print(f"Loaded image: {len(self.image_data)} bytes")
        except FileNotFoundError:
            print(f"Error: {image_path} not found! Please add a real jpg file.")
            self.image_data = None

    @task
    def search_image(self):
        if self.image_data:
            # 매 요청마다 메모리에 있는 바이너리 데이터 전송
            files = {"file": ("test.jpg", self.image_data, "image/jpeg")}

            # 타임아웃 5초 설정 (5초 넘으면 실패로 간주)
            with self.client.post(
                "/search", files=files, catch_response=True, timeout=5
            ) as response:
                if response.status_code != 200:
                    response.failure(f"Status code: {response.status_code}")
