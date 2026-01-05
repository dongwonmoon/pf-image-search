from locust import HttpUser, task, between
import io
import random
import os


# 가짜 이미지 생성 함수 (매번 파일을 읽지 않고 메모리에서 생성)
def generate_dummy_image():
    # 100바이트짜리 랜덤 더미 데이터 (포맷은 대충 맞춤)
    return io.BytesIO(b"\xff\xd8\xff" + b"\x00" * 1000)


class ImageSearchUser(HttpUser):
    # 사용자 한 명이 요청을 보내고 다음 요청까지 대기하는 시간 (1~3초)
    wait_time = between(1, 3)

    @task
    def search_image(self):
        # 1. 더미 이미지 준비
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        image_path = os.path.join(current_dir, "sample.jpg")
        
        try:
            with open(image_path, "rb") as image_file:
                files = {"file": image_file}
                # FastAPI 컨테이너로 POST 요청 전송
                self.client.post("/search", files=files)
        except FileNotFoundError:
            print("Error: sample.jpg not found!")
