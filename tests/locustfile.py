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
        bio = generate_dummy_image()
        files = {"file": ("sample.jpg", bio, "image/jpeg")}
        self.client.post("/search", files=files)
