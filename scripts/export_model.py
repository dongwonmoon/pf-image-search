import torch
import torchvision.models as models
import os

# 1. 저장할 경로 설정
MODEL_DIR = "plugins/models"
MODEL_PATH = os.path.join(MODEL_DIR, "mobilenet_v3_small.onnx")

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

print("Downloading Pre-trained MobileNetV3 Small model...")
# 2. 사전 학습된 MobileNetV3 Small 모델 로드 (가중치 포함)
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
model.eval()  # 추론 모드로 변경

# 3. 더미 입력 데이터 생성 (Batch Size 1, 3 Channels, 224x224 Height/Width)
# 모델이 어떤 모양의 데이터를 받는지 알려주기 위함
dummy_input = torch.randn(1, 3, 224, 224)

print(f"Exporting model to {MODEL_PATH}...")

# 4. ONNX 포맷으로 변환 및 저장
torch.onnx.export(
    model,
    dummy_input,
    MODEL_PATH,
    export_params=True,  # 모델 가중치 저장
    opset_version=12,  # ONNX 버전 (호환성 위함)
    do_constant_folding=True,  # 상수 폴딩 최적화
    input_names=["input"],  # 입력 노드 이름
    output_names=["output"],  # 출력 노드 이름
    dynamic_axes={  # 배치 크기(Batch Size)를 가변적으로 설정
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
)

print("✅ Model converted to ONNX successfully!")
print(f"File size: {os.path.getsize(MODEL_PATH) / 1024 / 1024:.2f} MB")
