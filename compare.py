import torch
from facenet_pytorch import InceptionResnetV1
from torchvision.models import efficientnet_b1, mobilenet_v3_large, mobilenet_v2
import time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_inference_time(model, device, input_size=(3, 224, 224), iterations=100):
    model.eval()
    model.to(device)
    input_tensor = torch.rand(1, *input_size, device=device)

    # Warm-up
    for _ in range(10):
        _ = model(input_tensor)

    # Measure inference time
    start_time = time.time()
    for _ in range(iterations):
        _ = model(input_tensor)
    end_time = time.time()

    # Calculate average inference time
    avg_inference_time = (end_time - start_time) / iterations
    return avg_inference_time

def compare_models():
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')

    models = {
        'InceptionResnetV1': InceptionResnetV1(pretrained='vggface2'),
        'EfficientNetB1': efficientnet_b1(pretrained=True),
        'MobileNetV3-Large': mobilenet_v3_large(pretrained=True),
        'MobileNetV2': mobilenet_v2(pretrained=True),
    }

    for device in devices:
        print(f"\nDevice: {device.upper()}")
        for model_name, model in models.items():
            params = count_parameters(model)
            inference_time = measure_inference_time(model, device=device)
            print(f"{model_name}: Parameters = {params}, Inference Time = {inference_time:.5f} seconds")

if __name__ == "__main__":
    compare_models()
