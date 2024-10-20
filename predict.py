import torch
import torch.nn.functional as F
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import models
from config import MODEL

# Function to load and crop an image
def load_and_crop_image(image_path, detector, device):
    image = Image.open(image_path)
    boxes, _ = detector.detect(image)
    if boxes is not None:
        box = boxes[0]
        box = [int(b) for b in box.tolist()]
        cropped_image = image.crop(box)
        transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        cropped_image = transform(cropped_image).unsqueeze(0).to(device)
        return cropped_image
    else:
        print(f"No face detected in {image_path}.")
        return None

# Cosine similarity function
def cosine_similarity(embedding1, embedding2):
    return F.cosine_similarity(embedding1, embedding2, dim=1)

# Predict function that calculates cosine similarity with both models
def predict(image_path1, image_path2, student_model, pretrained_model, device):
    # Initialize MTCNN for face detection
    mtcnn = MTCNN(keep_all=False, device=device)
    
    # Load and process images
    image1 = load_and_crop_image(image_path1, mtcnn, device)
    image2 = load_and_crop_image(image_path2, mtcnn, device)
    
    if image1 is not None and image2 is not None:
        # Student model embeddings
        embedding1_student = student_model(image1)
        embedding2_student = student_model(image2)
        similarity_student = cosine_similarity(embedding1_student, embedding2_student)
        print(f"Student Model Cosine Similarity: {similarity_student.item()}")

        # Pretrained facenet model embeddings
        embedding1_pretrained = pretrained_model(image1)
        embedding2_pretrained = pretrained_model(image2)
        similarity_pretrained = cosine_similarity(embedding1_pretrained, embedding2_pretrained)
        print(f"Facenet-Pytorch Model Cosine Similarity: {similarity_pretrained.item()}")
    else:
        print("One or both images could not be processed.")

if __name__ == "__main__":
    device = "cpu"

    # Load student model
    student_model = models.get_model(MODEL).eval()
    student_model.load_state_dict(torch.load('ckpt/mobilenet_v2_adam.pt', map_location=device))
    student_model = student_model.to(device)

    # Load pretrained facenet model
    pretrained_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # Example usage
    # image_path1 = 'sample/jeff2.jpg'
    image_path1 = 'sample/elon1.webp'
    image_path2 = 'sample/jeff1.jpeg'
    # image_path2 = 'sample/elon2.webp'
    predict(image_path1, image_path2, student_model, pretrained_model, device)
