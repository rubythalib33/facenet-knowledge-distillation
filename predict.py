import torch
import torch.nn.functional as F
from torchvision import transforms
from facenet_pytorch import MTCNN
from PIL import Image
import models
from config import MODEL

def load_and_crop_image(image_path, detector, device):
    image = Image.open(image_path)
    # Use MTCNN to find faces in the image and get the first detected face
    boxes, _ = detector.detect(image)
    if boxes is not None:
        # Assuming the first box is the target face
        box = boxes[0]
        # Convert box from numpy array to a list with correct order
        box = [int(b) for b in box.tolist()]
        # Crop the image to the face
        cropped_image = image.crop(box)
        # Convert the PIL Image to a tensor
        transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
        ])
        cropped_image = transform(cropped_image).unsqueeze(0).to(device)
        return cropped_image
    else:
        print(f"No face detected in {image_path}.")
        return None

def cosine_similarity(embedding1, embedding2):
    return F.cosine_similarity(embedding1, embedding2)

def predict(image_path1, image_path2, student_model):
    device = "cpu"
    
    # Initialize MTCNN
    mtcnn = MTCNN(keep_all=False, device=device)
    
    # Load and process images
    image1 = load_and_crop_image(image_path1, mtcnn, device)
    image2 = load_and_crop_image(image_path2, mtcnn, device)
    
    # Ensure both images were successfully processed
    if image1 is not None and image2 is not None:
        # Get embeddings from the student model
        embedding1 = student_model(image1)
        embedding2 = student_model(image2)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(embedding1, embedding2)
        print(f"Cosine similarity: {similarity.item()}")
    else:
        print("One or both images could not be processed.")

if __name__ == "__main__":
    # Load the student model
    student_model = models.get_model(MODEL).eval()
    device = "cpu"
    student_model.load_state_dict(torch.load('ckpt/mobilenet_v2_adam.pt'))
    student_model.to(device)
    
    # Example usage
    # image_path1 = 'sample/elon1.webp'
    image_path1 = 'sample/jeff2.jpg'
    image_path2 = 'sample/elon2.webp'
    image_path2 = 'sample/jeff1.jpeg'
    predict(image_path1, image_path2, student_model)
