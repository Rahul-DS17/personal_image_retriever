import os
import faiss
import torch
from PIL import Image
import numpy
from transformers import AutoProcessor, AutoModel
import subprocess

# Define the paths
clip_repo_path = "qai_hub_models/models/openai_clip"
mediapipe_repo_path = "qai_hub_models/models/mediapipe_face"

# Function to clone repositories if they don’t exist
def clone_repo(repo_url, target_path):
    if not os.path.exists(target_path):
        print(f"Cloning {repo_url} into {target_path}...")
        
        # Automatically respond "yes" to any prompts
        process = subprocess.run(
            f"Y | git clone --quiet {repo_url} {target_path}",
            shell=True,
            check=True
        )
    else:
        print(f"Repository already exists: {target_path}")

# Clone necessary repositories
clone_repo("https://github.com/openai/CLIP.git", clip_repo_path)
clone_repo("https://github.com/google/mediapipe.git", mediapipe_repo_path)

from qai_hub_models.models.openai_clip.model import Clip
from qai_hub_models.models.mediapipe_face.model import MediaPipeFace
from qai_hub_models.models.mediapipe_face.app import MediaPipeFaceApp
from torchvision import transforms
import pickle

def load_images(folder):
    """
    Loads image file paths from a given folder.
    Returns a list of image paths for supported formats (.jpg, .jpeg, .png, .JPG).
    """
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.JPG')):
            image_path = os.path.join(folder, filename)
            images.append(image_path)
    return images

def load_model(device, model_choice):
    """
    Loads a specified model (CLIP or JINA) onto the given device (CPU/GPU).
    Returns the processor and model.
    """
    if model_choice == "CLIP":
        processor = None  # No separate processor needed for CLIP
        model = Clip.from_pretrained()  # Load pre-trained CLIP model
    elif model_choice == "JINA":
        processor = AutoProcessor.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True)
        model = AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True).to(device)
    else:
        print("Please select a valid model")
        return None
    return processor, model

def transform_image(device, image):
    """
    Applies transformations to an image: resizing, converting to tensor, and normalizing.
    Returns a processed image tensor.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ToTensor(),          # Convert to tensor
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])  # Normalize
    ])
    return transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

def get_image_embedding(image_path, processor, model, device, model_choice):
    """
    Extracts an image embedding using the specified processor and model.
    Returns a normalized embedding tensor.
    """
    image = Image.open(image_path)
    if model_choice == "CLIP":
        image = transform_image(device, image)  # Apply transformations
        with torch.no_grad():
            outputs = model.image_encoder.to(device)(image)  # Get image features
    else:
        image = image.convert("RGB").resize((224, 224))  # Convert image to RGB and resize
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)  # Get image features
    return torch.nn.functional.normalize(outputs, p=2, dim=1)  # L2 normalize the embeddings

def get_text_embedding(text_query, processor, model, device, model_choice):
    """
    Extracts a text embedding using the specified processor and model.
    Returns a normalized embedding tensor.
    """
    if model_choice == "CLIP":
        inputs = model.tokenizer_func(text_query).to(device)  # Tokenize text
        with torch.no_grad():
            outputs = model.text_encoder.to(device)(inputs)  # Get text features
    else:
        inputs = processor(text=text_query, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.get_text_features(**inputs)  # Get text features
    return torch.nn.functional.normalize(outputs, p=2, dim=1)  # L2 normalize the embeddings

def create_faiss_index(embeddings):
    """
    Creates a FAISS index from a set of embeddings.
    Returns a FAISS index for similarity search.
    """
    embeddings_np = embeddings.numpy().astype('float32')  # Convert to NumPy array
    dimension = embeddings_np.shape[1]  # Get embedding dimension
    faiss_index = faiss.IndexFlatIP(dimension)  # Initialize FAISS index for inner product search
    faiss_index.add(embeddings_np)  # Add embeddings to the index
    return faiss_index

def load_fr_model(device):
    """
    Loads a face recognition model (MediaPipe for face detection and CLIP for feature extraction).
    Returns the MediaPipe face model and CLIP image encoder.
    """
    model = MediaPipeFace.from_pretrained()  # Load MediaPipe face detection model
    mediapipe_app = MediaPipeFaceApp(model=model)  # Initialize MediaPipe application
    clip_encoder = Clip.from_pretrained().image_encoder.to(device)  # Load CLIP image encoder
    
    return mediapipe_app, clip_encoder

def get_face_embeddings(img_path, mediapipe_app, clip_encoder, device):
    """
    Extracts face embeddings from an image using MediaPipe for face detection and CLIP for feature extraction.
    Returns normalized face embeddings or None if no faces are detected.
    """
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')  # Convert to RGB if not already
    embeddings = []
    batched_selected_boxes, _, _, _ = mediapipe_app.predict_landmarks_from_image(img, raw_output=True)
    
    if batched_selected_boxes is None:
        return None  # No faces detected
    else:
        for box in batched_selected_boxes[0]:
            # Extract coordinates (x1, y1, x2, y2)
            x1, y1 = box[0][0].int().item(), box[0][1].int().item()  # Convert to integers
            x2, y2 = box[1][0].int().item(), box[1][1].int().item()
            # Crop the image using the bounding box coordinates
            cropped_image = img.crop((x1, y1, x2, y2))
            cropped_image = transform_image(device, cropped_image)  # Apply transformations
            emb = clip_encoder(cropped_image)  # Extract features using CLIP
            embeddings.append(emb)
    return embeddings

def compute_rrf(rank_clip, rank_face, k=60):
    """
    Computes Reciprocal Rank Fusion (RRF) score for two ranked lists.
    Returns an RRF score for fusion-based ranking.
    """
    return 1 / (k + rank_clip) + 2 / (k + rank_face)

def normalize_embeddings(embeddings):
    """
    Normalizes embeddings using L2 normalization.
    Returns normalized embeddings.
    """
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)

def save_to_pickle(obj, file_path):
    """
    Saves an object to a file using pickle.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def load_from_pickle(file_path):
    """
    Loads an object from a pickle file.
    Returns the loaded object.
    """
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj
