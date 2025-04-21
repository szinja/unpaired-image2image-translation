import torch
import torch.nn as nn
from torchvision.models import vgg19
from PIL import Image

# Loading CLIP from openai/CLIP
import clip 

class CLIPLoss:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()  # Freeze weights
    
    def get_loss(self, images, text_prompts):
        # text_prompts consists of a list of strings the same size of images (example; ["a painting by Monet", ...])
        assert len(images) == len(text_prompts), "Batch size mismatch"

        # Normalize images and move them to device
        images = torch.nn.functional.interpolate(images, size=(224, 224), mode='bilinear')
        image_features = self.model.encode_image(images)

        # Tokenize text
        text_tokens = clip.tokenize(text_prompts).to(self.device)
        text_features = self.model.encode_text(text_tokens)

        # Normalize
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # Cosine similarity (here higher is better, so loss = 1 - sim)
        similarity = torch.sum(image_features * text_features, dim=1)
        loss = 1 - similarity.mean()
        return loss