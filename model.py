import cv2
import clip
import numpy as np

from PIL import Image

import torch
from torch import nn
from torchvision.transforms import transforms


class CLIP():
    def __init__(self, device='cpu'):
        self.device = device
        clip_model_name = "ViT-B/32"
        self.model, self.preprocess = clip.load(clip_model_name, device=device, jit=False)
        self.model = self.model.requires_grad_(False)
        self.model.eval()
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                              std=[0.26862954, 0.26130258, 0.27577711])

    @torch.no_grad()
    def embed_text(self,
                   text: str):
        text = clip.tokenize(text)
        # text_features = self.model.encode_text(text)
        return text

    def embed_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.preprocess(Image.fromarray(image)).unsqueeze(0)
        # image_features = self.model.encode_image(image)
        return image

    def get_probs(self,
                  image,
                  text):
        logits_per_image, logits_per_text = self.model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        return probs


class AffectRecognitionPipeline(nn.Module):
    def __init__(self,
                 path_cascade):
        super(AffectRecognitionPipeline, self).__init__()

        # Create the haar cascade
        self.dummy = nn.Parameter()
        self.face_classifier = cv2.CascadeClassifier(path_cascade)
        self.clip = CLIP()

    def forward(self,
                frame):
        faces = self.face_classifier.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Only recognize one face
        if len(faces) > 0:
            x, y, w, h = faces[0]

            # Use CLIP as affect recognition system
            image = self.clip.embed_image(frame[y:y+h, x:x+w, :])
            text = self.clip.embed_text(["A smiling face", "A frowning face"])
            probs = self.clip.get_probs(image, text)

            if np.argmax(probs[0]) == 0:
                cv2.putText(frame, "SMILE", (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                cv2.putText(frame, "FROWN", (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


        return frame