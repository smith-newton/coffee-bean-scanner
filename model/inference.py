import torch
import torch.nn.functional as F
import timm
from PIL import Image
from Transformer import val_transforms

classes = ['Dark', 'Green', 'Light', 'Medium']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = timm.create_model('efficientnet_b0', pretrained=False)
model.classifier = torch.nn.Linear(model.classifier.in_features, 4)
model.load_state_dict(torch.load("../coffee_model.pth", map_location=device))
model = model.to(device)
model.eval()

def predict_image(image):
    image = val_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)

    return {
        "class": classes[pred.item()],
        "confidence": float(confidence.item())
    }