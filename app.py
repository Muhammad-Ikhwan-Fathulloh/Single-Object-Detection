import base64
import io

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import mobilenet_v2


def cv2_image_to_base64(image):
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")


# Load the trained model
class ObjectDetectionModel(nn.Module):
    def __init__(self):
        super(ObjectDetectionModel, self).__init__()
        # Load MobileNetV2 backbone
        self.backbone = mobilenet_v2(weights="DEFAULT").features
        # Create custom head (fully connected layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 5),  # 1 for is_object and 4 for bounding box (x, y, w, h)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        is_object = torch.sigmoid(x[:, :1])  # Sigmoid for is_object
        bbox = torch.sigmoid(x[:, 1:])  # Bounding box normalized to [0, 1] range
        return is_object, bbox


# Initialize FastAPI app
app = FastAPI()

# Load the model and set it to evaluation mode
model = ObjectDetectionModel()
model.load_state_dict(
    torch.load("best_object_detection.pt", map_location=torch.device("cpu"))
)
model.eval()

# Transformation to apply to input images
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the image from the uploaded file
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        img_width, img_height = image.size

        # Apply transformations to the image
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            is_object, bbox = model(image_tensor)

        # Extract results
        is_object_value = is_object.item()  # Get single value
        bbox = bbox.squeeze(0).tolist()  # Convert tensor to list (x, y, w, h)
        x, y, w, h = bbox
        x = x * img_width
        y *= img_height
        w *= img_width
        h *= img_height
        #
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
        cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 1)

        if is_object_value > 0.5:
            response = {
                "is_object": True,
                "bbox": {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                },
                "image_base64": cv2_image_to_base64(cv_image),
            }
        else:
            response = {
                "is_object": False,
                "bbox": None,  # No object, return null for bbox
            }

        return JSONResponse(content=response)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)