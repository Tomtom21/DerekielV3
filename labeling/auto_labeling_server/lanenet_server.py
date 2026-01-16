from pathlib import Path
from fastapi import FastAPI, Request
from label_studio_ml.model import LabelStudioMLBase
import torch
from PIL import Image
from torchvision import transforms
import uvicorn
import numpy as np
import os
import sys

# Add model path for import
sys.path.append(str(Path(__file__).parent.parent.parent / "models" / "lanenet"))
from model import LaneNet

# Label config for keypoints (Label Studio)
label_config = """
<View style="position:relative; display:inline-block; width:100%;">
  <View style="position:relative; width:100%; overflow:hidden;">
    <Image name="img" value="$img" style="width:100%; display:block;" />
    <!-- row guides omitted for brevity -->
  </View>
  <KeyPointLabels name="lane_left" toName="img">
    <Label value="L_row_0" maxUsages="1" background="blue" />
    <Label value="L_row_1" maxUsages="1" background="blue" />
    <Label value="L_row_2" maxUsages="1" background="blue" />
    <Label value="L_row_3" maxUsages="1" background="blue" />
    <Label value="L_row_4" maxUsages="1" background="blue" />
    <Label value="L_row_5" maxUsages="1" background="blue" />
    <Label value="L_row_6" maxUsages="1" background="blue" />
    <Label value="L_row_7" maxUsages="1" background="blue" />
    <Label value="L_row_8" maxUsages="1" background="blue" />
    <Label value="L_row_9" maxUsages="1" background="blue" />
  </KeyPointLabels>
  <KeyPointLabels name="lane_right" toName="img">
    <Label value="R_row_0" maxUsages="1" background="lime" />
    <Label value="R_row_1" maxUsages="1" background="lime" />
    <Label value="R_row_2" maxUsages="1" background="lime" />
    <Label value="R_row_3" maxUsages="1" background="lime" />
    <Label value="R_row_4" maxUsages="1" background="lime" />
    <Label value="R_row_5" maxUsages="1" background="lime" />
    <Label value="R_row_6" maxUsages="1" background="lime" />
    <Label value="R_row_7" maxUsages="1" background="lime" />
    <Label value="R_row_8" maxUsages="1" background="lime" />
    <Label value="R_row_9" maxUsages="1" background="lime" />
  </KeyPointLabels>
</View>
"""

# Setpoint rows: 10 y-positions from 85% to 35.5%, 5.5% apart (from top)
def get_row_y_positions(img_height):
    return [int(img_height * (0.85 - i * 0.055)) for i in range(10)]

class LaneNetBackend(LabelStudioMLBase):
    def __init__(self, model_path=None, **kwargs):
        super().__init__(**kwargs)
        # Model path
        if model_path is None:
            model_path = Path(__file__).parent / "lanenet_best.pth"
        self.device = (
            torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.model = LaneNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((128, 256)),  # height=128, width=256
            transforms.ToTensor(),
        ])

    def predict(self, tasks, **kwargs):
        result_tasks = {
            "results": []
        }
        for task in tasks:
            # Getting the image and processing it with YOLO
            image_url = task['data']['img']
            image_path = self.get_local_path(image_url)
            orig_img = Image.open(image_path).convert("RGB")
            width, height = orig_img.size
            img = self.transform(orig_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(img).cpu().numpy()[0]
            
            # Keypoint format: [visibility, x] * 20 (10 left, 10 right)
            y_positions = get_row_y_positions(height)
            keypoints = []
            # Left lane: first 20 values (0-19, pairs)
            for i in range(10):
                vis = output[i*2]
                x_norm = output[i*2+1]
                if vis > 0.5:
                    x_pixel = float(x_norm * width)
                    y_pixel = float(y_positions[i])
                    keypoints.append({
                        "from_name": "lane_left",
                        "to_name": "img",
                        "type": "keypointlabels",
                        "value": {
                            "x": x_pixel / width * 100,
                            "y": y_pixel / height * 100,
                            "width": 0,
                            "keypointlabels": [f"L_row_{i}"]
                        }
                    })
            # Right lane: next 20 values (20-39, pairs)
            for i in range(10, 20):
                vis = output[i*2]
                x_norm = output[i*2+1]
                if vis > 0.5:
                    x_pixel = float(x_norm * width)
                    y_pixel = float(y_positions[i-10])
                    keypoints.append({
                        "from_name": "lane_right",
                        "to_name": "img",
                        "type": "keypointlabels",
                        "value": {
                            "x": x_pixel / width * 100,
                            "y": y_pixel / height * 100,
                            "width": 0,
                            "keypointlabels": [f"R_row_{i-10}"]
                        }
                    })
            result_tasks["results"].append({
                "model_version": "v1",
                "result": keypoints
            })
        return result_tasks
app = FastAPI()
backend = LaneNetBackend(label_config=label_config)

@app.post("/predict")
async def predict_endpoint(request: Request):
    data = await request.json()
    tasks = data.get("tasks", [])
    predictions = backend.predict(tasks)
    print(predictions)
    return predictions

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/setup")
def setup():
    return {"status": "setup complete"}

if __name__ == "__main__":
    uvicorn.run("lanenet_server:app", host="0.0.0.0", port=9091)
