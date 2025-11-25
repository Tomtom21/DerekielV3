from pathlib import Path
from fastapi import FastAPI, Request
from label_studio_ml.model import LabelStudioMLBase
from ultralytics import YOLO
import uvicorn

# Defining the model path
model_path = Path(__file__).parent / "best.pt"

# Defining the label config
label_config = """
<View>
  <RectangleLabels name="label" toName="image">
    <Label value="car-my" />
    <Label value="cone" />
    <Label value="person" />
    <Label value="speedsign" />
    <Label value="stoplight-green" />
    <Label value="stoplight-red" />
    <Label value="stoplight-yellow" />
    <Label value="stopsign" />
  </RectangleLabels>
  <Image name="image" value="$image" />
</View>
"""
label_lookup = {
    0: "car-my",
    1: "cone",
    2: "person",
    3: "speedsign",
    4: "stoplight-green",
    5: "stoplight-red",
    6: "stoplight-yellow",
    7: "stopsign"
}

# Defining the YOLOBackend class
class YOLOBackend(LabelStudioMLBase):
    def __init__(self, model_path=model_path, **kwargs):
        super().__init__(**kwargs)
        self.model = YOLO(model_path)

    def predict(self, tasks, **kwargs):
        result_tasks = {
            "results": []
        }
        for task in tasks:
            image_url = task['data']['image']
            image = self.get_local_path(image_url)
            results = self.model(image)
            predictions = []
            for result in results:
                boxes = result.boxes
                orig_shape = result.orig_shape  # (height, width)
                width = orig_shape[1]
                height = orig_shape[0]
                for box in boxes:
                    box_np = box.cpu().numpy()
                    x1, y1, x2, y2 = box_np.xyxy[0]
                    confidence = float(box_np.conf[0])
                    cls = int(box_np.cls[0])
                    predictions.append({
                        "from_name": "label",
                        "to_name": "image",
                        "type": "rectanglelabels",
                        "value": {
                            "x": float(x1 / width) * 100,
                            "y": float(y1 / height) * 100,
                            "width": float((x2 - x1) / width) * 100,
                            "height": float((y2 - y1) / height) * 100,
                            "rectanglelabels": [str(label_lookup.get(cls, "unknown"))]
                        }
                    })
            result_tasks["results"].append({
                "model_version": "v1",
                "result": predictions
            })
            print(result_tasks)
        return result_tasks

app = FastAPI()
backend = YOLOBackend(label_config=label_config)

@app.post("/predict")
async def predict_endpoint(request: Request):
    data = await request.json()
    tasks = data.get("tasks", [])
    predictions = backend.predict(tasks)
    return predictions

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/setup")
def setup():
    return {"status": "setup complete"}

if __name__ == "__main__":
    uvicorn.run("server2:app", host="0.0.0.0", port=9090)
