from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("yolov10n.pt")

# Define path to video file
source = "traffic3.mp4"

# Run inference on the source
model.predict(source,show=True,save=True)  # generator of Results objects