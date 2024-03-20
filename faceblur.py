import openvino as ov
import cv2
import numpy as np
import argparse  # Import argparse library

def draw_rectangle(img, x_min_ratio, y_min_ratio, x_max_ratio, y_max_ratio, color=(0, 255, 0), thickness=2):
    height, width = img.shape[:2]
    x_min = int(x_min_ratio * width)
    y_min = int(y_min_ratio * height)
    x_max = int(x_max_ratio * width)
    y_max = int(y_max_ratio * height)

    roi = img[y_min:y_max, x_min:x_max]
    blurred_roi = cv2.blur(roi, (29, 29), 0)
    img[y_min:y_max, x_min:x_max] = blurred_roi
    return img

def process_frame(frame, model, output_layer, ther=0.28):
    img = cv2.resize(frame, (300, 300))
    img_data = np.expand_dims(img.transpose(2, 0, 1), 0)
    boxes = model(img_data)[output_layer]
    boxes = np.reshape(boxes, (200, 7))
    for box in boxes:
        conf = box[2]
        if conf > ther:
            x_min, y_min, x_max, y_max = box[3:]
            frame = draw_rectangle(frame, x_min, y_min, x_max, y_max)
    return frame

def load_model(device, model_xml_path):
    core = ov.Core()
    model = core.read_model(model=model_xml_path)
    compiled_model = core.compile_model(model=model, device_name=device)
    input_layer_ir = compiled_model.input(0)
    output_layer_ir = compiled_model.output(0)
    return compiled_model, input_layer_ir, output_layer_ir

# Set up argument parser
parser = argparse.ArgumentParser(description='Video processing with OpenVINO.')
parser.add_argument('--input_video', type=str, required=True, help='Path to input video file')
parser.add_argument('--output_video', type=str, default='output_video.mp4', help='Path for saving output video file')
parser.add_argument('--device', type=str, default='CPU', help='Inference device name (e.g., CPU, GPU)')
parser.add_argument('--threshold', type=float, default=0.28, help='Confidence threshold for detection')
parser.add_argument('--model_xml', type=str, default='model.xml', help='Path to the model XML file')
args = parser.parse_args()

# Load model
model, _, output_layer = load_model(args.device, args.model_xml)

# Open the video file
cap = cv2.VideoCapture(args.input_video)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create a VideoWriter object to write the video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(args.output_video, fourcc, fps, (frame_width, frame_height), isColor=True)

# Process and write frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = process_frame(frame, model, output_layer, args.threshold)
    out.write(frame)  # Write the processed frame to the output video

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
