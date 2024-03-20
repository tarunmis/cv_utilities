# Face blur in videos on your *intel laptop/PC

This project utilizes the power of OpenVINO for video processing to detect objects in a video stream, applying a blurring effect to the detected objects. It leverages OpenVINO's inference capabilities along with OpenCV for video manipulation, making it suitable for various applications such as privacy enhancements in video streams.

## Installation

Before running this project, you need to install Python 3.10 or later. This project depends on OpenCV and OpenVINO Toolkit. Follow the steps below to set up your environment:


Usage
To use this script, you need to have a video file that you want to process. The script takes several command-line arguments, including the path to the input video, the path for the output video, the device to use for inference (e.g., CPU, GPU), the detection threshold, and the path to the model XML file.

Run the script with the following command, replacing the placeholders with your actual file paths and desired values:


python faceblur.py --input_video path/to/your/input/video.mp4 --output_video path/to/your/output/video.mp4 --device CPU --threshold 0.5 --model_xml path/to/your/model.xml

Arguments
--input_video: Path to the input video file (required).
--output_video: Path for saving the output video file (default: output_video.mp4).
--device: Inference device name (default: CPU).
--threshold: Confidence threshold for detection (default: 0.28).
--model_xml: Path to the model XML file (default: model.xml).

Dependencies
Python 3.10 or later
OpenCV-Python
OpenVINO Toolkit 2024.0.0
