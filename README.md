# Object-Detection-Using-CV
import os
import cv2
import numpy as np
import time
import requests
import pyttsx3
import threading
from queue import Queue
from typing import List, Tuple
# Constants
YOLO_DIR = r'C:\Users\HP\Downloads\yolo_files'
CFG_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
WEIGHTS_URL = "https://pjreddie.com/media/files/yolov3.weights"
NAMES_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
IMAGE_FOLDER = r'C:\Users\HP\Pictures\images_project'
OUTPUT_FOLDER = r'C:\Users\HP\Downloads\Processed_Images'

# Create necessary directories
for directory in [YOLO_DIR, OUTPUT_FOLDER]:
    os.makedirs(directory, exist_ok=True)

# Download required YOLO files if not already downloaded
def download_yolo_files():
    for file_name, url in zip(["yolov3.cfg", "yolov3.weights", "coco.names"], [CFG_URL, WEIGHTS_URL, NAMES_URL]):
        file_path = os.path.join(YOLO_DIR, file_name)
        if not os.path.isfile(file_path):
            print(f"Downloading {file_name}...")
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                with open(file_path, "wb" if file_name.endswith(".weights") else "w") as f:
                    f.write(response.content if file_name.endswith(".weights") else response.text)
            except requests.exceptions.RequestException as e:
                print(f"Failed to download {file_name}: {e}")
                exit()
# Load YOLOv3 model
def load_yolo_model():
    return cv2.dnn.readNetFromDarknet(os.path.join(YOLO_DIR, "yolov3.cfg"), os.path.join(YOLO_DIR, "yolov3.weights"))

# Load class labels
def load_class_labels():
    classes = []
    with open(os.path.join(YOLO_DIR, "coco.names"), "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes
# Define YOLO layers
def define_yolo_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers
# Initialize text-to-speech engine
tts_engine = pyttsx3.init()

speech_queue = Queue()
speech_lock = threading.Lock()
speech_running = False

def speak_from_queue():
    global speech_running
    while True:
        text = speech_queue.get()
        with speech_lock:
            if not speech_running:
                speech_running = True
                tts_engine.say(text)
                tts_engine.runAndWait()
                speech_running = False
        speech_queue.task_done()

speech_thread = threading.Thread(target=speak_from_queue)
speech_thread.daemon = True
speech_thread.start()
def speak(text):
    speech_queue.put(text)
def detect_objects(img, net, output_layers, classes):
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detected_objects = []  
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] 
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]} {confidences[i]:.2f}"
            detected_objects.append(classes[class_ids[i]])  
            color = colors[class_ids[i] % len(colors)]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)   
    return img, detected_objects
def process_image_folder(net, output_layers, classes):
    if not os.path.exists(IMAGE_FOLDER):
        print(f"Error: The specified folder does not exist: {IMAGE_FOLDER}")
        return
    for img_name in os.listdir(IMAGE_FOLDER):
        img_path = os.path.join(IMAGE_FOLDER, img_name)
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Skipping non-image file: {img_name}")
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Failed to load image at {img_path}. Skipping...")
            continue
        img, detected_objects = detect_objects(img, net, output_layers, classes)
        if detected_objects:
            detected_text = f"In image {img_name}, detected objects are: " + ", ".join(detected_objects)
        else:
            detected_text = f"No objects detected in image {img_name}."
        print(detected_text)
        speak(detected_text)
        cv2.imshow(f"Object Detection - {img_name}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        output_path = os.path.join(OUTPUT_FOLDER, f'output_{img_name}')
        cv2.imwrite(output_path, img)
        print(f"Detection results saved at {output_path}.")
def upload_image(net, output_layers, classes):
    image_path = input("Enter the path to the image you want to upload: ")
    if not os.path.isfile(image_path):
        print("Error: The specified file does not exist.")
        return
    if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        print("Error: The specified file is not an image.")
        return
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Failed to load image at {image_path}.")
        return
    img, detected_objects = detect_objects(img, net, output_layers, classes)
    if detected_objects:
        detected_text = f"In the uploaded image, detected objects are: " + ", ".join(detected_objects)
    else:
        detected_text = f"No objects detected in the uploaded image."
    print(detected_text)
    speak(detected_text)
    cv2.imshow("Uploaded Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Replace with your own upload API or service
    print(f"Uploading {image_path}...")
    # Add your upload logic here
def process_webcam(net, output_layers, classes):
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return
    cv2.namedWindow("Webcam Object Detection", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Webcam Object Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam. Exiting...")
            break
        frame, detected_objects = detect_objects(frame, net, output_layers, classes)
        if detected_objects:
            detected_text = "Detected objects: " + ", ".join(detected_objects)
            print(detected_text)
            speak(detected_text)
        cv2.imshow("Webcam Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
def main():
    download_yolo_files()
    net = load_yolo_model()
    classes = load_class_labels()
    output_layers = define_yolo_layers(net)
    while True:
        mode = input("Enter mode (image/webcam/upload/quit): ").strip().lower()
        if mode == "quit":
            print("Exiting the program.")
            break
        elif mode == "image":
            process_image_folder(net, output_layers, classes)
        elif mode == "webcam":
            process_webcam(net, output_layers, classes)
        elif mode == "upload":
            upload_image(net, output_layers, classes)
        else:
            print("Invalid mode selected. Please choose 'image', 'webcam', 'upload', or 'quit'.")
    print("Processing complete.")

if __name__ == "__main__":
    main()

