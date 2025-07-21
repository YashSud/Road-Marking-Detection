from ultralytics import YOLO
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog

# -------------------------------------------
# CUSTOM CLASSES & COLORS
# -------------------------------------------
CLASSES = {
    
   
    "DashedAndSolidYellowLines": (128, 0, 128),
    "DashedWhiteLines": (0, 128, 255),
    "DashedYellowLines": (255, 165, 0),
    "DoNotEnter": (255, 20, 147),
    "DoubleSolidWhiteLines": (192, 192, 192),
    "Go Straight": (0, 255, 127),
    "Go Straight or Left Turn": (100, 149, 237),
    "Go Straight or Right Turn": (34, 139, 34),
    "LeftTurn": (220, 20, 60),
    "No Parking": (139, 0, 0),
    "Stop": (255, 69, 0),
    "Zebra Crossing": (70, 130, 180)
}

# -------------------------------------------
# LOAD MODEL
# -------------------------------------------

model = YOLO(r"C:\Users\Yash Sud\OneDrive\Desktop\road marking\best.pt")  

# -------------------------------------------
# TKINTER GUI SETUP
# -------------------------------------------
root = tk.Tk()
root.title("Road Marking Detection")

label = tk.Label(root, text="Choose input type:", font=("Arial", 14))
label.pack(pady=10)

# New label to display detections
detection_list_label = tk.Label(root, text="Detections:", font=("Arial", 12), fg="blue", justify="left")
detection_list_label.pack(pady=10)

# -------------------------------------------
# IMAGE DETECTION FUNCTION
# -------------------------------------------
def detect_image(path):
    results = model(path)[0]
    img = cv2.imread(path)
    detections_text = ""

    for box in results.boxes:
        cls_id = int(box.cls)
        cls_name = results.names[cls_id]
        color = CLASSES.get(cls_name, (255, 255, 255))
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf.item()
        label = f"{cls_name}: {conf:.2f}"
        detections_text += f"{label}\n"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show detections
    detection_list_label.config(text="Detections:\n" + detections_text)
    cv2.imshow("Image Detection Result", img)

    # Save detected image
    output_path = path.rsplit(".", 1)[0] + "_detected.jpg"
    cv2.imwrite(output_path, img)
    print(f"Image saved at: {output_path}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -------------------------------------------
# VIDEO DETECTION FUNCTION
# -------------------------------------------
def detect_video(path):
    cap = cv2.VideoCapture(path)

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = path.rsplit(".", 1)[0] + "_detected.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        detections_text = ""

        for box in results.boxes:
            cls_id = int(box.cls)
            cls_name = results.names[cls_id]
            color = CLASSES.get(cls_name, (255, 255, 255))
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf.item()
            label = f"{cls_name}: {conf:.2f}"
            detections_text += f"{label}\n"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        detection_list_label.config(text="Detections:\n" + detections_text)
        cv2.imshow("Video Detection Result", frame)

        # Write the frame to output video
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved at: {output_path}")

# -------------------------------------------
# FILE SELECTOR
# -------------------------------------------
def select_file(mode):
    file_path = filedialog.askopenfilename()
    if mode == "image":
        detect_image(file_path)
    elif mode == "video":
        detect_video(file_path)

# -------------------------------------------
# GUI BUTTONS
# -------------------------------------------
btn_image = tk.Button(root, text="Detect from Image/Video", command=lambda: select_file("image"), width=30)
btn_image.pack(pady=10)

# -------------------------------------------
# MAIN LOOP
# -------------------------------------------
root.mainloop()
