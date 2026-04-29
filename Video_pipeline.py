import cv2
from ultralytics import YOLO
import time



crash_model = YOLO(r"D:\Real Time Accident Detection\runs\detect\train-2\weights\best.pt")
vehicle_model = YOLO("yolov8n.pt")

video_path = r"D:\Real Time Accident Detection\Videos\INSANE CAR CRASHES COMPILATION  __ Best of USA & Canada Accidents - part 39.mp4"
output_path = r"D:\Real Time Accident Detection\Outputs\output_video.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.Try Again later!")
    exit()





width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps <= 0:
    fps = 25

writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))




vehicle_classes = [1,2, 3, 5, 7]
vehicle_names = {1:"bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

def traffic_density_estimation(count):
    if count < 5:
        return "Low"
    elif count < 15:
        return "Medium"
    else:
        return "High"
previous_time = 0
while True:
    frame_captured, frame = cap.read()
    if not frame_captured:
        break

    start_latency = time.time()


    current_time=time.time()
    fps = 1/(current_time-previous_time)
    previous_time = current_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (300,100),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,255,255), 2)


    accident_detected = False
   
    crash_results = crash_model(frame, conf=0.55)

    for r in crash_results:
        for box in r.boxes:
            accident_detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            end_latency = time.time()
            latency = end_latency - start_latency
            

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, f"CRASH {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            cv2.putText(frame, f"Latency..: {latency:.2f}ms", (20,100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

  



    vehicle_results = vehicle_model(frame, conf=0.35)
    vehicle_count = 0

    for r in vehicle_results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls in vehicle_classes:
                vehicle_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                label = vehicle_names.get(cls, "vehicle")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                )

    density = traffic_density_estimation(vehicle_count)

 
    if accident_detected:
        status = "ACCIDENT!"
    else:
        status = "NO ACCIDENT"
    

    cv2.putText(frame, f"Status: {status}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255,0,0) if accident_detected else (0,255,0), 2)

    cv2.putText(frame, f"Vehicles: {vehicle_count}", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.putText(frame, f"Density: {density}", (20,120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    writer.write(frame)
    
    cv2.imshow("Accident Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

print("Done! Output saved.")