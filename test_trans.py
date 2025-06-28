from ultralytics import YOLO
import cv2
import numpy as np
import torch
from sklearn.metrics import mean_squared_error

print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

model_f = YOLO("models/Yolo8MF/weights/best.pt")
cap = cv2.VideoCapture('dataset/test.mp4')
def draw_dot_with_number(image, coordinate, index):
    # Draw a dot
    cv2.circle(image, coordinate, 5, (0, 255, 0), -1)  # Green dot

    # Draw the index number
    cv2.putText(image, str(index), (coordinate[0] - 10, coordinate[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Red number
transform_kp = np.load('field_coord.npy')
image_kp_prev = np.array([])
image_kp_cls_prev = []
frame_ind = 0
while cap.isOpened():
    success, frame = cap.read()
    if success:
        frame_ind+=1
        field_result = model_f.predict(frame,show = True,conf = 0.7)
        field_result = field_result[0]
        image_kp_cls = field_result.boxes.cls.cpu().numpy()
        image_kp_cls = list(image_kp_cls.astype(np.int32))
        transform_kp_f = transform_kp[image_kp_cls]
        image_kp = field_result.boxes.xyxy.cpu().numpy()
        image_kp = np.mean(image_kp.reshape(-1, 2, 2), axis=1)
        image_kp = image_kp.astype(np.int16)
        if(len(image_kp_cls) > 3):
            if frame_ind > 1:
                common_labels = set(image_kp_cls_prev) & set(image_kp_cls)
                if len(common_labels) > 3:
                    common_label_idx_prev = [image_kp_cls_prev.index(i) for i in common_labels] 
                    common_label_idx_curr = [image_kp_cls.index(i) for i in common_labels]
                    error = mean_squared_error(image_kp_prev[common_label_idx_prev],image_kp[common_label_idx_curr])
                    update_H = error > 50
                else:
                    update_H = True
            else:
                update_H = True
            if update_H:
                homography, _ = cv2.findHomography(image_kp, transform_kp_f)
            image_kp_prev = image_kp.copy()
            image_kp_cls_prev = image_kp_cls.copy()
        graph = cv2.imread('field_graph.jpg')


        height, width = graph.shape[:2]
        warped_image = cv2.warpPerspective(frame, homography, (width, height))
        combined_image = cv2.addWeighted(graph, 0.5, warped_image, 0.5, 0)
        # for coord, i  in zip(image_kp, image_kp_cls):
        #     draw_dot_with_number(frame, coord, i)
        # cv2.imshow("YOLOv8 Tracking", frame)
        cv2.imshow('abc', combined_image)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('c'):
            continue
        elif key == ord('q'):
            break
    else:
        break