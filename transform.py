from ultralytics import YOLO
import cv2
import torch
import numpy as np

def transform(video_path,graph_path,transform_kp,device):
    model_p = YOLO("models/Yolo8LP/weights/best.pt")
    model_f = YOLO("models/Yolo8MF/weights/best.pt")
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            
            field_result = model_f.predict(frame)
            field_result = field_result[0]
            image_kp_cls = field_result.boxes.cls.cpu().numpy()
            image_kp_cls = image_kp_cls.astype(np.int32)

            transform_kp_f = transform_kp[image_kp_cls]
            image_kp = field_result.boxes.xyxy.cpu().numpy()
            image_kp = np.mean(image_kp.reshape(-1, 2, 2), axis=1)
            image_kp = image_kp.astype(np.int16)
            homography, _ = cv2.findHomography(image_kp, transform_kp_f)

            player_result = model_p.track(frame,persist = True)
            player_result = player_result[0]
            player_box = player_result.boxes.xyxy.cpu().numpy()
            player_cls = player_result.boxes.cls.cpu().numpy() # 0 player, 1 ref, 2 ball
            player_id = player_result.boxes.id.cpu().numpy()
            player_xloc = np.mean(player_box[:,[0,2]],axis = 1)
            player_yloc = np.max(player_box[:,[1,3]], axis = 1)
            player_loc = np.hstack([np.expand_dims(player_xloc,1),np.expand_dims(player_yloc,1),np.ones_like(np.expand_dims(player_yloc,1))])
            transformed_player_loc = np.dot(homography,player_loc.T).T
            transformed_player_loc /= transformed_player_loc[:, 2][:, np.newaxis]
            graph = cv2.imread(graph_path)
    
    # Define colors for each class
            colors = [(255, 0, 0), (0, 0, 0), (255, 255, 255)]
            for xyz, cls, obj_id in zip(transformed_player_loc, player_cls.astype(np.int16), player_id):
                x,y,_ = xyz
                color = colors[cls]

                cv2.circle(graph,(int(x),int(y)), 5, color, -1)
                cv2.putText(graph, str(obj_id), (int(x) - 5, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.imshow("YOLOv8 Tracking", graph)
            cv2.imshow("YOLOv8 Tracking reference", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    kp = np.load('field_coord.npy')
    transform('dataset/test.mp4','field_graph.jpg',kp,torch.device('cuda'))
    # from collections import defaultdict

    # import cv2
    # import numpy as np

    # from ultralytics import YOLO

    # # Load the YOLOv8 model
    # model = YOLO("models/Yolo8LP/weights/best.pt")

    # # Open the video file
    # video_path = "dataset/test.mp4"
    # cap = cv2.VideoCapture(video_path)

    # # Store the track history
    # track_history = defaultdict(lambda: [])

    # # Loop through the video frames
    # while cap.isOpened():
    #     # Read a frame from the video
    #     success, frame = cap.read()

    #     if success:
    #         # Run YOLOv8 tracking on the frame, persisting tracks between frames
    #         results = model.track(frame, persist=True)

    #         # Get the boxes and track IDs
    #         boxes = results[0].boxes.xywh.cpu()
    #         track_ids = results[0].boxes.id.int().cpu().tolist()

    #         # Visualize the results on the frame
    #         annotated_frame = results[0].plot()

    #         # Plot the tracks
    #         for box, track_id in zip(boxes, track_ids):
    #             x, y, w, h = box
    #             track = track_history[track_id]
    #             track.append((float(x), float(y)))  # x, y center point
    #             if len(track) > 30:  # retain 90 tracks for 90 frames
    #                 track.pop(0)

    #             # Draw the tracking lines
    #             points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
    #             cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

    #         # Display the annotated frame
    #         cv2.imshow("YOLOv8 Tracking", annotated_frame)

    #         # Break the loop if 'q' is pressed
    #         if cv2.waitKey(1) & 0xFF == ord("q"):
    #             break
    #     else:
    #         # Break the loop if the end of the video is reached
    #         break

    # # Release the video capture object and close the display window
    # cap.release()
    # cv2.destroyAllWindows()