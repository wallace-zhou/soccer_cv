from ultralytics import YOLO
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
import skimage
from sklearn.metrics import mean_squared_error

def find_color(player_box, player, color_list_lab, palette_list, frame):
    box = player_box[player]                   
    obj_img = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])] 
    width, height = obj_img.shape[1], obj_img.shape[0]
    #shrink the box to incorporate less of thd grass
    x1 = np.max([(width // 2) - (width // 6), 1])
    x2 = (width // 2) + (width // 6)
    y1 = np.max([(height // 3) - (height // 6), 1])
    y2 = (height // 3) + (height // 6)
    center_filter = obj_img[y1:y2, x1:x2]
    pil_img = Image.fromarray(np.uint8(center_filter))  
    #reduce the image size to just that of the small box
    reduced = pil_img.convert("P", palette=Image.Palette.WEB)
    #find the colors of the image
    palette = reduced.getpalette()                     
    palette = [palette[3 * n:3 * n + 3] for n in range(256)]      
    color_count = [(n, palette[m]) for n, m in reduced.getcolors()]                
    RGB_df = pd.DataFrame(color_count, columns = ['cnt', 'RGB']).sort_values(
                                                by = 'cnt', ascending = False).iloc[
                                                (0,5)[0]:(0,5)[1], :]
    palette = list(RGB_df.RGB) 
                
    palette_list.append(palette)
    players_distance_features = []
    #iterate through each palette in the lists of palettes
    for palette in palette_list:
        palette_distance = []
        palette_lab = [skimage.color.rgb2lab([i/255 for i in color]) for color in palette]
        #iterate through each of the folors in thre paleette list
        for color in palette_lab:
            distance_list = []
            #iterate thtouhg each team color to find the color with the least distacne
            for c in color_list_lab:
                distance = skimage.color.deltaE_cie76(color, c)                             
                distance_list.append(distance)                                              
            palette_distance.append(distance_list)                                          
        players_distance_features.append(palette_distance)                
    players_teams_list = 0
    
    for dist in players_distance_features:
        vote = []
        for dist_list in dist:
            team_idx = dist_list.index(min(dist_list)) // 2                   
            vote.append(team_idx)                                              
        players_teams_list = (max(vote, key=vote.count))
    return players_teams_list

def transform(video_path,graph_path,transform_kp,device,save):
    model_p = YOLO("models/Yolo8LP/weights/best.pt")
    model_f = YOLO("models/Yolo8MF/weights/best.pt")
    cap = cv2.VideoCapture(video_path)
    prev_ball_x = 0
    prev_ball_y = 0
    frames_since_update = 0
    speed = 0
    image_kp_prev = np.array([])
    image_kp_cls_prev = []
    frame_ind = 0

    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    graph = cv2.imread(graph_path)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'H264')  # You can also use 'MP4V' for .mp4
    out = cv2.VideoWriter('output_video.mp4', fourcc, frame_rate, (graph.shape[1], graph.shape[0]))

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            frame_ind+=1
            field_result = model_f.predict(frame,conf = 0.5)
            field_result = field_result[0]
            image_kp_cls = field_result.boxes.cls.cpu().numpy()
            image_kp_cls = list(image_kp_cls.astype(np.int32))
            if(field_result.boxes.shape[0] == 0):
                break
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
                        update_H = error > 10
                    else:
                        update_H = True
                else:
                    update_H = True
                if update_H:
                    homography, _ = cv2.findHomography(image_kp, transform_kp_f)
                image_kp_prev = image_kp.copy()
                image_kp_cls_prev = image_kp_cls.copy()
            player_result = model_p.track(frame,persist = True, conf = 0.60)
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


            height, width = graph.shape[:2]
            warped_image = cv2.warpPerspective(frame, homography, (width, height))
            # Define colors for each class
            cls_colors = [(255, 0, 0), (0, 0, 0), (255, 255, 255)]
            colors = {"Chelsea":[(41,71,138), (220,98,88)],
                        "Man City":[(144,200,255), (188,199,3)] #(Players kit color, GK kit color)                
                    }
            # colors = {"France":[(33, 48, 77), (242, 232, 34)],
            #             "Croatia":[(255,255,255), (66, 255, 28)] #(Players kit color, GK kit color)                
            #         }
            colors_list = colors["Chelsea"] + colors["Man City"]
            # colors_list = colors["France"] + colors["Croatia"]
            color_list_lab = [skimage.color.rgb2lab([i/255 for i in c]) for c in colors_list] 
            color_array = [(138,71,41), (255,200,144)]
            palette_list = []                                                                
            count = 0
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
            for xyz, cls, obj_id in zip(transformed_player_loc, player_cls.astype(np.int16), player_id):
                x,y,_ = xyz
                #if class == player send to function to determine which team color
                if(cls == 0):
                    color = color_array[find_color(player_box, count, color_list_lab, palette_list, frame)]
                #if class == ball solve for new speed of ball, currently in pizels per frame, transitioning to mps seems diffcult since frame does not cover entire pitch
                elif(cls == 2):
                    color = (255, 255, 255)
                    speed = (np.sqrt((x - prev_ball_x)**2 + (y - prev_ball_y)**2)) / frames_since_update * 0.15 * frame_rate #0.15 meter per pixel and 30 frame per second
                    frames_since_update = 0
                    prev_ball_x = x
                    prev_ball_y = y
                #else class == referee, color == black
                else:
                    color = cls_colors[cls]
                cv2.putText(graph, str(round(speed,2)) + ' m/s', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.circle(graph,(int(x),int(y)), 5, color, -1)
                cv2.putText(graph, str(obj_id), (int(x) - 5, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (int(player_box[count][0]), int(player_box[count][1])), 
                                    (int(player_box[count][2]), int(player_box[count][3])), (0, 255, 0), 2)
                count += 1
                frames_since_update += 1
            print(speed)
            cv2.imshow("YOLOv8 Tracking", graph)    
            cv2.imshow("YOLOv8 Tracking reference", frame)
            cv2.imshow("YOLOv8 Tracking reference", warped_image)
            # key = cv2.waitKey(0) & 0xFF
            # if key == ord('c'):
            #     continue
            # elif key == ord('q'):
            #     break
            if(save):
                out.write(graph)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break
    out.release()
    cap.release()
    cv2.destroyAllWindows()


def transform_to_vid(video_path, graph_path, transform_kp, device):
    model_p = YOLO("models/Yolo8LP/weights/best.pt")
    model_f = YOLO("models/Yolo8MF/weights/best.pt")
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties for VideoWriter
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'H264')  # You can also use 'MP4V' for .mp4
    out = cv2.VideoWriter('output_video.mp4', fourcc, frame_rate, (frame_width, frame_height))
    
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
            homography, _ = cv2.findHomography(image_kp, transform_kp_f, cv2.RANSAC, 1.0)
            player_result = model_p.track(frame,persist = True)
            player_result = player_result[0]
            player_box = player_result.boxes.xyxy.cpu().numpy()
            player_cls = player_result.boxes.cls.cpu().numpy() # 0 player, 1 ref, 2 ball
            player_id = player_result.boxes.id.cpu().numpy() if player_result.boxes.id is not None else None
            player_xloc = np.mean(player_box[:,[0,2]],axis = 1)
            player_yloc = np.max(player_box[:,[1,3]], axis = 1)
            player_loc = np.hstack([np.expand_dims(player_xloc,1),np.expand_dims(player_yloc,1),np.ones_like(np.expand_dims(player_yloc,1))])

            player_coords = player_loc[:, :2]
            player_coords = player_coords[np.newaxis, ...]  # same as player_coords[np.newaxis]
            player_points_transformed = cv2.perspectiveTransform(player_coords, homography)
            player_points_transformed = player_points_transformed[0]
            graph = cv2.imread(graph_path)
            transformed_player_loc = np.hstack((player_points_transformed, np.ones((player_points_transformed.shape[0], 1))))
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red for players, green for refs, blue for ball

            # Draw transformed player locations and label them
            for point, cls, pid in zip(player_points_transformed, player_cls.astype(int), player_id.astype(int)):
                cv2.circle(graph, (int(point[0]), int(point[1])), 5, colors[cls], -1)
                cv2.putText(graph, f'ID:{pid}', (int(point[0])+10, int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            out.write(graph)
            
        else:
            # Break the loop if the end of the video is reached
            break
            
    # Release everything if job is finished
    cap.release()
    out.release()  # Save the output video
    cv2.destroyAllWindows()


if __name__ == '__main__':
    kp = np.load('field_coord.npy')
    transform('dataset/test.mp4','field_graph.jpg',kp,torch.device('cuda'),True)
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