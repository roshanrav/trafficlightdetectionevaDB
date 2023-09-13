import evadb
import sys
import cv2
from ipywidgets import Video

def annotate_video_with_traffic_light_color(detections, input_video_path, output_video_path):
    object_bbox_color = (207, 248, 64)
    text_color = (255, 255, 255)
    thickness = 2

    vcap = cv2.VideoCapture(input_video_path)
    width = int(vcap.get(3))
    height = int(vcap.get(4))
    fps = vcap.get(5)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # Codec
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_id = 0
    ret, frame = vcap.read()

    while ret:
        frame_detections = detections
        frame_detections = frame_detections[['yolo.bboxes', 'yolo.labels']][frame_detections.index == frame_id]

        if not frame_detections.empty:
            bbox_list, label_list = frame_detections.values[0]

            for bbox, label in zip(bbox_list, label_list):
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), object_bbox_color, thickness)
                
                if label == "traffic light":
                    # Extract the region of interest (ROI) for the traffic light
                    traffic_light_roi = frame[y1:y2, x1:x2]
                    
                    # Determine the color of the traffic light
                    color = determine_traffic_light_color(traffic_light_roi)
                    
                    if color == "red":
                        text = "TRAFFIC LIGHT: RED, STOP!"
                    elif color == "green":
                        text = "TRAFFIC LIGHT: GREEN, GO!"
                    else:
                        text = "TRAFFIC LIGHT: UNKNOWN"

                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, thickness)
                else:
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, thickness)

                cv2.putText(frame, 'Frame ID: ' + str(frame_id), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, thickness)

            video.write(frame)

        frame_id += 1
        ret, frame = vcap.read()

    video.release()
    vcap.release()

def determine_traffic_light_color(roi):
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_red = (0, 50, 50)
    upper_red = (10, 255, 255)
    lower_green = (40, 50, 50)
    upper_green = (80, 255, 255)
    
    mask_red = cv2.inRange(hsv_roi, lower_red, upper_red)
    mask_green = cv2.inRange(hsv_roi, lower_green, upper_green)
    red_pixel_count = cv2.countNonZero(mask_red)
    green_pixel_count = cv2.countNonZero(mask_green)
    if red_pixel_count > green_pixel_count:
        return "red"
    else:
        return "green"

def main():
    cursor = evadb.connect().cursor()

    # Drop existing table if it exists
    response = cursor.query("DROP TABLE IF EXISTS ObjectDetectionVideos;").df()
    print(response)

    video_file = sys.argv[1]
    input_path = video_file
    response = cursor.load(input_path, table_name="ObjectDetectionVideos", format="VIDEO").df()
    print(response)
    response = cursor.query("""
        CREATE UDF IF NOT EXISTS Yolo
        TYPE ultralytics
        'model' 'yolov8m.pt';
        """).df()
    print(response)

    yolo_query = cursor.table("ObjectDetectionVideos")
    yolo_query = yolo_query.filter("id < 100")
    yolo_query = yolo_query.select("id, Yolo(data)")

    # Get detections and annotate the video
    response = yolo_query.df()
    print(response)

    input_path = video_file
    output_path = "res" + video_file

    # Annotate the video with traffic light colors
    annotate_video_with_traffic_light_color(response, input_path, output_path)
    Video.from_file(output_path)

if __name__ == '__main__':
    main()