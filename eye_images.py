import os
import cv2


# Функция для создания датасета из видеофайлов
def create_dataset_from_videos(input_dir, output_dir, label):
    video_files = os.listdir(input_dir)

    for video_file in video_files:
        video_path = os.path.join(input_dir, video_file)
        video_name = os.path.splitext(video_file)[0]
        output_subdir = os.path.join(output_dir, label)
        os.makedirs(output_subdir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % 10 == 0:
                frame_filename = f"{video_name}_{frame_count}.jpg"
                output_path = os.path.join(output_subdir, frame_filename)
                cv2.imwrite(output_path, frame)

        cap.release()


input_open_eye_dir = 'videos/open_eye_videos'
input_closed_eye_dir = 'videos/closed_eye_videos'
output_dir = 'images'

create_dataset_from_videos(input_open_eye_dir, output_dir, 'open')

create_dataset_from_videos(input_closed_eye_dir, output_dir, 'closed')