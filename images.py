import cv2
import os


# Функция для разделения видео на кадры и аннотирования
def preprocess_videos(video_folder, output_folder, annotations):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for video_file in os.listdir(video_folder):
        if video_file.endswith('.mp4'):
            video_path = os.path.join(video_folder, video_file)
            vidcap = cv2.VideoCapture(video_path)
            success, image = vidcap.read()
            count = 0

            while success:

                frame_path = os.path.join(output_folder, f'{video_file[:-4]}_{count}.jpg')
                cv2.imwrite(frame_path, image)
                annotation = annotations[video_file]
                success, image = vidcap.read()
                count += 1

            vidcap.release()

video_folder = 'videos'
output_folder = 'images'

annotations = {'working.mp4': 'work', 'sleeping.mp4': 'sleep'}

preprocess_videos(video_folder, output_folder, annotations)