import cv2
import time
from tensorflow.keras.models import load_model

# Загрузка модели для определения состояния глаз
eye_model = load_model('eye_model.h5')

# Загрузка каскадов для обнаружения лиц и глаз
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Инициализация видеопотока
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

# Переменные
start_time = time.time()
work_timer = 0
sleep_timer = 0
total_work_time = 0
total_sleep_time = 0
closed_eyes_counter = 0
sleeping = False
sleep_threshold = 10  # Время для определения сна (в секундах)
closed_eyes_threshold = 5  # Порог для определения закрытых глаз (в секундах)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Преобразование кадра в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц на кадре
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:

        # Область интереса - лицо
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Обнаружение глаз на лице
        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) == 0:
            # Увеличение счетчика закрытых глаз
            closed_eyes_counter += 1

            # Если глаза были закрыты достаточно долго, считаем, что человек засыпает
            if closed_eyes_counter >= closed_eyes_threshold * 60:
                sleeping = True
                sleep_timer += time.time() - start_time
                total_sleep_time += time.time() - start_time
                cv2.putText(frame, 'Sleeping', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                cv2.putText(frame, 'Eyes Closed', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            work_timer = 0
        else:
            # Сброс счетчика закрытых глаз
            closed_eyes_counter = 0

            # Обнаружены открытые глаза, сброс таймера сна
            sleeping = False
            sleep_timer = 0
            total_work_time += time.time() - start_time
            cv2.putText(frame, 'Eyes Open', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Отображение времени работы и времени сна
        cv2.putText(frame, f'Total Work Time: {total_work_time:.2f} s', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f'Total Sleep Time: {total_sleep_time:.2f} s', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Отрисовка прямоугольника вокруг лица
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('frame', frame)
    start_time = time.time()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()