import cv2
import requests
import numpy as np

# Đường dẫn đến API nhận diện khuôn mặt
url = 'http://172.21.2.101:8000/recog' 

# Khởi tạo webcam
cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read()

    # Chuyển đổi ảnh thành định dạng byte để gửi qua mạng
    _, img_encoded = cv2.imencode('.jpg', frame)
    response = requests.post(url, data=img_encoded.tobytes())

    # Xử lý kết quả trả về (nếu cần)
    if response.status_code == 200:
        result = response.json()
        face_names = result['name']
        face_prob = result['probability']
        face_locations = result['face_pos']
        # print(face_names, face_prob)
        # print("\n")
        # print(face_pos)

    if response.status_code == 200:
        result = response.json()
        face_name = result['name']
        face_prob = result['probability']
        face_locations = result['face_pos']

        # Vẽ hình chữ nhật và tên quanh khuôn mặt
        (left, top, right, bottom) = face_locations
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        text_to_display = f"{face_name}: {face_prob:.2f}" 
        cv2.putText(frame, text_to_display, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


    # Hiển thị khung hình
    cv2.imshow('Nhan Dien Khuon Mat', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()