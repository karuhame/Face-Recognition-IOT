import cv2
import os
import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', help='Path to output_dir', )
    parser.add_argument('--video_path', type=str, help='Path of the video you want to test on.')

    args = parser.parse_args()
    # Đường dẫn đến thư mục lưu ảnh
    output_folder = args.output_dir

    # Tạo thư mục lưu ảnh (nếu chưa tồn tại)
    os.makedirs(output_folder, exist_ok=True)

    # Đường dẫn đến file video
    video_path = args.video_path

    # Mở video
    video = cv2.VideoCapture(video_path)

    # Kiểm tra xem video đã được mở thành công hay chưa
    if not video.isOpened():
        print("Không thể mở video")
        exit()

    # Đọc các khung hình từ video
    frame_count = 0
    while True:
        # Đọc một khung hình từ video
        ret, frame = video.read()

        # Kiểm tra xem việc đọc khung hình đã thành công hay chưa
        if not ret:
            break

        # Tạo đường dẫn lưu ảnh
        image_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")

        # Lưu khung hình thành file ảnh
        cv2.imwrite(image_path, frame)

        # Tăng số lượng khung hình đã đọc
        frame_count += 1

        # Hiển thị thông tin về số lượng khung hình đã được trích xuất
        print(f"Đã trích xuất {frame_count} khung hình")

    # Đóng video
    video.release()
main()