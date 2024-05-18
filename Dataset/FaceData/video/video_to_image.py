import cv2
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', help='Path to output_dir', required=True)
    parser.add_argument('--video_path', type=str, help='Path of the video you want to test on.', required=True)

    args = parser.parse_args()
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
    # Lấy FPS của video
    fps = video.get(cv2.CAP_PROP_FPS)
    print(f"FPS của video: {fps}")

    # Số lượng khung hình cần lưu mỗi giây
    frames_to_save_per_second = 4
    frame_interval = int(fps / frames_to_save_per_second)
    print(f"Số khung hình giữa các lần lưu: {frame_interval}")

    frame_count = 0
    saved_frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Chỉ lưu khung hình sau mỗi frame_interval khung hình
        if frame_count % frame_interval == 0:
            image_path = os.path.join(output_folder, f"frame_{saved_frame_count}.jpg")
            cv2.imwrite(image_path, frame)
            saved_frame_count += 1
            print(f"Đã trích xuất {saved_frame_count} khung hình")

        frame_count += 1

    # Đóng video
    video.release()

if __name__ == "__main__":
    main()