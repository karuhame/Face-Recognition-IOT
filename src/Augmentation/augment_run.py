import cv2
import numpy as np
import os
from skimage.util import random_noise



def create_noisy_images(inp_folder, out_folder):
    os.makedirs(out_folder, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại
    
    for foldername in os.listdir(inp_folder):
        folder_path = os.path.join(inp_folder, foldername)
        if not os.path.isdir(folder_path):
            continue  # Bỏ qua nếu không phải là thư mục

        folder_output_path = os.path.join(out_folder, foldername)
        os.makedirs(folder_output_path, exist_ok=True)  # Tạo thư mục đầu ra cho từng thư mục đầu vào
        
        for filename in os.listdir(folder_path):
            # Đường dẫn đầy đủ tới tệp tin đầu vào
            input_path = os.path.join(folder_path, filename)

            # Load ảnh từ tệp tin đầu vào
            image = cv2.imread(input_path, 1)

            # Tạo 10 ảnh với mức độ nhiễu khác nhau
            for i in range(5):
                # Tính mức độ nhiễu cho từng lần lặp
                amount = 0.001 + i * 0.002

                # Thêm nhiễu salt-and-pepper vào ảnh
                noise = random_noise(image, mode='s&p', amount=amount)

                # Chuyển đổi ảnh từ kiểu float trong khoảng [0, 1] sang kiểu 'uint8' trong khoảng [0, 255]
                noise = np.array(255 * noise, dtype=np.uint8)
                
                output_filename = f"noise_{filename}_{i}.jpg"  # Tên file đầu ra duy nhất
                # Đường dẫn đầy đủ tới tệp tin đầu ra
                output_path = os.path.join(folder_output_path, output_filename)  # Đường dẫn đầy đủ tới file đầu ra

                # Lưu ảnh vào tệp tin đầu ra
                cv2.imwrite(output_path, noise)
        
def create_blur_image(inp_folder, out_folder):
    os.makedirs(out_folder, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại
    
    for foldername in os.listdir(inp_folder):
        folder_path = os.path.join(inp_folder, foldername)
        if not os.path.isdir(folder_path):
            continue  # Bỏ qua nếu không phải là thư mục

        folder_output_path = os.path.join(out_folder, foldername)
        os.makedirs(folder_output_path, exist_ok=True)  # Tạo thư mục đầu ra cho từng thư mục đầu vào
        
        for filename in os.listdir(folder_path):
            # Đường dẫn đầy đủ tới tệp tin đầu vào
            input_path = os.path.join(folder_path, filename)

            # Load ảnh từ tệp tin đầu vào
            image = cv2.imread(input_path, 1)

            # Loop để tạo 10 ảnh mờ với các mức độ khác nhau
            for i in range(1, 6):
                # Tạo kernel với numpy
                kernel_size = i + 2
                kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)

                # Áp dụng bộ lọc
                img_blur = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
                output_filename = f"blur_{filename}_{i}.jpg"  # Tên file đầu ra duy nhất
                output_path = os.path.join(folder_output_path, output_filename)  # Đường dẫn đầy đủ tới file đầu ra

                # Ghi ảnh vào file
                cv2.imwrite(output_path, img_blur)
                
def adjust_gamma(image, gamma=1.0):
   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)


def create_ilumination_image(inp_folder, out_folder):
    os.makedirs(out_folder, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại
    
    for foldername in os.listdir(inp_folder):
        folder_path = os.path.join(inp_folder, foldername)
        if not os.path.isdir(folder_path):
            continue  # Bỏ qua nếu không phải là thư mục

        folder_output_path = os.path.join(out_folder, foldername)
        os.makedirs(folder_output_path, exist_ok=True)  # Tạo thư mục đầu ra cho từng thư mục đầu vào
        
        for filename in os.listdir(folder_path):
            # Đường dẫn đầy đủ tới tệp tin đầu vào
            input_path = os.path.join(folder_path, filename)

            # Load ảnh từ tệp tin đầu vào
            image = cv2.imread(input_path, 1)
        
            for i in range(5):
                gamma = 0.5 + i * 0.5  # Giá trị gamma thaiy đổi từ 0.5 đến 3
                adjusted = adjust_gamma(image, gamma=gamma)
                output_filename = f"gamma_{filename}_{i}.jpg"  # Tên file đầu ra duy nhất

                output_path = os.path.join(folder_output_path, output_filename)  # Đường dẫn đầy đủ tới file đầu ra
                cv2.imwrite(output_path, adjusted)

