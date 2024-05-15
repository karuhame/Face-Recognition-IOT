# PBL5

# Cài đặt thư viện
pip install -r requirements.txt
# Nếu có thiếu thì tiếp tục pip install tới khi chạy đc

# Tiền xử lý dữ liệu
# Cú pháp: python src/align_dataset_mtcnn.py <raw data_dir> <aligned data_dir> <augmentation data_dir> --image_size 160 --margin 32  --random_order --gpu_memory_fraction 0.25

python src/align_dataset_mtcnn.py  Dataset/FaceData/raw Dataset/FaceData/processed Dataset/FaceData/augment_data_10 --image_size 160 --margin 32  --random_order --gpu_memory_fraction 0.25

# Train model
python src/classifier.py TRAIN Dataset/FaceData/augment_data_10 Models/20180402-114759.pb Models/raw_10_img_3_aug.pkl --batch_size 1000

# Test
python src/face_rec_cam.py --modelPath Models/raw_10_img_3_aug.pkl
python src/face_rec.py --modelPath Models/raw_10_img_3_aug.pkl

# Run Flask
python src/face_rec_flask.py

python requestFlask.py 


