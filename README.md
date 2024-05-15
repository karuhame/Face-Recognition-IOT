# PBL5

# Cài đặt thư viện
pip install -r requirements.txt

# Tiền xử lý dữ liệu
python src/align_dataset_mtcnn.py  Dataset/FaceData/raw Dataset/FaceData/processed --image_size 160 --margin 32  --random_order --gpu_memory_fraction 0.25

# Train model
python src/classifier.py TRAIN Dataset/FaceData/processed Models/20180408-102900.pb Models/facemodel.pkl --batch_size 1000

# Test
python src/face_rec_cam.py 
