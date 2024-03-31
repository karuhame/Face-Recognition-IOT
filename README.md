# PBL5

# Cài đặt thư viện
pip install -r requirements.txt

# Cài pretrained model
- Link: https://drive.usercontent.google.com/download?id=1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz&export=download&authuser=0
- Tạo folder Models rồi giải nén
  ![image](https://github.com/karuhame/PBL5/assets/62510010/fc0b8c99-6fb1-4647-b8bf-998e86ce0bac)

# Tiền xử lý dữ liệu
python src/align_dataset_mtcnn.py  Dataset/FaceData/raw Dataset/FaceData/processed --image_size 160 --margin 32  --random_order --gpu_memory_fraction 0.25

# Train model
python src/classifier.py TRAIN Dataset/FaceData/processed Models/20180408-102900.pb Models/facemodel.pkl --batch_size 1000

# Test
python src/face_rec_cam.py
