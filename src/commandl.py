import subprocess

def preprocessed_img():
    file_path = "E:/1. Bachkhoa/3. Year 3 Seminar 2/3.PBL5/PBL5/src/align_dataset_mtcnn.py"
    raw_data_path = "E:/1. Bachkhoa/3. Year 3 Seminar 2/3.PBL5/PBL5/Dataset/FaceData/raw"
    process_data_path = "E:/1. Bachkhoa/3. Year 3 Seminar 2/3.PBL5/PBL5/Dataset/FaceData/processed"
    augment_data_path = "E:/1. Bachkhoa/3. Year 3 Seminar 2/3.PBL5/PBL5/Dataset/FaceData/augment_data_10"
    # python_script_path = file_path +' '+ raw_data_path +' '+ process_data_path +' '+ augment_data_path + ' --image_size 160 --margin 32  --random_order --gpu_memory_fraction 0.25'
    additional_args = [raw_data_path, process_data_path, augment_data_path, "--image_size", "160",  "--margin",  "32",  "--random_order", "--gpu_memory_fraction",  "0.25", "--augment", "True"]
    
    subprocess.run(["python", file_path] + additional_args, shell=True)

def train(data_dir, pretrain_model, classify_model, batch_size = 1000, use_split_dataset = False, nrof_train_images_per_class = 10):
    file_path = "src/classifier.py"
    additional_args =['TRAIN', data_dir, pretrain_model, classify_model, 
                      '--batch_size', str(batch_size)]
    
    subprocess.run(["python", file_path] + additional_args, shell=True)

def classify(data_dir, pretrain_model, classify_model, batch_size = 1000, use_split_dataset = False, nrof_train_images_per_class = 10):
    file_path = "src/classifier.py"
    additional_args =['CLASSIFY', data_dir, pretrain_model, classify_model, 
                      '--batch_size', str(batch_size)]
    
    subprocess.run(["python", file_path] + additional_args, shell=True)

    
def main():
    # preprocessed_img()
    new_processed_dir = "Dataset/FaceData/new_processed"
    data_dir = "Dataset/FaceData/augment_data_5"
    pretrain_model = "Models/20180402-114759.pb"
    classify_model = "Models/DTC_10.pkl"
    # train(data_dir, pretrain_model, classify_model)
    classify(new_processed_dir, pretrain_model, classify_model)
main()