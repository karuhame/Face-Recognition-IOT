<!-- PROJECT TITLE -->

# Face Recognition IoT

> Real-time facial recognition pipeline that prepares data, trains a FaceNet-based classifier and exposes recognition over HTTP for IoT devices.

A PBL5 (Project-Based Learning) assignment focused on bringing state-of-the-art face recognition to resource-constrained devices. The project combines **TensorFlow / Keras**, **MTCNN**, **FaceNet** embeddings, an **SVM** classifier and a lightweight **Flask** REST API. Example clients are provided for both webcam and remote devices.

---

## Table of Contents

- [Face Recognition IoT](#face-recognition-iot)
  - [Table of Contents](#table-of-contents)
  - [About The Project](#about-the-project)
    - [Built With](#built-with)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Usage](#usage)
    - [1️⃣ Data preparation](#1️⃣-data-preparation)
    - [2️⃣ Model training](#2️⃣-model-training)
    - [3️⃣ Inference](#3️⃣-inference)
  - [Project Structure](#project-structure)
  - [Acknowledgements](#acknowledgements)

---

## About The Project

The repository provides an end-to-end workflow:

1. **Data alignment & augmentation** using MTCNN.
2. **Embedding extraction** via FaceNet pre-trained weights.
3. **Classifier training** (SVM / Softmax / Triplet-loss options).
4. **Real-time inference** from webcam or via a REST endpoint.

The result is a compact model that can be deployed on edge devices such as Raspberry Pi or NVIDIA Jetson.

### Built With

- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [scikit-learn](https://scikit-learn.org/)
- [OpenCV](https://opencv.org/)
- [Flask](https://flask.palletsprojects.com/)
- [MTCNN](https://github.com/ipazc/mtcnn)

---

## Getting Started

### Prerequisites

- Python ≥ 3.8
- pip (comes with Python)
- (Optional) NVIDIA GPU + CUDA ≥ 11 for faster training

```bash
# Clone the repository
$ git clone https://github.com/<your_username>/Face-Recognition-IOT.git
$ cd Face-Recognition-IOT

# Install Python dependencies
$ pip install -r requirements.txt
```

### Installation

1. Download the FaceNet checkpoint `20180402-114759.pb` and place it inside the `Models/` directory (already committed for convenience).
2. (Optional) Prepare a custom dataset following `Dataset/FaceData/raw` structure.

---

## Usage

### 1️⃣ Data preparation

Align, crop and (optionally) augment raw images:

```bash
python src/align_dataset_mtcnn.py \
  Dataset/FaceData/raw \
  Dataset/FaceData/processed \
  Dataset/FaceData/augment_data_10 \
  --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.25
```

### 2️⃣ Model training

Train an SVM classifier on the generated embeddings:

```bash
python src/classifier.py TRAIN \
  Dataset/FaceData/augment_data_10 \
  Models/20180402-114759.pb \
  Models/raw_10_img_3_aug.pkl \
  --batch_size 1000
```

### 3️⃣ Inference

1. **Webcam (desktop)**
   ```bash
   python src/face_rec_cam.py --modelPath Models/raw_10_img_3_aug.pkl
   ```
2. **Flask REST API (edge / cloud)**
   ```bash
   python src/face_rec_flask.py  # default http://0.0.0.0:8000
   ```
3. **Sample client**
   ```bash
   python requestFlask.py        # streams webcam frames to the API
   ```

The API expects a base64-encoded JPEG under the `image` field and returns JSON:

```json
{
  "name": "John_Doe",
  "probability": 0.92,
  "face_pos": [left, top, right, bottom]
}
```

---

## Project Structure

```
Face-Recognition-IOT/
├─ Dataset/              # Raw, processed & augmented data
├─ Models/               # Pre-trained FaceNet & trained classifiers
├─ src/                  # Source code
│  ├─ align/             # MTCNN utilities
│  ├─ templates/         # Flask HTML templates
│  └─ ...
├─ test/                 # Unit / integration tests
├─ video/                # Demo videos
└─ README.md
```

Don’t forget to give the project a ⭐ if you like it!

---

## Acknowledgements

- [othneildrew/Best-README-Template](https://github.com/othneildrew/Best-README-Template) – inspiration for this README
- [David Sandberg – FaceNet](https://github.com/davidsandberg/facenet)
