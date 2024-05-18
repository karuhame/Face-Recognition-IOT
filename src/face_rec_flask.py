from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask
from flask import render_template , request
from flask_cors import CORS, cross_origin
import tensorflow as tf
import argparse
import facenet
import os
import sys
import math
import pickle
import align.detect_face
import numpy as np
import cv2
import collections
from sklearn.svm import SVC
import base64

from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField, MultipleFileField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired

MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709
IMAGE_SIZE = 182
INPUT_IMAGE_SIZE = 160
CLASSIFIER_PATH = './Models/raw_10_img_3_aug.pkl'
FACENET_MODEL_PATH = './Models/20180402-114759.pb'

# Load The Custom Classifier
with open(CLASSIFIER_PATH, 'rb') as file:
    model, class_names = pickle.load(file)
print("Custom Classifier, Successfully loaded")

tf.Graph().as_default()

# Cai dat GPU neu co
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))


# Load the model
print('Loading feature extraction model')
facenet.load_model(FACENET_MODEL_PATH)

# Get input and output tensors
images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]
pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "src/align")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads'
CORS(app)

class UploadFileForm(FlaskForm):
    files = MultipleFileField("Files", validators=[InputRequired()])
    submit = SubmitField("Upload Files")

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        print("Form: ", form.files.data)
        for file in form.files.data:
            filename = file.filename
            print("name: ", filename)
            file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename)))
            print("YES")
        return "Files have been uploaded."
    return render_template('index.html', form=form)

@app.route('/recog', methods=['POST']) 
@cross_origin()
def upload_img_file():
    if request.method == 'POST':
        data = request.json
        # base 64
        name="Unknown"
        f = data['image']
        # w = int(request.form.get('w'))
        # h = int(request.form.get('h'))
        # print(w)
        # print(h)

        decoded_string = base64.b64decode(f)
        frame = np.fromstring(decoded_string, dtype=np.uint8)      
        # # Hiển thị hình ảnh cropped
        # cv2.imshow("Frame", frame)
        # cv2.waitKey(0)

        # # Đóng cửa sổ hiển thị
        # cv2.destroyWindow("Frame") 
        
        frame = cv2.imdecode(frame, cv2.IMREAD_ANYCOLOR)  # cv2.IMREAD_COLOR in OpenCV 3.1
        cv2.imshow("Frame", frame)
        cv2.waitKey(0)

        # Đóng cửa sổ hiển thị
        cv2.destroyWindow("Frame") 
        
        print("frame shape: ", frame.shape)
        #frame = frame.reshape(w,h,3)
        bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

        faces_found = bounding_boxes.shape[0]

        if faces_found > 0:
            det = bounding_boxes[:, 0:4]
            bb = np.zeros((faces_found, 4), dtype=np.int32)
            for i in range(faces_found):
                bb[i][0] = det[i][0]
                bb[i][1] = det[i][1]
                bb[i][2] = det[i][2]
                bb[i][3] = det[i][3]
                
                cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                cv2.imshow("Image", cropped)
                cv2.waitKey(0)

                # # Đóng cửa sổ hiển thị
                cv2.destroyWindow("Image")
                scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                    interpolation=cv2.INTER_CUBIC)
                scaled = facenet.prewhiten(scaled)
                scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                emb_array = sess.run(embeddings, feed_dict=feed_dict)
                
                print("Emb_array: ", emb_array.shape)
                
                # Dua vao model de classifier
                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[
                    np.arange(len(best_class_indices)), best_class_indices]
                
                # Lay ra ten va ty le % cua class co ty le cao nhat
                best_name = class_names[best_class_indices[0]]
                print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))

                if best_class_probabilities > 0.6:
                    name = class_names[best_class_indices[0]]
                else:
                    name = "Unknown"


        return name


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port='8000')

