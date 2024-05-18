import requests
import base64
from PIL import Image
import io

url = 'http://127.0.0.1:8000/recog'
image_file = r"E:\1. Bachkhoa\3. Year 3 Seminar 2\3.PBL5\PBL5\Dataset\FaceData\new_raw\ThanhMinh\frame_64.jpg"

with open(image_file, 'rb') as f:
    image_data = f.read()
    encoded_image = base64.b64encode(image_data).decode('utf-8')

image = Image.open(io.BytesIO(image_data))
print("Image size: ", image.size)

payload = {
    'image': encoded_image,
    # 'w': w,
    # 'h': h
}

response = requests.post(url, json=payload)
print("res:")
print(response.text)