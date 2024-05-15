import requests
import base64
from PIL import Image

url = 'http://127.0.0.1:8000/recog'
image_file = 'Dataset/FaceData/new_processed/AnhTuan/frame_453.png'

with open(image_file, 'rb') as f:
    image_data = f.read()
    encoded_image = base64.b64encode(image_data).decode('utf-8')

print(encoded_image)
payload = {
    'image': encoded_image,
    # 'w': w,
    # 'h': h
}

response = requests.post(url, json=payload)
print("res:")
print(response.text)