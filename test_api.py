 import urllib.request
import urllib.parse
import json
import os
import cv2
import numpy as np
import mimetypes

# Create a dummy image
img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
cv2.imwrite("dummy.jpg", img)

# 1. Upload
boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
with open('dummy.jpg', 'rb') as f:
    img_data = f.read()

body = (
    f'--{boundary}\r\n'
    f'Content-Disposition: form-data; name="file"; filename="dummy.jpg"\r\n'
    f'Content-Type: image/jpeg\r\n\r\n'
).encode('utf-8') + img_data + f'\r\n--{boundary}--\r\n'.encode('utf-8')

req = urllib.request.Request('http://localhost:8000/api/upload', data=body)
req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
try:
    with urllib.request.urlopen(req) as response:
        res = json.loads(response.read().decode())
        print("Upload Response:", res)
        image_id = res['image_id']
except urllib.error.HTTPError as e:
    print("Upload Failed:", e.read().decode())
    exit(1)

# 2. Analyze
data = urllib.parse.urlencode({'image_id': image_id, 'pyramid_levels': 4}).encode('utf-8')
req2 = urllib.request.Request('http://localhost:8000/api/analyze', data=data)
try:
    with urllib.request.urlopen(req2) as response:
        res = json.loads(response.read().decode())
        print("Analyze Success! Keys:", res.keys())
except urllib.error.HTTPError as e:
    print("Analyze Failed:", e.read().decode())
