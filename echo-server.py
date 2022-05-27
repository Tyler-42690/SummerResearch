# echo-server.py

import socket
from tkinter import filedialog
import VGG11
import torchvision.transforms as transforms
import cv2
import glob as glob
import numpy as np
import torch
from PIL import Image

# simple image transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],
                         std=[0.5])
])

extension = "png"
image_name = "pythonimage."+extension
def results():
    # get all the test images path
    image_paths = glob.glob(image_name)
    for i, image_path in enumerate(image_paths):
        orig_img = cv2.imread(image_path)
    # convert to grayscale to make the image single channel
        image = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        image = transform(image)
    # add one extra batch dimension
        image = image.unsqueeze(0).to(device)
    # forward pass the image through the model
        outputs = model(image)
    # get the index of the highest score
    # the highest scoring indicates the label for the Digit MNIST dataset
        label = np.array(outputs.detach()).argmax()
        print(f"{image_path.split('/')[-1].split('.')[0]}: {label}")
    # put the predicted label on the original image
        cv2.putText(orig_img, str(label), (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    2, (0, 255, 0), 2)
    # show and save the resutls
        cv2.imshow('Result', orig_img)
        cv2.waitKey(0)
        cv2.imwrite(f"result_{i}."+extension, orig_img)
        return "result_0."+extension

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
condition = True
s.connect((HOST, PORT))
f = open(s,"wb")
s.listen(1)
while condition:
    image = s.recv(1024)
    if str(image) == "b''":
        condition = False
    f.write(image)
# inferencing on CPU
device = 'cpu'
# initialize the VGG11 model
model = VGG11(in_channels=1, num_classes=10)
# load the model checkpoint
checkpoint = torch.load('model.pth')
# load the trained weights
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()



s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
    
s.listen(1)
image = Image(results())
conn, addr = s.accept()
    #with conn:
     #   print(f"Connected by {addr}")
      #  while True:
         #   data = conn.recv(1024)
          #  if not data:
           #     break
          #  conn.sendall(data)
if conn != 0:
    for i in image:
        conn.send(i)

