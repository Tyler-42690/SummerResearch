# echo-server.py
import time
import sys
import socket
import io
from tkinter import filedialog
import VGG11
import torchvision.transforms as transforms
import cv2
import glob as glob
import numpy as np
import torch
from PIL import Image, ImageFilter

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

    
#HOST = "192.168.1.154"  # Standard loopback interface address (localhost)
HOST = "127.0.0.1" #Localhost
PORT = 4567  # Port to listen on (non-privileged ports are > 1023)
BUFFER_SIZE = 4096


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen()


client, addr = server.accept()
file_stream = io.BytesIO()
recv_data = client.recv(BUFFER_SIZE)
start = time.time()   
while recv_data:
    file_stream.write(recv_data)
    recv_data = client.recv(BUFFER_SIZE)

    if recv_data == b"%IMAGE_COMPLETED%":
          break

image = Image.open(file_stream)
#image = image.filter(ImageFilter.GaussianBlur(radius=10)) Line caused AI error
image.save('clientfiles/'+image_name, format = 'PNG')

end = time.time() 

print("Total time of initial data retrieval: " + str(end-start)+" seconds.")


start1 = time.time()
# inferencing on CPU
device = 'cpu'
# initialize the VGG11 model
model = VGG11.VGG11(in_channels=1, num_classes=10)
# load the model checkpoint
checkpoint = torch.load('model.pth')
# load the trained weights
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

orig_img = cv2.imread('clientfiles/pythonimage.png')
    # convert to grayscale to make the image single channel
        
        
image1 = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
image1 = transform(image1)
    # add one extra batch dimension
image1 = image1.unsqueeze(0).to(device)
    # forward pass the image through the model
outputs = model(image1)
end = time.time()
print("Model processing time: " + str(end-start1)+ " seconds.")
    # get the index of the highest score
    # the highest scoring indicates the label for the Digit MNIST dataset
label = np.array(outputs.detach()).argmax()
original_stdout = sys.stdout # Save a reference to the original standard output
start2 = time.time()
with open('clientfiles/output.txt', 'w') as file:
    sys.stdout = file
    print(f"{'pythonimage'}: {label}")
    sys.stdout = original_stdout
file.close()
    # put the predicted label on the original image
#cv2.putText(orig_img, str(label), (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 
              #  2, (0, 255, 0), 2)
    # show and save the resutls
#@cv2.imshow('Result', orig_img)
#cv2.waitKey(0)
#cv2.imwrite(f"clientfiles/edited.png", orig_img)

with open('clientfiles/output.txt', 'rb') as file:
    file_data = file.read(BUFFER_SIZE)
    while file_data:
        client.send(file_data)
        file_data = file.read(BUFFER_SIZE)
client.send(b"%IMAGE_COMPLETED%")
file.close()

    #with conn:
     #   print(f"Connected by {addr}")
      #  while True:
         #   data = conn.recv(1024)
          #  if not data:
           #     break
          #  conn.sendall(data)
#client.close()
#server.close()
end = time.time()
print("Text file transfer time: " + str(end-start2)+" seconds.")
print("Runtime: "+str(end-start)+" seconds.")