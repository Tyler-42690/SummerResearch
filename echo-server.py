# echo-server.py
import time
import socket
import io
import numpy as np
from PIL import Image
import torch
import VGG11
import torchvision.transforms as transforms

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

image2 = Image.open(file_stream)
#image = image.filter(ImageFilter.GaussianBlur(radius=10)) Line caused AI error
image2.save('clientfiles/'+image_name, format = 'PNG')

end = time.time() 

print("Total time of initial data retrieval: " + str(end-start)+" seconds.")

start1 = time.time()

# initialize the VGG11 model
model = VGG11.VGG11(in_channels=1, num_classes=10)
# load the model checkpoint
checkpoint = torch.load('model.pth')
# load the trained weights
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
start3 = time.time()

transform = transforms.ToTensor()
tensor = transform(image2)
tensor = tensor.unsqueeze_(0)
print("Shape of tensor: " + str(tensor.shape))

end = time.time()
print("Model pre-processing time: " + str(end-start3)+ " seconds.")
    # forward pass the image through the model
start4 = time.time()

outputs = model(tensor)
end = time.time()
print("Model processing time: " + str(end-start4)+ " seconds.")
label = np.array(outputs.detach()).argmax() # get the index of the highest score

start2 = time.time()
with open('clientfiles/output.txt', 'w') as file:
    file.write(f"{'pythonimage'}: {label}")
file.close()

with open('clientfiles/output.txt', 'rb') as file:
    file_data = file.read(BUFFER_SIZE)
    while file_data:
        client.send(file_data)
        file_data = file.read(BUFFER_SIZE)
client.send(b"%IMAGE_COMPLETED%")
file.close()

client.close()
server.close()
end = time.time()
print("Text file transfer time: " + str(end-start2)+" seconds.")
print("Runtime: "+str(end-start)+" seconds.")