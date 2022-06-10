# echo-client.py
import time
import socket
import cv2

BUFFER_SIZE = 4096
#HOST = "192.168.1.154"  # The server's hostname or IP address
HOST = "192.168.1.159" #Raspberry PI
#HOST = "127.0.0.1" #Localhost if needed 
PORT = 4567  # The port used by the server
extension = "png"
image_name = "pythonimage."+extension

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))
orig_img = cv2.imread('documents/9.png',0)
    # convert to grayscale to make the image single channel
        
        
#image1 = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
image1 = cv2.resize(orig_img,(224,224))

cv2.imwrite(f"documents/9.png", image1)

with open('documents/9.png', 'rb') as file:
    
    # add one extra batch dimension
    
    file_data = file.read(BUFFER_SIZE)
    while file_data:
        client_socket.send(file_data)
        file_data = file.read(BUFFER_SIZE)
start = time.time()
client_socket.send(b"%IMAGE_COMPLETED%")

#Receives Modified Image Scored by AI
with open('documents/output.txt', 'wb') as file:
    recv_data = client_socket.recv(BUFFER_SIZE)
    while recv_data:
        file.write(recv_data)
        recv_data = client_socket.recv(BUFFER_SIZE)

        if recv_data == b"%IMAGE_COMPLETED%":
            break
end = time.time()
print("Runtime = "+str(end-start))