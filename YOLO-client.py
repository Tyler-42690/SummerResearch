# echo-client.py
import time
import socket

start = time.time()
BUFFER_SIZE = 4096
#HOST = "192.168.1.154"  # The server's hostname or IP address
HOST = "127.0.0.1" #Localhost if needed 
PORT = 4567  # The port used by the server
extension = "png"
image_name = "pythonimage."+extension
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
condition = True #Receives Image
client_socket.connect((HOST, PORT))

with open('documents/9.png', 'rb') as file:
    file_data = file.read(BUFFER_SIZE)
    while file_data:
        client_socket.send(file_data)
        file_data = file.read(BUFFER_SIZE)

client_socket.send(b"%IMAGE_COMPLETED%")

#Receives Modified Image Scored by AI
with open('documents/edited.png', 'wb') as file:
    recv_data = client_socket.recv(BUFFER_SIZE)
    while recv_data:
        file.write(recv_data)
        recv_data = client_socket.recv(BUFFER_SIZE)

        if recv_data == b"%IMAGE_COMPLETED%":
            break
end = time.time()
print("Runtime = "+str(end-start))