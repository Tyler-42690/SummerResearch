# echo-client.py

import socket

HOST = "192.168.1.154"  # The server's hostname or IP address
PORT = 65432  # The port used by the server
extension = "png"
image_name = "pythonimage."+extension
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
condition = True
s.connect((HOST, PORT))
f = open(s,"wb")
while condition:
    image = s.recv(1024)
    if str(image) == "b''":
        condition = False
    f.write(image)


#print(f"Received {data!r}")