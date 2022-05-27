# echo-client.py
from tkinter import filedialog
import socket

HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 65432  # The port used by the server
extension = "png"
image_name = "pythonimage."+extension
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
condition = True #Receives Image
s.connect((HOST, PORT))
f = open(s,"wb")
data = filedialog.askopenfile(initialdir="/documents/")#Sent Image Directory
path = str(data.name)
image = open(path,"rb")
conn, addr = s.accept() #Sends image
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

s.listen(1)
#Receives Modified Image Scored by AI
while condition:
    image = s.recv(1024)
    if str(image) == "b''":
        condition = False
    f.write(image)
