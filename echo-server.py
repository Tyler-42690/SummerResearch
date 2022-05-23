# echo-server.py

import socket
from tkinter import filedialog

HOST = "192.168.1.154"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
    

data = filedialog.askopenfile(initialdir="/Documents/InitialTable.png")
path = str(data.name)
image = open(path,"rb")

s.listen(1)
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