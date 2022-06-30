import socket
import cv2
import time
import sys

BUFFER_SIZE = 2048

def connect():
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('localhost',6677))
    return client

def modify_img(img : cv2.Mat):
    #image1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image1 = cv2.resize(img,(224,224))
    cv2.imwrite(f"documents/9.png", image1)

def send_file(filename : str, s : socket.socket):
    with open(filename, 'rb') as file:
        while True:
            image_data = file.read(BUFFER_SIZE)
            s.send(image_data)
            if len(image_data) < BUFFER_SIZE:
                break


def receive_file(filename : str, s : socket.socket):
    with open(filename,'wb') as file:
        while True:
            image_chunk = s.recv(BUFFER_SIZE)
            file.write(image_chunk)
            if len(image_chunk) < BUFFER_SIZE:
                break


def main():
    client = connect()
    # start = time.time()
    orig_img = cv2.imread('documents/9.png')
    modify_img(orig_img)
    send_file("documents/9.png", client)
    receive_file("documents/output.txt", client)
    client.close()
    # print(f"Runtime :{time.time()-start:.04f} seconds")

if __name__ == '__main__':
    main()