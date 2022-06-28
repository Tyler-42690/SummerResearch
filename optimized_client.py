import socket
import cv2
import time

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
    file = open(filename,'rb')
    start = time.time()
    transmitted = 0
    bytes_transmitted = 0
    image_data = file.read(BUFFER_SIZE)
    bytes_transmitted = len(image_data)
    while image_data:
        transmitted += bytes_transmitted
        elapsed = int(time.time()-start)
        if elapsed > 1:
            expected_transmit = BUFFER_SIZE * elapsed
            transmit_delta = transmitted - expected_transmit
            if transmit_delta > 0:
                time.sleep(float(transmit_delta)/BUFFER_SIZE)
                transmitted = 0
                start = time.time()
        s.send(image_data)
        image_data = file.read(BUFFER_SIZE)
    s.send(b"%IMAGE_COMPLETED%")     
    file.close()

def receive_file(filename : str, s : socket.socket):
    file = open(filename,'wb')
    image_chunk = s.recv(BUFFER_SIZE)
    while image_chunk:
        file.write(image_chunk)
        image_chunk = s.recv(BUFFER_SIZE)
        if image_chunk == b"%IMAGE_COMPLETED%":
          break
    file.close()

def main():
    client = connect()
    start = time.time()
    orig_img = cv2.imread('documents/9.png')
    modify_img(orig_img)
    send_file("documents/9.png", client)
    receive_file("documents/output.txt", client)
    client.close()
    print("Runtime :"+ str(time.time()-start)+ " seconds")

if __name__ == '__main__':
    main()