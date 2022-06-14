import socket
import time
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions
from tensorflow.keras.applications import VGG16

BUFFER_SIZE = 2048

def preprocess(filename : str):
    img = image.load_img(filename, color_mode='rgb', target_size=(224,224))
    array = image.img_to_array(img)
    array = np.expand_dims(array, axis = 0)
    array = preprocess_input(array)
    return array

def predictions(array):
    model = VGG16(weights='imagenet')
    features = model.predict(array)
    return decode_predictions(features)


def bind():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('localhost',6677))
    server.listen()
    return server.accept()[0]

def send_file(filename : str, s : socket.socket):
    file = open(filename,'rb')
    image_data = file.read(BUFFER_SIZE)
    while image_data:
        s.send(image_data)
        image_data = file.read(BUFFER_SIZE)
        
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
    client = bind()
    start = time.time()
    receive_file("clientfiles/9.png", client)
    array = preprocess('clientfiles/9.png')
    file = open('clientfiles/output.txt', 'w')
    file.write(f"{'pythonimage'}: {predictions(array)}")
    file.close()
    send_file("clientfiles/output.txt", client)
    client.close()
    print("Runtime: "+ str(time.time()-start)+ " seconds")

if __name__ == '__main__':
    main()