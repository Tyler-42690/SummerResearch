# echo-server.py
import time
import socket
#import io
import numpy as np
#from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions
from tensorflow.keras.applications import VGG16

extension = "png"
image_name = "pythonimage."+extension
    
#HOST = "192.168.1.154"  # Standard loopback interface address (localhost)
#HOST = "127.0.0.1" #Localhost
HOST = '0.0.0.0'
PORT = 6677  # Port to listen on (non-privileged ports are > 1023)
BUFFER_SIZE = 4096

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen()

client, addr = server.accept()
file = open(image_name,"wb")
recv_data = client.recv(BUFFER_SIZE)
start = time.time()   
while recv_data:
    file.write(recv_data)
    recv_data = client.recv(BUFFER_SIZE)

    #if recv_data == b"%IMAGE_COMPLETED%":
     #     break
file.close()
#image2 = Image.open(file_stream)
#image2.save('clientfiles/'+image_name, format = 'PNG')

#end = time.time() 

#print("Total time of initial data retrieval: " + str(end-start)+" seconds.")

#start1 = time.time()

#model = models.vgg19_bn(pretrained = False)
model = VGG16(weights='imagenet')
#model.eval()
#start3 = time.time()

img = image.load_img('clientfiles/pythonimage.png', color_mode='rgb', target_size=(224,224))
array = image.img_to_array(img)
array = np.expand_dims(array, axis = 0)
array = preprocess_input(array)


#end = time.time()
#print("Model pre-processing time: " + str(end-start3)+ " seconds.")
    # forward pass the image through the model
#start4 = time.time()
features = model.predict(array)
prediction = decode_predictions(features)
#end = time.time()
#print("Model processing time: " + str(end-start4)+ " seconds.")

#start2 = time.time()
file = open('clientfiles/output.txt', 'w')
file.write(f"{'pythonimage'}: {prediction}")
file.close()

with open('clientfiles/output.txt', 'rb') as file:
    file_data = file.read(BUFFER_SIZE)
    while file_data:
        client.send(file_data)
        file_data = file.read(BUFFER_SIZE)
file.close()
client.close()
server.close()
#end = time.time()
#print("Text file transfer time: " + str(end-start2)+" seconds.")
print("Runtime: "+str(time.time()-start)+" seconds.")