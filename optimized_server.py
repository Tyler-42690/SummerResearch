import socket
import time
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import torch
import numpy as np
import os
import cv2

import numpy as np
import config
BUFFER_SIZE = 2048

c = config.Config()
BANDWIDTH_BYTES = float(c.BANDWIDTH_MB) * 2**20
FRAMEWORK = c.FRAMEWORK
MODEL = c.MODEL
MODE = c.INF_MODE

warmup_count = 100 if MODE = 'cuda' else 1
server = None

def warmup(model):
    print("Warming up.")
    imarray = np.random.rand(*(224, 224), 3) * 255
    # orig_img = cv2.imread('documents/9.png')
    # image1 = cv2.resize(orig_img,(224,224))
    # cv2.imwrite(f"documents/warmup.png", image1)
    tensor = conversion_to_tensor(Image.fromarray(imarray.astype('uint8')).convert('RGB'))
    for i in range(warmup_count):
        _ = model(tensor)

def load_model(mode : str = MODE):
    # model = models.squeezenet1_1(pretrained=True)
    # model = models.mobilenet_v3_small(pretrained=True)
    model = models.vgg16(pretrained=True)
    # model = models.alexnet(pretrained=True)
    model.eval()
    model.to(mode)
    #warmup loop for fairness
    warmup(model)
    print("Ready for Client.")
    return model

def conversion_to_tensor(img : Image.Image, mode : str = MODE):
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
    ])  
    img1 = transform(img)
    return img1.unsqueeze(0).to(mode)
    
def bind():
    global server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('localhost',6677))
    server.listen()
    return server.accept()[0]

def send_file(filename : str, s : socket.socket):
    with open(filename,'rb') as file:
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

        
model = None
client = None
categories = None

def main():
    global server
    global model
    global client
    global categories
    model = load_model()
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    client = bind()
    start = time.time()
    
    
    receive_file("clientfiles/9.png", client)
    upload_timer = time.time()
    tensor = conversion_to_tensor(Image.open('clientfiles/9.png'))
    process_timer = time.time()
    # outputs = None
    # for i in range(100):
    #     outputs = model(tensor)[0]
    # with open('clientfiles/output.txt', 'w') as file:
    #     file.write(f"{'pythonimage'}: {torch.nn.functional.softmax(outputs, dim=0)}")
    # inference_timer = time.time()
    (prediction, process_time, inference_time) = test_loop()
    with open('clientfiles/output.txt', 'w') as file:
        file.write(f"{'pythonimage'}: {prediction}")
    send_file("clientfiles/output.txt", client)
    # client.close()
    # if server:
    # #     server.close() # quick hack to make this work
    # download_timer = time.time()
    # upload_delay = os.path.getsize("clientfiles/9.png")/BANDWIDTH_BYTES - (upload_timer-start)
    # upload_delay = upload_delay if upload_delay > 0 else 0
    # download_delay = os.path.getsize("clientfiles/output.txt")/BANDWIDTH_BYTES - (download_timer-inference_timer)
    # download_delay = download_delay if download_delay > 0 else 0
    # time.sleep(upload_delay + download_delay)
    
    # print(f"Upload: {(upload_timer - start):.04f} seconds")
    # print(f"Upload_artificial: {(upload_delay):.04f} seconds")
    print(f"Processing: {(process_time):0.4f} seconds")
    print(f"Inference: {(inference_time):0.4f} seconds")
    # print(f"Download: {(download_timer - inference_timer):0.4f} seconds")
    # print(f"Download_artificial: {(download_delay):0.4f} seconds")
    # print(f"Overall: {(time.time() - start):0.4f}")

def test_loop():
    global model
    tensor = None
    upload_timer = time.time()
    for i in range(100):
        tensor = conversion_to_tensor(Image.open('clientfiles/9.png'))
    process_timer = time.time()
    predictions = None
    for i in range(100)):
        predictions = model(tensor)[0]
    probabilities = torch.nn.functional.softmax(predictions[0], dim=0)
    # Show top categories per image
    top1_prob, top1_catid = torch.topk(probabilities, 1)
    prediction = categories[top1_catid]
    inference_timer = time.time()
    return (prediction, (process_timer-upload_timer)/100, (inference_timer-process_timer)/100)

if __name__ == '__main__':
    main()
