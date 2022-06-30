import socket
import time
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import torch
import os
import cv2


import config
BUFFER_SIZE = 2048

c = config.Config()
BANDWIDTH_BYTES = int(c.BANDWIDTH_MB) * 2**20
FRAMEWORK = c.FRAMEWORK
MODEL = c.MODEL
MODE = c.INF_MODE

def warmup(model):
    orig_img = cv2.imread('documents/9.png')
    image1 = cv2.resize(orig_img,(224,224))
    cv2.imwrite(f"documents/warmup.png", image1)
    tensor = conversion_to_tensor(Image.open('documents/warmup.png'))
    for i in range(5):
        model(tensor)

def load_model(mode : str = MODE):
    model = models.squeezenet1_1(pretrained=True)
    model.eval()
    model.to(mode)
    #warmup loop for fairness
    warmup(model)
    return model

def conversion_to_tensor(img : Image.Image, mode : str = MODE):
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
    ])  
    img1 = transform(img)
    img1 = img1.unsqueeze(0).to(mode)
    return img1

def bind():
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

        
def main():
    model = load_model() 
    client = bind()
    start = time.time()
    receive_file("clientfiles/9.png", client)
    upload_timer = time.time()
    tensor = conversion_to_tensor(Image.open('clientfiles/9.png'))
    process_timer = time.time()
    outputs = model(tensor)[0]
    with open('clientfiles/output.txt', 'w') as file:
        file.write(f"{'pythonimage'}: {torch.nn.functional.softmax(outputs, dim=0)}")
    inference_timer = time.time()
    send_file("clientfiles/output.txt", client)
    client.close()
    download_timer = time.time()
    upload_delay = os.path.getsize("clientfiles/9.png")/BANDWIDTH_BYTES - (upload_timer-start)
    upload_delay = upload_delay if upload_delay > 0 else 0
    download_delay = os.path.getsize("clientfiles/output.txt")/BANDWIDTH_BYTES - (download_timer-inference_timer)
    download_delay = download_delay if download_delay > 0 else 0
    time.sleep(upload_delay + download_delay)
    print(f"Upload: {(upload_timer - start):.04f} seconds")
    print(f"Upload_artificial: {(upload_delay):.04f} seconds")
    print(f"Processing: {(process_timer - upload_timer):0.4f} seconds")
    print(f"Inference: {(inference_timer - process_timer):0.4f} seconds")
    print(f"Download: {(download_timer - inference_timer):0.4f} seconds")
    print(f"Download_artificial: {(download_delay):0.4f} seconds")
    print(f"Overall: {(time.time() - start):0.4f}")

if __name__ == '__main__':
    main()