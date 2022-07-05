import socket
import time
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import torch
import numpy as np
import os
import cv2


import config
BUFFER_SIZE = 2048

c = config.Config()
BANDWIDTH_BYTES = float(c.BANDWIDTH_MB) * 2**20
FRAMEWORK = c.FRAMEWORK
MODEL = c.MODEL
MODE = c.INF_MODE

def warmup(model):
    imarray = np.random.rand(*(224,224), 3) * 255
    warmup_image = Image.fromarray(imarray.astype('uint8')).convert('RGB')
    tensor = conversion_to_tensor(warmup_image)
    for i in range(100):
        model(tensor)
    print("Warmup complete.")

def load_model(mode : str = MODE):
    model = models.mobilenet_v2(pretrained=True)
    # model = models.squeezenet1_1(pretrained=True)
    # model = models.alexnet(pretrained=True)
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
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    client = bind()
    start = time.time()
    receive_file("clientfiles/9.png", client)
    upload_timer = time.time()
    tensor = conversion_to_tensor(Image.open('clientfiles/9.png'))
    process_timer = time.time()
    predictions = model(tensor)[0]
    probabilities = torch.nn.functional.softmax(predictions[0], dim=0)
    # Show top categories per image
    top1_prob, top1_catid = torch.topk(probabilities, 1)
    prediction = categories[top1_catid]
    with open('clientfiles/output.txt', 'w') as file:
        file.write(f"{'pythonimage'}: {prediction}")
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