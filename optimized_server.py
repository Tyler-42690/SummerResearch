import socket
import time
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import torch

BUFFER_SIZE = 2048

def load_model(mode : str = 'cpu'):
    model = models.squeezenet1_1(pretrained=True)
    model.eval()
    model.to(mode)
    return model

def conversion_to_tensor(img : Image.Image, mode : str = 'cpu'):
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
        bytes_transmitted += len(image_data) 
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
    model = load_model() 
    client = bind()
    start = time.time()
    receive_file("clientfiles/9.png", client)
    start1 = time.time()
    tensor = conversion_to_tensor(Image.open('clientfiles/9.png'))
    file = open('clientfiles/output.txt', 'w')
    start2 = time.time()
    outputs = model(tensor)[0]
    file.write(f"{'pythonimage'}: {torch.nn.functional.softmax(outputs, dim=0)}")
    file.close()
    send_file("clientfiles/output.txt", client)
    client.close()
    print("Runtime: "+ str(time.time()-start)+ " seconds")
    print("Runtime1: "+ str(time.time()-start1)+ " seconds")
    print("Runtime2: "+ str(time.time()-start2)+ " seconds")
if __name__ == '__main__':
    main()