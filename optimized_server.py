import socket
import time
import torchvision.transforms as transforms
import VGG11
import torch
from PIL import Image

BUFFER_SIZE = 2048

def load_model(mode : str):
    model = VGG11.VGG11(in_channels=1, num_classes=10)
    checkpoint = torch.load('model.pth')
# load the trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(mode)
    return model

def conversion_to_tensor(img : Image.Image, mode : str = 'cpu'):
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],
                         std=[0.5])
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
    model = load_model('cpu') 
    tensor = conversion_to_tensor(Image.open('clientfiles/9.png'))
    file = open('clientfiles/output.txt', 'w')
    file.write(f"{'pythonimage'}: {model(tensor)}")
    file.close()
    send_file("clientfiles/output.txt", client)
    client.close()
    print("Runtime: "+ str(time.time()-start)+ " seconds")

if __name__ == '__main__':
    main()