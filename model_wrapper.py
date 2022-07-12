import config
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, models

c = config.Config()
selected_model = c.MODEL 
selected_mode = c.INF_MODE
image_size = (224,224)

if selected_model == "AlexNet":
    selected_model ="alexnet"
elif selected_model == "SqueezeNet":
    selected_model= "squeezenet1_1"
elif selected_model == "MobileNet":
    selected_model ="mobilenet_v3s"
elif selected_model == "VGG":
    selected_model = "vgg16"

model = torch.hub.load('pytorch/vision:v0.11.0', selected_model, pretrained=True)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class Model:
    def __init__(self,) -> None:
        global model 
        model.eval()
        if torch.cuda.is_available():
            model.to(selected_mode)
        with open("imagenet_classes.txt", "r") as f:
            self.categories = [s.strip() for s in f.readlines()]
        self.warmup()
        self.model = model


    def predict(self, img):
        if isinstance(img, Image.Image):
            if img.size != image_size:
                img = img.resize(image_size)
        else:
            img = Image.load_img(img, target_size=image_size)
        input_tensor = preprocess(img)
        x = input_tensor.unsqueeze(0)
        input_batch = x.to('cpu')
        if torch.cuda.is_available():
            input_batch = x.to(selected_mode)
        with torch.no_grad():
            predictions = model(input_batch)
        probabilities = torch.nn.functional.softmax(predictions[0], dim=0)
        # Show top categories per image
        top1_prob, top1_catid = torch.topk(probabilities, 1)
        prediction = self.categories[top1_catid]
        return prediction

    
    def warmup(self, iterations = 100):
        imarray = np.random.rand(*image_size, 3) * 255
        for i in range(iterations):
            warmup_image = Image.fromarray(imarray.astype('uint8')).convert('RGB')
            _ = self.predict(warmup_image)
        print("Warmup Complete.")