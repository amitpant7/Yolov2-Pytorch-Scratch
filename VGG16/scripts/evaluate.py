import torch 
import sys

from PIL import Image

from config import DEVICE
from config import CLASS_NAMES
from preprocessor import DATA_TRANSFORMS
from VGG16 import custom_VGG16, my_relu


def transform_image(image_path):
   
    """
    Process the image and add the batch dimension
    
    Args:
        image_path (file): file of the image for processing

    Returns:
        image: In the format suitable to fit the model
    """
    img = Image.open(image_path)
    img = DATA_TRANSFORMS(img)
    img = torch.unsqueeze(img, 0)
    img = img.to(DEVICE)
    return img


def evaluate(model_path, image_path):
    
    """Predict the class of image

    Args:
        image_path (file_path): File of image to test
        model_path (.pth): Path of pretrained vgg16 model
    """
    
    img = transform_image(image_path)
    model = torch.load(model_path)
    model.to(DEVICE)
    pred = torch.argmax(model(img))
    print(CLASS_NAMES[pred.item()])
    

def main():
    if len(sys.argv)>2:
        model_pth = sys.argv[1]
        img = sys.argv[2]
        evaluate(model_pth, img)
    
    else:
        print("Provide model path and image")        

if __name__ == "__main__":
    print("Inferencing the model")
    main()


    
    


