from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

DATA_TRANSFORMS = Compose([
        Resize(224),
        CenterCrop(224),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
   