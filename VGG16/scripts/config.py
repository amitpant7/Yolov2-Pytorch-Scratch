from torch import device, cuda

CLASS_NAMES = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
} 
  
DEVICE = device('cuda:0' if cuda.is_available() else 'cpu')