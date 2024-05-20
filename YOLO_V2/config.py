import torch

NO_OF_ANCHOR_BOX = N = 5
S = 13  #No of girds in Yolov2
NO_OF_CLASS = C =  4
HEIGHT = H = 416
WIDTH = W = 416
SCALE = 32

DEVICE =device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = batch_size = 8


ANCHOR_BOXES = A = [[ 5.3623,  8.1648],
        [ 2.6154,  3.6588],
        [ 5.8350,  7.8299],
        [ 2.7199,  3.6921],
        [ 9.7198, 10.3616]]  #Calculated using K-means on the African Animals dataset