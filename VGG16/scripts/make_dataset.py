
from torchvision.datasets import CIFAR10
from torch.utils import data
from preprocessor import DATA_TRANSFORMS
from config import CLASS_NAMES
 

def make_dataset(data_transforms = DATA_TRANSFORMS, batch_size = 64):
    train_set = CIFAR10(root = './Data', download = True, 
                                          train = True, transform = data_transforms)

    test_set =  CIFAR10(root = '/', download = True, 
                                            train = False, transform = data_transforms)

    train_set = data.Subset(train_set, range(5000))
    test_set = data.Subset(train_set, range(1000)) 

    print('The size of dataset is :', len(train_set), len(test_set))


    train_loader =  data.DataLoader(train_set, batch_size,  shuffle = True)
    test_loader = data.DataLoader(test_set, batch_size, shuffle = False, )

    
    #Making things easy, so storing it in a dictionary 
    dataloader = {
        'train': train_loader,
        'val': test_loader
    }

    dataset_sizes = {
        "train": len(train_set),
        "val": len(test_set)
    }
    
    return dataloader, dataset_sizes
