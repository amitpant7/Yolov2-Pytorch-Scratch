import torch
import math

from utils import mean_average_precision
from ..config import *



def evaluate_model(model, dataloaders=dataloaders, dataset_sizes=dataset_sizes, batch_size=BATCH_SIZE, device=DEVICE, phase = 'val'):
    """
    Evaluates a PyTorch model on the validation dataset.

    Args:
    - model (torch.nn.Module): The PyTorch model to be evaluated.
    - dataloaders (dict): A dictionary containing dataloaders for different phases (e.g., 'train', 'val').
    - dataset_sizes (dict): A dictionary containing the sizes of datasets for different phases.
    - batch_size (int): The batch size for evaluation.
    - device (torch.device): The device (e.g., 'cuda' or 'cpu') on which to perform evaluation.

    Returns:
    - all_preds (torch.Tensor): Predictions made by the model, reshaped for evaluation.
    - all_targets (torch.Tensor): Ground truth labels, reshaped for evaluation.
    """

    model = model.to(device)
    model.eval()
    no_of_batches = math.ceil(dataset_sizes[phase] / batch_size)
    all_preds = torch.empty((no_of_batches, batch_size, S, S, N, C+5))
    all_targets = torch.empty((no_of_batches, batch_size, S, S, N , C+5))

    for i, (image, target) in enumerate(dataloaders[phase]):
        image = image.to(device)
        preds = model(image)

        # if the last batch doesn't have enough images, it will throw error while updating tensor.
        try:
            all_preds[i] = preds.detach().to('cpu')
            all_targets[i] = target.detach().to('cpu')
        except:
            print('Last batch has shape', preds.shape)

    all_preds = all_preds.view(-1, S, S, N, C+5)
    all_targets = all_targets.view((-1, S, S, N, C+5))

    
    map = mean_average_precision(all_preds.to(device), all_targets)
    print('The mean average precision on {phase}:', map.item() )
    return map

