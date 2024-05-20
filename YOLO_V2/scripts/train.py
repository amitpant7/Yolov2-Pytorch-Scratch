import time
import os
import shutil
import math
import torch

from tqdm.auto import tqdm
from ..config import *
from utils import mean_average_precision

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, batch_size=BATCH_SIZE, num_epochs=5, device=DEVICE):
    """
    Train the model and evaluate its performance.

    Args:
        model (torch.nn.Module): The model to train.
        criterion (callable): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        scheduler (torch.optim.lr_scheduler): The learning rate scheduler.
        dataloaders (dict): Dictionary containing 'train' and 'val' dataloaders.
        dataset_sizes (dict): Dictionary containing sizes of 'train' and 'val' datasets.
        batch_size (int, optional): The batch size. Defaults to BATCH_SIZE.
        num_epochs (int, optional): Number of epochs to train. Defaults to 5.
        device (torch.device, optional): The device to use for training. Defaults to DEVICE.

    Returns:
        torch.nn.Module: The trained model with the best validation mAP.
    """
    since = time.time()
    best_map = 0

    tempdir = 'models/temp'
    os.makedirs(tempdir, exist_ok=True)
    best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                no_of_batches = math.ceil(dataset_sizes[phase] / batch_size)
                all_preds = torch.zeros((no_of_batches, batch_size, S, S, N, C + 5))
                all_targets = torch.zeros((no_of_batches, batch_size, S, S, N, C + 5))

            running_loss = 0.0
            i = 0  # Initialize batch index for 'val' phase

            for inputs, targets in tqdm(dataloaders[phase], leave=False):
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    else:
                        # For mAP calculation
                        try:
                            all_preds[i] = outputs.detach().cpu()
                            all_targets[i] = targets.detach().cpu()
                        except:
                            pass
                        i += 1

                running_loss += loss.item() * inputs.size(0)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f}')

            if phase == 'val':
                all_preds = all_preds.view(-1, S, S, N, C+5)
                all_targets = all_targets.view(-1, S, S, N, C+5)
                mAP = mean_average_precision(all_preds.to(device), all_targets)
                print('Mean Average Precision:', mAP.item())

                if mAP > best_map:
                    best_map = mAP
                    torch.save(model.state_dict(), best_model_params_path)

        print()

    time_elapsed = time.time() - since
    print('Model with Best mAP:', best_map)
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    model.load_state_dict(torch.load(best_model_params_path))
    return model
