from torch.nn import CrossEntropyLoss
from torch.optim import Adam, lr_scheduler

from scripts.VGG16 import VGG16 
from scripts.make_dataset import make_dataset
from scripts.train import train

def main():
   
   #Loading the data
    dataloader, dataset_sizes = make_dataset(batch_size=64)

    # Initialize VGG16 model
    model = VGG16(num_classes=10)  # Modify based on your implementation

    # Train settings 
    criterion = CrossEntropyLoss()
    optimizer_ft = Adam(model.parameters(), lr= 0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
    # Train the model
    model = train(optimizer_ft, exp_lr_scheduler , criterion, dataloader, dataset_sizes, num_epochs=15)


if __name__ == "__main__":
    main()
