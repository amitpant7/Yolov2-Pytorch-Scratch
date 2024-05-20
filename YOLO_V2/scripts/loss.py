import torch
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from ..config import DEVICE, C


class YoloV2_Loss(torch.nn.Module):
    """
    YOLOv2 Loss Function

    This class implements the loss function for the YOLOv2 object detection model.
    It includes components for objectness, bounding box regression, and class probabilities.

    Attributes:
        lambda_no_obj (torch.Tensor): Weight for no-object loss.
        lambda_obj (torch.Tensor): Weight for object loss.
        lambda_class (torch.Tensor): Weight for class probability loss.
        lambda_bb_cord (torch.Tensor): Weight for bounding box coordinate loss.
        C (int): Number of classes.
        binary_loss (torch.nn.Module): Binary cross-entropy loss with logits.
        logistic_loss (torch.nn.Module): Cross-entropy loss for class probabilities.
        regression_loss (torch.nn.Module): Mean squared error loss for bounding box regression.
    """

    def __init__(self, C=C, device=DEVICE):
        """
        Initializes the YOLOv2 loss function.

        Args:
            C (int, optional): Number of classes. Defaults to 4.
            device (str, optional): Device to place the tensors on. Defaults to 'cpu'.
        """
        super(YoloV2_Loss, self).__init__()
        self.lambda_no_obj = torch.tensor(1.3, device=device)
        self.lambda_obj = torch.tensor(1.0, device=device)
        self.lambda_class = torch.tensor(1.0, device=device)
        self.lambda_bb_cord = torch.tensor(8.0, device=device)
        self.C = C

        # Loss functions
        self.binary_loss = BCEWithLogitsLoss()  # Binary cross-entropy with logits

        self.logistic_loss = (
            CrossEntropyLoss()
        )  # Cross-entropy loss for class probabilities

        self.regression_loss = (
            MSELoss()
        )  # Mean squared error loss for bounding box regression

    def forward(self, pred, ground_truth):
        """
        Computes the YOLOv2 loss.

        Args:
            pred (torch.Tensor): Predictions from the model. Shape (B, S, S, A*(5+C)).
            ground_truth (torch.Tensor): Ground truth labels. Shape (B, S, S, A*(5+C)).

        Returns:
            torch.Tensor: Total loss.
        """
        # Identify object and no-object cells
        obj = ground_truth[..., 0] == 1
        no_obj = ground_truth[..., 0] == 0

        # No-object loss
        no_obj_loss = self.binary_loss(
            pred[no_obj][..., 0], ground_truth[no_obj][..., 0]
        )

        # Object loss
        obj_loss = self.binary_loss(pred[obj][..., 0], ground_truth[obj][..., 0])

        # Bounding box regression loss
        # Predicted bounding box coordinates: sigmoid for x, y and exp for w, h
        pred_bb = torch.cat(
            (torch.sigmoid(pred[obj][..., 1:3]), torch.exp(pred[obj][..., 3:5])), dim=-1
        )
        
        gt_bb = ground_truth[obj][..., 1:5]
        bb_cord_loss = self.regression_loss(pred_bb, gt_bb)

        # Class probability loss
        pred_prob = pred[obj][..., 5:]

        class_loss = self.logistic_loss(
            pred_prob, ground_truth[obj][..., 5:]
        )  # the classes are one-hot encoded so no .long()

        # Total loss calculation with weighted components
        total_loss = (
            self.lambda_bb_cord * bb_cord_loss
            + self.lambda_no_obj * no_obj_loss
            + self.lambda_obj * obj_loss
            + self.lambda_class * class_loss
        )

        return total_loss
