import torch
import numpy as np
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from transforms import rev_transform

from config import *
from utils import non_max_suppression, process_preds


def show(imgs):
    """
    Displays a list of images in a grid format.

    Args:
        imgs (list of torch.Tensor): List of images to be displayed.

    Returns:
        None
    """
    total_images = len(imgs)
    num_rows = (total_images + 1) // 2  # Calculate the number of rows
    fig, axs = plt.subplots(nrows=num_rows, ncols=2, squeeze=False, figsize=(12, 12))

    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        row_idx = i // 2
        col_idx = i % 2
        axs[row_idx, col_idx].imshow(np.asarray(img))
        axs[row_idx, col_idx].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.show()


def visualize_bb(samples):
    """
    Visualizes bounding boxes on a list of images.

    Args:
        samples (list of dict): List of samples, each containing an image, bounding boxes, and labels.

    Returns:
        None
    """
    images = []
    for sample in samples:
        img = sample["image"].to("cpu")
        img = rev_transform(img)
        img = (img * 224).to(torch.uint8)
        bboxes = sample["bbox"].to("cpu").numpy()
        labels = sample["labels"].to("cpu")

        _, height, width = img.size()
        colors = [
            (0, 0, 255),
            (255, 0, 0),
            (0, 255, 0),
            (255, 255, 0),
        ]

        corr_bboxes = []
        for bbox in bboxes:
            x, y = bbox[0], bbox[1]  # Center of the bounding box
            box_width, box_height = bbox[2], bbox[3]

            # Calculate the top-left and bottom-right corners of the rectangle
            x1 = int(x - box_width / 2)
            y1 = int(y - box_height / 2)
            x2 = int(x + box_width / 2)
            y2 = int(y + box_height / 2)

            corr_bboxes.append([x1, y1, x2, y2])

        corr_bboxes = torch.tensor(
            corr_bboxes
        )  # Convert to tensor for draw_bounding_boxes
        img_with_bbox = draw_bounding_boxes(
            img, corr_bboxes, colors=[colors[label % 4] for label in labels], width=3
        )
        images.append(img_with_bbox)

    show(images)


def visualize_outputs(
    indices, model, dataset, device=DEVICE, thres=0.9, iou_threshold=0.5
):
    """
    Visualizes the output predictions of the model on a set of images from the dataset.

    Args:
        indices (list of int): List of indices of the images to visualize.
        model (torch.nn.Module): The trained model to use for predictions.
        dataset (torch.utils.data.Dataset): The dataset containing the images and targets.
        device (torch.device): The device (CPU or GPU) to perform computations on.
        thres (float, optional): The threshold for objectness score to filter predictions. Defaults to 0.9.

    Returns:
        None
    """
    images_with_bb = []

    for index in indices:
        # Load the image and target from the dataset
        image, target = dataset[index]
        image = image.to(device)
        model = model.to(device)

        # Set the model to evaluation mode
        model.eval()

        # Get predictions from the model
        preds = model(image.unsqueeze(0))

        # Process the predictions
        preds = process_preds(preds)

        # Filter predictions based on the threshold
        obj = preds[..., 0] > thres

        bboxes = preds[obj][..., 1:5]
        scores = torch.flatten(preds[obj][..., 0])
        _, ind = torch.max(preds[obj][..., 5:], dim=-1)
        classes = torch.flatten(ind)

        # Apply non-max suppression to get the best bounding boxes
        best_boxes = non_max_suppression(bboxes, scores, io_threshold=iou_threshold)

        filtered_bbox = bboxes[best_boxes]
        filtered_classes = classes[best_boxes]

        if filtered_classes.size(0) > 0:
            sample = {
                "image": image.detach().cpu(),
                "bbox": filtered_bbox.detach().cpu(),
                "labels": filtered_classes.detach().cpu(),
            }

            images_with_bb.append(sample)

    # Visualize the bounding boxes on the images
    visualize_bb(
        images_with_bb
    )  # Blue->Buffalo, Red--> Elephant, Green--> Rhino, Yellow--> Zebra
