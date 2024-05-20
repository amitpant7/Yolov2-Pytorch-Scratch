import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.io import read_image
from torchvision import tv_tensors

from ..config import *


class AfricanWildlifeDataset(Dataset):
    def __init__(self, rootdir, transform=None):
        """
        Initializes the dataset.

        Parameters:
        - rootdir (str): The root directory containing image folders.
        - transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = rootdir
        self.image_paths = []
        self.transform = transform

        class_names = os.listdir(self.root_dir)
        for directory in class_names:
            files = os.listdir(os.path.join(self.root_dir, directory))
            self.image_paths += [
                os.path.join(directory, file)
                for file in files
                if os.path.splitext(file)[1] == ".jpg"
            ]

    def __len__(self):
        """
        Returns the total number of images.

        Returns:
        - int: Number of images in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Fetches the sample and target for the given index.

        Parameters:
        - idx (int): Index of the sample to fetch.

        Returns:
        - tuple: (image, target) where target is the formatted target for YOLOv2.
        """
        sample = self._make_sample(idx)
        img, labels, bboxes = sample["image"], sample["labels"], sample["bbox"]
        _, height, width = img.size()

        target = self._make_target(bboxes, labels, height, width)

        return img, target

    def _make_sample(self, idx):
        """
        Creates a sample formatted as a dictionary.

        Parameters:
        - idx (int): Index of the sample to create.

        Returns:
        - dict: A dictionary with 'image', 'labels', and 'bbox'.
        """
        img_path = os.path.join(self.root_dir, self.image_paths[idx])
        target_path = os.path.splitext(img_path)[0] + ".txt"
        img = read_image(img_path)
        _, height, width = img.size()
        bbox = []
        labels = []

        with open(target_path, "r") as f:
            data = f.readlines()
            for line in data:
                values = line.split()
                labels.append(int(values[0]))
                temp_bbox = [float(val) for val in values[1:]]

                x, y = (
                    temp_bbox[0] * width,
                    temp_bbox[1] * height,
                )  # center of the bounding box
                box_width, box_height = temp_bbox[2] * width, temp_bbox[3] * height
                bbox += [[x, y, box_width, box_height]]

        # Converting the bboxes into pytorch bbox tensor
        bboxes = tv_tensors.BoundingBoxes(
            bbox, format="CXCYWH", canvas_size=img.shape[-2:]
        )

        sample = {"image": img, "labels": torch.tensor(labels), "bbox": bboxes}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _make_target(self, bboxes, labels, height, width):
        """
        Formats the target in YOLOv2 format.

        Parameters:
        - bboxes (tv_tensors.BoundingBoxes): Bounding boxes of the objects.
        - labels (torch.Tensor): Labels of the objects.
        - height (int): Height of the image.
        - width (int): Width of the image.

        Returns:
        - torch.Tensor: The formatted target tensor.
        """

        target = torch.zeros(S, S, NO_OF_ANCHOR_BOX, 1 + 4 + C)  # S*S*N, 1+4+C

        to_exclude = []
        for bbox, label in zip(bboxes, labels):
            cx, cy = bbox[0] / SCALE, bbox[1] / SCALE  # Float values
            pos = (int(cx), int(cy))
            pos = min(pos[0], 12), min(pos[1], 12)
            bx, by = cx - int(cx), cy - int(cy)
            box_width, box_height = bbox[2] / SCALE, bbox[3] / SCALE

            assigned_anchor_box = self.match_anchor_box(
                box_width, box_height, to_exclude
            )
            anchor_box = ANCHOR_BOXES[assigned_anchor_box]

            bw_by_Pw, bh_by_ph = box_width / anchor_box[0], box_height / anchor_box[1]
            target[pos[0], pos[1], assigned_anchor_box, 0:5] = torch.tensor(
                [1, bx, by, bw_by_Pw, bh_by_ph]
            )
            target[pos[0], pos[1], assigned_anchor_box, 5 + int(label)] = 1

            to_exclude.append(assigned_anchor_box)

        return target

    def inverse_target(self, ground_truth, S=S, SCALE=SCALE, anchor_boxes=ANCHOR_BOXES):
        """
        Converts the target tensor back to bounding boxes and labels.

        Parameters:
        - ground_truth (torch.Tensor): The ground truth tensor.
        - S (int, optional): The size of the grid. Default is 13.
        - SCALE (int, optional): The scale factor. Default is 32.
        - anchor_boxes (list, optional): List of anchor boxes. Default is None.

        Returns:
        - tuple: (bbox, labels) where bbox are the bounding boxes and labels are the object labels.
        """
        bboxes = []
        labels = []
        ground_truth = ground_truth.to(device)

        cx = cy = (torch.tensor([i for i in range(13)], device=device),)

        # for getting the center point of pred bb, bx = sig(tx)+cx in paper
        ground_truth = ground_truth.permute(0, 3, 4, 2, 1)
        ground_truth[..., 1:2, :, :] += cx
        ground_truth = ground_truth.permute(0, 1, 2, 4, 3)
        ground_truth[..., 2:3, :, :] += cy
        ground_truth = ground_truth.permute((0, 3, 4, 1, 2))  # bakck to B,13,13,5,9

        ground_truth[..., 1:3] *= 32  # to pixels

        # Calculating the height and width in pixels

        # anchor_matrix = torch.empty_like(preds)
        ground_truth[..., 3:5] *= torch.tensor(anchor_boxes, device=device)
        # preds+=anchor_matrix
        ground_truth[..., 3:5] = ground_truth[..., 3:5] * 32  # back to pixel values

        bbox = ground_truth[ground_truth[..., 0] == 1][..., 1:5]
        _, labels = torch.max(
            ground_truth[ground_truth[..., 0] == 1][..., 5:].view(-1, C), dim=-1
        )

        return bbox, labels

    def match_anchor_box(
        self, bbox_w, bbox_h, to_exclude=[], anchor_boxes=ANCHOR_BOXES
    ):
        """
        Matches the bounding box to the closest anchor box.

        Parameters:
        - box_width (float): The width of the bounding box.
        - box_height (float): The height of the bounding box.
        - to_exclude (list): List of anchor boxes to exclude.

        Returns:
        - int: Index of the matched anchor box.
        """
        iou = []
        for i, box in enumerate(anchor_boxes):
            if i in to_exclude:
                iou.append(0)
                continue
            intersection_width = min(box[0], bbox_w)  # Scale up as h, w in range 0-13
            intersection_height = min(box[1], bbox_h)
            I = intersection_width * intersection_height
            IOU = I / (bbox_w * bbox_h + box[0] * box[1] - I)
            iou.append(IOU)

        iou = torch.tensor(iou)
        return torch.argmax(iou, dim=0).item()
