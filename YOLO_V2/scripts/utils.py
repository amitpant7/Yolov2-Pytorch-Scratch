import torch


import torch.nn as nn
from torchvision import ops
from torcheval.metrics import AUC

from config import *


def make_conv_layers(arch_config, in_channels=3):
    layers = []
    in_channels = in_channels
    for value in arch_config:
        if type(value) == tuple:
            out_channels, kernel_size = value
            layers += [
                nn.Conv2d(
                    in_channels, out_channels, kernel_size, padding="same", bias=False
                ),  # not using bias as batchnorm
                nn.BatchNorm2d(value[0]),
                nn.LeakyReLU(negative_slope=0.1),
            ]

            in_channels = out_channels

        elif value == "M":
            layers += [nn.MaxPool2d(kernel_size=2)]

    return nn.Sequential(*layers)


def convert_to_corners(boxes):
    """
    Convert bounding boxes from (x_center, y_center, width, height) format to
    (x_min, y_min, x_max, y_max) format.

    Args:
        boxes (Tensor): Tensor of shape (N, 4) containing bounding boxes in the format (x_center, y_center, width, height).

    Returns:
        Tensor: Tensor of shape (N, 4) containing bounding boxes in the format (x_min, y_min, x_max, y_max).
    """
    x_center, y_center, width, height = boxes.unbind(1)
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2
    return torch.stack((x_min, y_min, x_max, y_max), dim=1)


def intersection_over_union(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : list or tuple
        Format: [center_x, center_y, width, height]
        The (center_x, center_y) position is the center of the bounding box,
        and width and height define its dimensions.
    bb2 : list or tuple
        Format: [center_x, center_y, width, height]
        The (center_x, center_y) position is the center of the bounding box,
        and width and height define its dimensions.

    Returns
    -------
    float
        in [0, 1]
    """
    bb1 = bb1.to("cpu")
    bb2 = bb2.to("cpu")
    bboxes = torch.vstack((bb1, bb2))
    # Convert center-width-height format to top-left and bottom-right format
    bboxes = convert_to_corners(bboxes)
    bb1_x1, bb1_y1, bb1_x2, bb1_y2 = bboxes[0]
    bb2_x1, bb2_y1, bb2_x2, bb2_y2 = bboxes[1]

    # Ensure validity of bounding boxes
    if bb1_x1 > bb1_x2 or bb1_y1 > bb1_y2 or bb2_x1 > bb2_x2 or bb2_y1 > bb2_y2:
        return 0

    # Determine the coordinates of the intersection rectangle
    x_left = max(bb1_x1, bb2_x1)
    y_top = max(bb1_y1, bb2_y1)
    x_right = min(bb1_x2, bb2_x2)
    y_bottom = min(bb1_y2, bb2_y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute the area of both AABBs
    bb1_area = (bb1_x2 - bb1_x1) * (bb1_y2 - bb1_y1)
    bb2_area = (bb2_x2 - bb2_x1) * (bb2_y2 - bb2_y1)

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    return iou


def process_preds(preds, anchor_boxes=ANCHOR_BOXES, device=DEVICE):
    """Takes predictions in float and returns the pixel values as [obj_score, center_cords(x,y), w, h, class_prob]

    Args:
        preds (_type_): shape[B, S, S, N, C+5]
        anchor_boxes (anchor_boxes, optional): _description_. Defaults to ANCHOR_BOXES.
        thre (float, optional): Thershold value to consider prediction as object. Defaults to 0.5.

    Returns:
        tensor: New preds with [conf_score, bbcordi, classes]
    """

    # Calculating the center coordinates of predicted bounding box.
    sig = torch.nn.Sigmoid()
    preds[..., 0:1] = sig(preds[..., 0:1])  # objectness score.
    preds[..., 1:3] = sig(
        preds[..., 1:3]
    )  # sig(tx) in paper, back to pixesl from float

    # for getting the center point of pred bb, bx = sig(tx)+cx in paper

    cx = cy = torch.tensor([i for i in range(S)], device=device)
    preds = preds.permute(
        (0, 3, 4, 2, 1)
    )  # permute to obtain the shape (B,5,9, 13,13) so that 13,13 can be updated

    preds[..., 1:2, :, :] += cx
    preds = preds.permute(0, 1, 2, 4, 3)
    preds[..., 2:3, :, :] += cy
    preds = preds.permute((0, 3, 4, 1, 2))  # bakck to B,13,13,5,9

    preds[..., 1:3] *= 32  # to pixels

    # Calculating the height and width in pixels

    # Calculating the height and widht
    preds[..., 3:5] = torch.exp(preds[..., 3:5])  # pw*e^tw in paper
    # anchor_matrix = torch.empty_like(preds)
    preds[..., 3:5] *= torch.tensor(anchor_boxes, device=device)
    # preds+=anchor_matrix
    preds[..., 3:5] = preds[..., 3:5] * 32  # back to pixel values

    return preds


def non_max_suppression(boxes, scores, iou_threshold=0.4):
    """
    Perform non-maximum suppression to eliminate redundant bounding boxes based on their scores.

    Args:
        boxes (Tensor): Tensor of shape (N, 4) containing bounding boxes in the format (x_center, y_center, width, height).
        scores (Tensor): Tensor of shape (N,) containing confidence scores for each bounding box.
        threshold (float): Threshold value for suppressing overlapping boxes.

    Returns:
        Tensor: Indices of the selected bounding boxes after NMS.
    """
    # Convert bounding boxes to [x_min, y_min, x_max, y_max] format
    boxes = convert_to_corners(boxes)
    #     print(boxes)

    # Apply torchvision.ops.nms
    keep = ops.nms(boxes, scores, iou_threshold)

    return keep


# depends on non_max_supression_implementation and post_processing


# iou_thres_for_corr_predn- Min iou with ground bb to consider it as correct prediction.
def mean_average_precision(
    predictions, targets, data, iou_thres_nms=0.4, iou_thres_for_corr_predn=0.4, C=C
):
    """Calculates Mean avg precision for a single batch, to calculate for all batch collect prediction
    and targets in a tensor and pass it here

    Args:
        predictions (_type_): Model outputs in the tensor format (B, S,S,N,C+5)
        targets (_type_): _description_
        data (_type_): Custom Dataset instance
        iou_thres_nms (float, optional): Threshold for IOU in non max supression. Defaults to 0.4.
        iou_thres_for_corr_predn (float, optional):  Min iou with ground bb to consider it as correct prediction. Defaults to 0.4.
    """

    ep = 1e-6

    predictions = predictions.detach().clone()
    targets = targets.detach().clone()
    # getting back pixel values:
    processed_preds = process_preds(predictions)
    pr_matrix = torch.empty(
        9, C, 2
    )  # Precision and recall values at 9 different levels of threh(confidance score)

    for thres in range(1, 10, 1):

        ground_truth = targets.clone()

        conf_thres = thres / 10

        local_pr_matrix = torch.zeros(
            C, 3
        )  # Corr_pred, total_preds, ground_truth for every class

        for i in range(processed_preds.size(0)):  # looping over all preds

            # processing the preds to make it suitable
            preds = processed_preds[i]
            obj = preds[..., 0] > conf_thres

            bboxes = torch.flatten(preds[obj][..., 1:5], end_dim=-2)
            scores = torch.flatten(preds[obj][..., 0])
            _, ind = torch.max(preds[obj][..., 5:], dim=-1)
            classes = torch.flatten(ind)

            best_boxes = non_max_suppression(bboxes, scores, iou_thres_nms)

            filtered_bbox = bboxes[best_boxes]
            filtered_classes = classes[best_boxes]

            #         print(filtered_bbox[filtered_classes==0])
            gt_bboxes, labels = data.inverse_target(
                ground_truth[i].unsqueeze(0)
            )  # inverse_target expects batched
            #         print(gt_bboxes, labels)
            # matche the one bbox among the predicted boxes with the ground thruth box that gives higesht iou.
            tracker = torch.zeros_like(labels)  # to keep track of matched boxes

            for c in range(C):
                total_preds = torch.sum(filtered_classes == c)
                corr_preds = 0
                actual_count = torch.sum(labels == c)
                for box in filtered_bbox[filtered_classes == c]:
                    best_iou = 0
                    for index, value in enumerate(labels):
                        if c == value:

                            iou = intersection_over_union(
                                box, gt_bboxes[index]
                            )  # format is cx,cy, w,h

                            if iou > best_iou and tracker[index] == 0:
                                best_iou = iou
                                temp = index
                    #
                    if best_iou > iou_thres_for_corr_predn:
                        tracker[temp] = 1
                        corr_preds += 1

                local_pr_matrix[c] += torch.tensor(
                    [corr_preds, total_preds, actual_count]
                )

            precision, recall = local_pr_matrix[:, 0] / (
                local_pr_matrix[:, 1] + ep
            ), local_pr_matrix[:, 0] / (
                local_pr_matrix[:, 2] + ep
            )  # pr at a certain threshold c
            #             print(precision, recall) # should be of shape C

            pr_matrix[thres - 1] = torch.cat(
                (precision.view(-1, 1), recall.view(-1, 1)), dim=1
            )

    # precision_list = torch.nan_to_num(torch.tensor(precision_list), nan = 0)
    pr_matrix = pr_matrix.permute(1, 0, 2)  # now shape class, all pr values

    # lets calculate the mean precision
    metric = AUC(n_tasks=C)
    metric.update(pr_matrix[..., 0], pr_matrix[..., 1])
    average_precision = metric.compute()

    return average_precision.mean()


def check_model_accuracy(preds, targets, thres=0.5):
    total_class, class_corr = 0, 0
    total_obj, obj_corr = 0, 0
    total_no_obj, no_obj_corr = 0, 0
    sig = torch.nn.Sigmoid()

    # No object score will be the recall value

    # Class Score will be the recall

    obj = targets[..., 0] == 1  # mask
    no_obj = targets[..., 0] == 0

    preds[..., 0] = sig(preds[..., 0])

    class_corr = torch.sum(
        (
            torch.argmax(preds[obj][..., 5:], dim=-1)
            == torch.argmax(targets[obj][..., 5:], dim=-1)
        )
    )

    total_class = torch.sum(obj)

    obj_corr = torch.sum(preds[obj][..., 0] > thres)
    total_obj = torch.sum(obj) + 1e-6  # to avoid divide by zero

    no_obj_corr = torch.sum(preds[no_obj][..., 0] < thres)

    total_no_obj = torch.sum(no_obj)

    return torch.tensor(
        [total_class, class_corr, total_obj, obj_corr, total_no_obj, no_obj_corr]
    )

    #     print('Class Score', (100*class_corr/total_class_pred).item())


#     print('Object Score', (100*obj_corr/total_obj_prd).item())
#     print('No object Score', (100*no_obj_corr/total_no_obj).item())


def cal_epoch_acc(
    total_class_pred, class_corr, total_obj_prd, obj_corr, total_no_obj, no_obj_corr
):
    print("Class Score (R)", 100 * class_corr / total_class_pred)
    print("Object Score (R)", 100 * obj_corr / total_obj_prd)
    print("No object Score (R)", 100 * no_obj_corr / total_no_obj)
