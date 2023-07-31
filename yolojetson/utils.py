# utils.py
# Some common functions

import cv2
import numpy as np

import yolojetson.constants


def visualise_predictions(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    """
    Annotates an image with the given bounding boxes.

    :param img (numpy.ndarray): The image to be annotated, shape HxWx3
    :param boxes: List of bounding boxes in format [[start_x_0, start_y_0, end_x_0, end_y_0], [start_x_1, start_y_1, end_x_1, end_y_1], ...]
    :param scores: List of confidence scores for each bbox.
    :param cls_ids: List of class IDs for each bbox.
    :param conf: List of confidences for each bbox.
    :returns: numpy.ndarray of shape HxWx3 with bbox annotations superimposed.
    """
    for i in range(len(boxes)):
        score = scores[i]
        if score < conf:
            continue
        box = boxes[i]
        cls_id = int(cls_ids[i])
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (yolojetson.constants._COLORS[cls_id % 80] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(yolojetson.constants._COLORS[cls_id % 80]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (yolojetson.constants._COLORS[cls_id % 80] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img
