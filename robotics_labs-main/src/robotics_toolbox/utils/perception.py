#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2025-09-21
#     Author: Martin Cífka <martin.cifka@cvut.cz>
#
from typing import List
from numpy.typing import ArrayLike
import numpy as np
import cv2  # noqa


def find_hoop_homography(images: ArrayLike, hoop_positions: List[dict]) -> np.ndarray:
    """
    Find homography based on images containing the hoop and the hoop positions loaded from
    the hoop_positions.json file in the following format:

    [{
        "RPY": [-0.0005572332585040621, -3.141058227474627, 0.0005185830258253442],
        "translation_vector": [0.5093259019899434, -0.17564068853313258, 0.04918733225140541]
    },
    {
        "RPY": [-0.0005572332585040621, -3.141058227474627, 0.0005185830258253442],
        "translation_vector": [0.5093569397977782, -0.08814069881074972, 0.04918733225140541]
    },
    ...
    ]
    """

    images = np.asarray(images)
    assert images.shape[0] == len(hoop_positions)

    # todo HW03: Detect circle in each image
    # todo HW03: Find homography using cv2.findHomography. Use the hoop positions and circle centers.

    #rotacni a translacni vektor v hoop_positions a podle obrazku
    #1 najit kruznice v obrazku
    #najit homografii z roviny obrazku do roviny (v hoop_positions je treti slozka vzdy stejan - nezajima me)
    #homografie 3x3 matice mezi dvema rovinama transformuje souradnice

    #H = cv2.findHomogramphy(center_circle, world_points) [0]
    #y = H @ circle_centers

    #03_homography.py - nacte obrazky, nacte json, zavola fci - to nam da matici
    #vezme si souradnice a oseka z nich x, y
    #prida 1 aby byly homogenni
    #vezmu si H_inv
    #

    #nezapomen si pretahnout rozzipovany data

    return np.eye(3)
