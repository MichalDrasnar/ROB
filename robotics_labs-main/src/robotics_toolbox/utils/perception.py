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
    H = np.eye(3)
    src_points = [] #circle centers in image
    dst_points = [] #hoop positions in world coordinates

    #RGB to GRAY
    for i in range(images.shape[0]):
        img = images[i]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=100, param2=30, minRadius=10, maxRadius=100)
        circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=100, param2=30, minRadius=97, maxRadius=102)
        #cv2.imshow("Image", img_gray)
        #cv2.waitKey(0)  
        #cv2.destroyAllWindows()
        x = circles[0][0][0] if circles is not None else None
        y = circles[0][0][1] if circles is not None else None
        """
        if circles is not None:
            print(f"Image {i}: Detected circle at (x={circles[0][0][0]}, y={circles[0][0][1]}) with radius {circles[0][0][2]}")
        else:
            print(f"Image {i}: No circles detected")"""
        
        if x is not None and y is not None:
            #print("AAAAAAAAAAAAAAAAA")
            src_points.append([x, y, 1.0])
            hoop_pos = hoop_positions[i]['translation_vector']
            dst_points.append([hoop_pos[0], hoop_pos[1], 1.0])

    #Convert to numpy arrays
    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)
    H, mask = cv2.findHomography(src_points, dst_points, method=0)
    #print("Homography matrix H:")
    #print(H)

    #testovani
    for i in range(src_points.shape[0]):
        src_pt = src_points[i]
        dst_pt = dst_points[i]
        projected_pt = H @ src_pt
        projected_pt /= projected_pt[2]  # Normalize to make the last coordinate 1
        print(f"Source point: {src_pt}, Projected point: {projected_pt}, Actual point: {dst_pt}")
        print("\n")

    return H



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