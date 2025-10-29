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
    img_fails = 0

    for i in range(images.shape[0]):
        print("" \
        "" \
        "Processing image ", i)
        img = images[i]

        #DST
        x, y = hoop_positions[i]["translation_vector"][:2]
        dst_points.append([x, y, 1.0])



        #SRC (+img_fails)

        #BGR to HSV
        #parameters for thresholding black
        Hue = 100
        Sat = 100
        Val = 100
        threshold = 60
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # not needed conversions thanks to just defined Hue, Sat, Val and trheshhold
        """H = Hue / 2  # 0-179 for H
        S = (Sat / 100) * 255  # percentage to 0-255
        V = (Val / 100) * 255  # percentage to 0-255"""
        #Creating lower and upper bounds
        lower_bound = np.array([Hue - threshold, Sat - threshold, Val - threshold])
        upper_nound = np.array([Hue + threshold, Sat + threshold, Val + threshold])
        # Make a mask for range betwen bounds lower_bound and upper_bound
        mask = cv2.inRange(hsv, lower_bound, upper_nound)

        """
        #visualization of mask
        cv2.imshow("Mask", mask)
        cv2.waitKey(200)  # zobrazí 2 vteřiny
        cv2.destroyAllWindows()  # zavře okno po pauze"""


        #blur - mmedain or gaussian
        blur = cv2.medianBlur(mask, 5)
        #blur = cv2.GaussianBlur(mask, (5,5), 0)

        #finding cicrles with cv2.HoughCircles
        circles = cv2.HoughCircles( # Hough transform to find circles
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1, # downsampling
            minDist=max(10, blur.shape[0]//8), # minimal center distance between circles
            param1=100, # upper edge det. threshold
            param2=15, # acumulator threshold
            minRadius=120, #px - min radius
            maxRadius=150 #px - no max limit
        )
        ## circles = example: [[[x, y, r]], ...]


        """
        # visualization of found circles
        if circles is not None and circles.shape[1] > 1:
            #now i dont want to deal with multiple circles found
            #TODO
            print("TOOOO MANY CIRCLES FOUND!!!")
            print(len(circles), "circles found in image ", i)
        elif circles is not None:
            print("Circle found in image ", i)
            print("radius: ", circles[0][0][2], "px")
            circles = np.round(circles[0, :]).astype(int)  # [ [x, y, r], ... ]
            for (x, y, r) in circles:
                cv2.circle(img, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            cv2.imshow("Detected Circles", img)
            cv2.waitKey(200)  # zobrazí 2 vteřiny
            cv2.destroyAllWindows()  # zavře okno po pauze
        else:
            print("No circle found in image ", i)
            continue #skip this image if no circle found"""
        

        if circles is not None and circles.shape[1] > 1 :
            print("Error: multiple circles found in", i)
            continue
        elif circles is None:
            img_fails += 1
            print("Error: no circles found in", i)   
            continue

        #centres of detected circles
        x, y, r = circles[0][0]
        src_points.append([x, y])

    #not enough points detected
    if (len(src_points) < 4):
        print("Error: not enough points detected:", len(src_points))
        return np.eye(3)

    #converz to np arrays
    src_pts = np.asarray(src_points, dtype=np.float32)
    dst_pts = np.asarray(dst_points, dtype=np.float32)

    H, inliers = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC)
    if H is None:
        print("Error: Homography could not be computed.")
        return np.eye(3)
    
    print("FOUND HOMOGRAPHY")



    """#testovani
    for i in range(len(src_points)):
        src_pt = src_points[i]
        dst_pt = dst_points[i]
        projected_pt = H @ src_pt
        projected_pt /= projected_pt[2]  # Normalize to make the last coordinate 1
        print(f"Source point: {src_pt}, Projected point: {projected_pt}, Actual point: {dst_pt}")
        print("\n")"""

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