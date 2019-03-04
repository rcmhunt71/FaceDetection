#!/usr/bin/env python

import argparse
import os

import cv2 as cv


ANALYZE = True

FRONTAL = 'frontal'
PROFILE = 'profile'

CLASSIFIER_PATH = '/miniconda3/envs/face-detection/share/OpenCV/haarcascades'
FRONTAL_FACE_XML = 'haarcascade_frontalface_alt.xml'
PROFILE_FACE_XML = 'haarcascade_profileface.xml'

FRONTAL_FACE = os.path.sep.join([CLASSIFIER_PATH, FRONTAL_FACE_XML])
PROFILE_FACE = os.path.sep.join([CLASSIFIER_PATH, PROFILE_FACE_XML])


def get_cli_args():
    arguments = argparse.ArgumentParser()
    arguments.add_argument("image", )
    arguments.add_argument("--profile", '-p', help="Detect Profile Faces", default=False, action='store_true')
    arguments.add_argument("--frontal", '-f', help="Detect Frontal Faces", default=True, action='store_true')

    return arguments.parse_args()


def highlight_cascade_matches(matches):
    for (column, row, width, height) in matches:
        cv.rectangle(original_image,
                     (column, row),
                     (column + width, row + height),
                     (0, 255, 0), 2)


def analyze_image(analysis, image, classifier):
    check_msg = "Checking image for {classification_type}..."

    print(check_msg.format(classification_type=analysis))
    frontal_face_cascade = cv.CascadeClassifier(classifier)
    matches = frontal_face_cascade.detectMultiScale(image)
    highlight_cascade_matches(matches)


def display_image_with_matches(image):
    cv.imshow('Image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    args = get_cli_args()
    if not os.path.exists(args.image):
        print("ERROR:\n\tUnable to find '{img}'".format(img=args.image))
        exit()

    # Read in the actual image
    original_image = cv.imread(args.image)

    # Translate to gray-scale for integral image map and detection
    grayscale_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)

    if args.frontal:
        analyze_image(analysis=FRONTAL, image=grayscale_image, classifier=FRONTAL_FACE)

    if args.profile:
        analyze_image(analysis=PROFILE, image=grayscale_image, classifier=PROFILE_FACE)

    # Show image with matches highlighted in green rectangles
    display_image_with_matches(original_image)
