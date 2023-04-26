# import required libraries
import glob
import os
import cv2
import pandas as pd
import torch
import torchvision
from PIL.Image import Image


def crop_image(path):
    import cv2

    # read the input image
    img = cv2.imread(path)

    # convert to grayscale of each frames
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # read the haarcascade to detect the faces in an image
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

    # detects faces in the input image
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    print('Number of detected faces:', len(faces))

    cropped = []
    # loop over all detected faces
    if len(faces) > 0:
        for i, (x, y, w, h) in enumerate(faces):
            # To draw a rectangle in a face
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            face = img[y:y + h, x:x + w]
            # cv2.imshow("Cropped Face", face)
            # cv2.imwrite(f'face{i}.jpg', face)
            # print(f"face{i}.jpg is saved")
            cropped.append(face)

    # display the image with detected faces
    # cv2.imshow("image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    if len(cropped) != 0:
        return cropped[0]
    return img


def main():
    root_dir = '/data/faces/ARCHIVE/img_celeba_under_1000'
    root_dir_cropped = 'data/faces/ARCHIVE/img_celeba_under_1000_cropped'
    image_paths = sorted(glob.glob(os.path.join(root_dir, "*.*")))

    for path in image_paths:
        base = os.path.basename(path)
        ci = crop_image(path)
        new_dir = root_dir_cropped + '/' + 'cropped_' + base
        cv2.imwrite(new_dir, ci)

if __name__ == '__main__':
    main()
