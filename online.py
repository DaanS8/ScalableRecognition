import db_image
import os
import cv2 as cv

def main():
    image_path = ""
    while not os.path.isfile(image_path):
        image_path = input("Give path to input image (jpg):")
        if image_path[-4:] != ".jpg":
            image_path += ".jpg"

    img = cv.imread(image_path)
    if img is None:
        raise Exception("Error reading given image, img is None.")

    # GO!


if __name__ == "__main__":
    main()
