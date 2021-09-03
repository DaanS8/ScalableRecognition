from parameters import *


# resize the given image to the given dimensions using the method inter.
def image_resize(image, width=None, height=None, inter=cv.INTER_AREA):
    # copied from https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        # calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        # calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    return cv.resize(image, dim, interpolation=inter)


def resize_gray_store(path):
    if path[-4:] != ".jpg":  # raise error if not a jpg
        raise ValueError("All files should be have the extension .jpg. " + path + " isn't a jpg.")
    try:
        img = cv.imread(path, cv.IMREAD_COLOR)
        if img is None:
            print("WARNING: reading " + path + " resulted in None.")
        # process only if colored image or image to large
        elif len(img.shape) == 3 or img.shape[0] > MAX_IMAGE_SIZE or img.shape[1] > MAX_IMAGE_SIZE:
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # convert to grayscale
            height, width = img_gray.shape

            # check which dimension is the largest, and if it exceeds MAX_IMAGE_SIZE, if so resize image
            if width >= height and width > MAX_IMAGE_SIZE:
                img_gray = image_resize(img_gray, width=MAX_IMAGE_SIZE)
            elif height > MAX_IMAGE_SIZE:
                img_gray = image_resize(img_gray, height=MAX_IMAGE_SIZE)

            # if the img is resized and grayscaled, write output
            cv.imwrite(path, img_gray)
    except Exception as e:
        print("Exception at", path, str(e))


def get_resize_gray(path):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    height, width = img.shape
    # check which dimension is the largest, and if it exceeds MAX_IMAGE_SIZE, if so resize image
    if width >= height and width > MAX_IMAGE_SIZE:
        img = image_resize(img, width=MAX_IMAGE_SIZE)
    elif height > MAX_IMAGE_SIZE:
        img = image_resize(img, height=MAX_IMAGE_SIZE)
    return img
