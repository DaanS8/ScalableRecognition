import utils
import resize
import os
import numpy as np


names = -1


def get_names():  # Load the pickle file "names.p" if it exists
    global names
    if names == -1:
        names = utils.get_pickled("names.p")
    return names


def get_ids():
    # Get every id of every db entry
    names = get_names()
    if names is not None:
        return np.array(list(names.keys()), dtype=np.uint32)
    else:
        return np.array([int(file[:-4]) for file in os.listdir("data/")], dtype=np.uint32)


class DbImage:
    # Class for database images
    def __init__(self, id):
        self.id = id

    def get_id(self):
        return self.id

    def get_name(self):
        all_names = get_names()
        if all_names is not None:
            return all_names.get(self.id, "")
        else:
            return ""

    def get_image(self):
        return resize.get_resize_gray("data/" + str(self.id) + ".jpg")

    def get_kp_des(self, delete=False):
        # Get kp and des from database image
        # Try to get from disk
        # If not on disk, calculate and store on disk
        data_path = "calc/" + str(self.id)
        data = utils.get_pickled(data_path)
        if data is not None:
            if delete:
                os.remove(data_path)
            return utils.convert_kpl_to_kp(data[0]), data[1]
        else:
            img = self.get_image()
            if img is not None:
                kp, des = utils.sift.detectAndCompute(img, None)
                if not delete:
                    utils.pickle_data((utils.convert_kp_to_kpl(kp), des), data_path)
                return kp, des
        return None, None  # default
