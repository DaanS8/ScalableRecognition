import utils
import resize
import os
import numpy as np


names = -1


def get_names():
    global names
    if names == -1:
        names = utils.get_pickled("names.p")
    return names


def get_ids():
    names = get_names()
    if names is not None:
        return np.array(list(names.keys()), dtype=np.uint32)
    else:
        return np.array([int(file[:-4]) for file in os.listdir("data/")], dtype=np.uint32)


class DbImage:
    def __init__(self, id):
        self.id = id

    def get_id(self):
        return self.id

    def get_name(self):
        return get_names().get(self.id, "")

    def get_image(self):
        return resize.get_resize_gray("data/" + str(self.id) + ".jpg")

    def get_kp_des(self, delete=False):
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
                utils.pickle_data((utils.convert_kp_to_kpl(kp), des), data_path)
                return kp, des
        return None, None  # default
