from parameters import *
import pickle
import numpy as np


sift = cv.SIFT_create()
bf = cv.BFMatcher(cv.NORM_L2)


def get_pickled(path):
    try:
        with open(path, 'rb') as handle:
            data = pickle.load(handle)
        return data
    except Exception:
        return None


def get_pickle_data_chunks(path):
    data = list()
    try:
        with open(path, 'rb') as f:
            try:
                while True:
                    data.append(pickle.load(f))
            except EOFError:
                pass
    except FileNotFoundError:
        pass
    return data


def pickle_data(data, path, mode='wb'):
    with open(path, mode) as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


# kp helpers
def convert_kp_to_kpl(kp):
    return np.array([(p.pt, p.size, p.angle, p.response, p.octave, p.class_id) for p in kp], dtype="object")


def convert_kpl_to_kp(kp_l):
    return [cv.KeyPoint(x=p[0][0], y=p[0][1], _size=p[1], _angle=p[2], _response=p[3], _octave=p[4], _class_id=p[5]) for p in kp_l]


def get_closest_indexes(base, compare):
    return np.array([i.trainIdx for i in bf.match(compare, base)], dtype=np.uint32)
