"""
Run this file to setup the tree system used for initial scoring.
Look at parameters.py before running!

Note: the ETA/reauired diskspace ar not that accurate. Use it as an order of magnitude for the actual values.

frequently used abbreviations:
kp - keypoint
des - descriptor
db - database

Files structure:
(input)     data/   : all db images. assumption: all images have extension '.jpg'.
(temp)      tmp/    : temporary folder for storing checkpoints
(output)    calc/   : contains all stored kp and des of db images
(output)    tree.p  : the resulting tree
"""
import os
import time
import db_image
import parallelize
import random
from parameters import *
import numpy as np
import datetime
import kmeans_tree
import utils
import numpy_indexed as npi


def create_missing_output_folders():
    """
    Make sure all folders used by program exist.
    """
    if not os.path.isdir("data/"):  # raise exception if no db images are provided
        raise Exception("Folder data/ is empty. No db images provided.")
    if not os.path.isdir("calc/"):
        os.mkdir("calc/")
    if not os.path.isdir("tmp/"):
        os.mkdir("tmp/")


def eta(all_ids):
    """
    Try 100 random items to estimate duration of the whole program.
    Returns estimations of:
    - Total runtime
    - Runtime of resize and grayscale operations
    - Runtime of calculating all kp & des of db images
    - Runtime of tree
    - Size of all kp and des
    - Extra disk space used during runtime
    - The fraction of the data that should be used in the first level of the tree
    Note: tree runtime is a guesstimate
    """
    eta_resize, eta_calc, eta_tree, tmp_size_gb = 0, 0, 0, 0

    # define which ids to test
    multiply_by = len(all_ids)/100
    ids_to_test = all_ids[np.random.choice(all_ids.shape[0], 100, replace=False)]

    # eta for resizing
    if RESIZE_IMAGES:
        start = time.time()
        paths_to_test = ["data/" + str(i) + ".jpg" for i in ids_to_test]
        parallelize.parallelize_resize(paths_to_test)
        eta_resize = (time.time() - start) * multiply_by

    # eta for calculating des
    start = time.time()
    test_data = parallelize.parallelize_calc(ids_to_test)
    eta_calc = (time.time() - start) * multiply_by

    # est. size of des
    calc_size_gb = 0
    for i, d in test_data:
        calc_size_gb += i.nbytes + d.nbytes
    calc_size_gb *= multiply_by / 10**9

    # weight
    weight = min(1, MAX_MEMORY_USAGE_GB / (2.1 * calc_size_gb))

    # eta for tree, hard to guesstimate
    eta_tree = 7 * calc_size_gb * np.log(K * L + 1) * np.log(ATTEMPTS_KMEANS * CRITERIA[1] + 1)
    if weight == 1:
        eta_tree /= 2

    # total eta
    eta_tot = eta_resize + eta_calc + eta_tree
    if PRE_CALC_DES and SAVE_DISK_SPACE_DURING_RUNNING:
        eta_tot += eta_calc

    # raise error if to little memory provided based on size of des
    if 1/weight > K:
        raise MemoryError("To little memory allocated. Max memory defined by user {} GB. "
                          "Min memory needed {:.2f} GB.".format(MAX_MEMORY_USAGE_GB, 2.1 * calc_size_gb / K))

    # est. extra disk space used during runtime
    if weight < 1 and not SAVE_DISK_SPACE_DURING_RUNNING:
        tmp_size_gb = calc_size_gb

    return eta_tot, eta_resize, eta_calc, eta_tree, calc_size_gb, tmp_size_gb, weight


def format_ids_des(data):
    """
    Helper to format chunked data.
    """
    ids, des = list(), list()

    # check if useful data is given
    if len(data) == 0:
        return ids, des

    # process data
    for i, d in data:
        if d is not None and 0 != np.size(d, axis=0):
            ids.extend([i] * np.size(d, axis=0))
            des.append(d)
    return np.array(ids, np.uint32), np.concatenate(des, axis=0, dtype=np.float32)


def main():
    create_missing_output_folders()

    all_ids = db_image.get_ids()

    # ETA
    eta_tot, eta_resize, eta_calc, eta_tree, calc_size_gb, tmp_size_gb, weight = eta(all_ids)
    print("(PERMANENT) Est. size of calc/ folder: {:.2f}GB".format(calc_size_gb))
    print("(TEMPORARY) Est. extra disk space used during offline fase: {:.2f}".format(tmp_size_gb))
    print("Total ETA: " + str(datetime.timedelta(seconds=eta_tot)))

    # Resizing and grayscale images
    print("1) Start resizing and grayscaling images, ETA: " + str(datetime.timedelta(seconds=eta_resize)))
    start = time.time()
    if RESIZE_IMAGES:
        parallelize.parallelize_resize(["data/" + str(i) + ".jpg" for i in all_ids])
    print("Resizing and grayscaling finished! Runtime: " + str(datetime.timedelta(seconds=time.time() - start)))

    # Calculate and store all kp & des of images
    print("2) Start calculating des, ETA: " + str(datetime.timedelta(seconds=eta_calc)))
    start = time.time()
    ids, des = format_ids_des(parallelize.parallelize_calc(all_ids))
    print("Calculating des finished! Runtime: " + str(datetime.timedelta(seconds=time.time() - start)))

    # Build tree
    print("3) Start building tree, ETA: " + str(datetime.timedelta(seconds=eta_tree)))
    start = time.time()
    tree = kmeans_tree.KMeansTree(K, L, CRITERIA, ATTEMPTS_KMEANS)
    if weight == 1:
        for attempt in range(ATTEMPTS_TREE_BRANCH):
            try:
                tree.build_branch(ids, des, attempts_level=ATTEMPTS_TREE_LEVEL)
            except Exception as e:
                print("Failed attempt", e)
            else:
                break
        else:
            raise Exception('Failed building tree.')
    else:
        # build first node
        clusters = tree.build_node_from_given_data(des)

        # cluster descriptors in their respective nodes
        for image_id in all_ids:
            des = db_image.DbImage(image_id).get_kp_des(delete=SAVE_DISK_SPACE_DURING_RUNNING)[1]

            if des is not None and len(des) != 0:  # only use valid images
                # get closest cluster center for every des
                indices = utils.get_closest_indexes(clusters, des)
                # sort des in groups with the same cluster center
                indices, des = npi.group_by(indices, des)
                # append grouped des into their respective files
                for i, d in zip(indices, des):
                    utils.pickle_data((image_id, d), "tmp/" + str(i), mode="ab")

        # build all branches on level 1
        for node_index in range(K):
            # load data of branch
            ids, des = format_ids_des(utils.get_pickle_data_chunks("tmp/" + str(node_index)))
            # try a few times to build the branch
            for attempt in range(ATTEMPTS_TREE_BRANCH):
                try:
                    tree.build_branch(ids, des, level=1, index=node_index, attempts_level=ATTEMPTS_TREE_LEVEL)
                except Exception as e:
                    print("Failed attempt", e)
                else:
                    break
            else:
                raise Exception('Failed building branch ' + str(node_index) + ".")

            try:  # remove file if exists
                os.remove("tmp/" + str(node_index))
            except OSError:
                pass

    tree.finalise()
    print("Building tree finished! Runtime: " + str(datetime.timedelta(seconds=time.time() - start)))


if __name__ == "__main__":
    main()
