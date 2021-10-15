import os
import time

import resize
import utils
import kmeans_tree
from parameters import *
import cv2 as cv
import numpy as np
import parallelize
import db_image
import numpy_indexed as npi
from collections import OrderedDict

trees = ["sift_all_tree.p"]
test_folder = "testset/"


def _look_for_sub_test_folders():
    # Check if there are sub test folders in the given test_folder
    # If there are sub folders, use these as test_folders
    # If there are no sub folders, use the given test_folder in test_folders
    try:
        test_folders = [test_folder + name_folder + "/" for name_folder in os.listdir(test_folder)]
        if len(test_folders) == 0:
            test_folders = [test_folder]
    except Exception:  # only a main folder
        test_folders = [test_folder]
    return test_folders


def final_scoring(kp, des, initial_scores):
    results = dict()
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    for index in initial_scores:
        kp_d, des_d = db_image.DbImage(index).get_kp_des()
        knn_matches = matcher.knnMatch(des, des_d, 2)  # result == 2 closest matches

        # -- Filter matches using the Lowe's ratio test
        ratio_thresh = 0.7
        good_matches = []
        for m, n in knn_matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

        if len(good_matches) > 10:
            src_pts = np.float32([kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_d[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            final = np.count_nonzero(mask)
            if final > 5:
                results[index] = final
    return results


def main():
    test_folders = _look_for_sub_test_folders()

    for tree_path in trees:
        tree = kmeans_tree.KMeansTree(tree_path)

        start_total = time.time()
        t0, t1, t2, t3, t4, t5 = 0, 0, 0, 0, 0, 0
        processed_images = 0
        for folder in test_folders:
            start = time.time()
            paths = [folder + file_name for file_name in os.listdir(folder)]
            parallelize.parallelize_resize(paths)
            t0 += time.time() - start
            processed_images += len(paths)

            result = dict()
            percentage_correct = dict()
            percentage_incorrect = dict()
            certain = [0, 0]
            uncertain = [0, 0]
            no_result = [0, 0]

            print("Start processing folder '{}' with tree '{}'".format(folder, tree_path))
            for path in paths:
                try:
                    start = time.time()
                    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
                    kp, des = utils.sift.detectAndCompute(img, None)
                    t1 += time.time() - start

                    start = time.time()
                    indices, scores = tree.initial_scoring(des)
                    t2 += time.time() - start

                    start = time.time()
                    if "Junk" not in path:
                        correct_id = int(path[path.rfind("/") + 1:-4])
                    else:
                        correct_id = -1

                    sort = sorted(zip(indices, scores), key=lambda item: item[1], reverse=True)

                    correct_index = -1
                    if correct_id != -1:
                        for index, item in enumerate(sort):
                            if item[0] == correct_id:
                                correct_index = index
                                break
                    result[correct_index] = result.get(correct_index, 0) + 1
                    t3 += time.time() - start

                    start = time.time()
                    considered = np.argpartition(scores, -NB_OF_IMAGES_CONSIDERED)[-NB_OF_IMAGES_CONSIDERED:]
                    final_result = final_scoring(kp, des, indices[considered])
                    t4 += time.time() - start

                    start = time.time()
                    if len(final_result) > 0:
                        minimal_value = max(final_result.values()) * 0.1
                        sum_values = sum(final_result.values())
                        good = {k: v for k, v in final_result.items() if k >= minimal_value}

                        first = True
                        for key, value in sorted(good.items(), key=lambda item: item[1], reverse=True):
                            certainty_percentage = min(100, value - 5) * value / sum_values
                            if first:
                                first = False
                                if correct_id != -1:
                                    if key == correct_id:
                                        percentage_correct[certainty_percentage] = percentage_correct.get(certainty_percentage, 0) + 1
                                    else:
                                        percentage_incorrect[certainty_percentage] = percentage_incorrect.get(certainty_percentage, 0) + 1

                                    # Certain/Uncertain as boolean value
                                    if sum_values < 105 or certainty_percentage < 50:
                                        if key == correct_id:
                                            uncertain[0] += 1
                                        else:
                                            uncertain[1] += 1
                                    else:
                                        if key == correct_id:
                                            certain[0] += 1
                                        else:
                                            certain[1] += 1
                    else:
                        if correct_id == -1:
                            no_result[1] += 1
                        else:
                            no_result[0] += 1
                    t5 += time.time() - start
                except Exception as e:
                    print("Error at {} with error: {}".format(path, e))

            print("position index", OrderedDict(sorted(result.items())))
            print("Certainty percentage correct", OrderedDict(sorted(percentage_correct.items())))
            print("Certainty percentage incorrect", OrderedDict(sorted(percentage_incorrect.items())))
            print("Certain", certain)
            print("Uncertain", uncertain)
            print("No Result", no_result)

        print("Average time per image: {:.2f}s.".format((time.time() - start_total)/processed_images))

        sum_times = (t0 + t1 + t2 + t3 + t4 + t5)/100
        t0, t1, t2, t3, t4, t5 = t0/sum_times,  t1/sum_times, t2/sum_times, t3/sum_times, t4/sum_times, t5/sum_times
        print("Percentages: Preprocessing images {:.2f}, Calculating kp & des {:.2f}, Initial Scoring {:.2f}, "
              "Accuracy Testing {:.2f}, Final Scoring {:.2f}, Certainty Calculation {:.2f}."
              .format(t0, t1, t2, t3, t4, t5))



if __name__ == "__main__":
    main()
