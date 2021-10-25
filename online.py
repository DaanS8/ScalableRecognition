import db_image
import kmeans_tree
import resize
import os
import utils
import numpy as np
import time
from parameters import *


def final_scoring(kp, des, initial_scores):
    results = dict()
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    for index in initial_scores:
        # Get kp and des of image from disk
        kp_d, des_d = db_image.DbImage(index).get_kp_des()

        # Matcher
        knn_matches = matcher.knnMatch(des, des_d, 2)  # result == 2 closest matches

        # -- Filter matches using the Lowe's ratio test
        ratio_thresh = 0.7
        good_matches = []
        for m, n in knn_matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

        # If at least 10 keypoints can be tested
        if len(good_matches) > 10:
            # Get x,y coordinates
            src_pts = np.float32([kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_d[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Search homography
            _, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

            # Count nb of inliers
            final = np.count_nonzero(mask)

            # If at least 6 inliers, good result!
            if final > 5:
                results[index] = final

    return results


def main():
    tree = kmeans_tree.KMeansTree("sift_more_leaves.p")

    while True:
        # Get an image path to process
        image_path = ""
        while (not os.path.isfile(image_path)) and image_path != "q":
            image_path = input("Give path to input image (jpg), enter q to quit:")
            if image_path[-4:] != ".jpg" and image_path != "q":
                image_path += ".jpg"

        # Check if exit is needed
        print("")
        if image_path == "q":
            print("Exiting.")
            break

        start = time.time()

        # Load grayscaled image
        img = resize.get_resize_gray(image_path)
        if img is None:
            raise ValueError("Reading image resulted in None.")

        # Calculate kp and des
        kp, des = utils.sift.detectAndCompute(img, None)

        # Initial scoring
        indices, scores = tree.initial_scoring(des)

        # Only use the best NB_OF_IMAGES_CONSIDERED in final scoring
        considered = np.argpartition(scores, -NB_OF_IMAGES_CONSIDERED)[-NB_OF_IMAGES_CONSIDERED:]
        final_result = final_scoring(kp, des, indices[considered])

        if len(final_result) > 0:  # any results found?
            # Start calculating certainty scores
            minimal_value = max(final_result.values()) * 0.2
            good = {k: v for k, v in final_result.items() if v >= minimal_value}
            sum_values = sum(good.values())

            first = True
            for key, value in sorted(good.items(), key=lambda item: item[1], reverse=True):
                certainty_percentage = min(100, value - 5) * value / sum_values

                # Check if the best result is certain/uncertain
                if first:
                    first = False
                    if sum_values < 105 or certainty_percentage < 50:
                        print("Uncertain results, please consider taking a new picture.")

                # lookup name of the result
                name = db_image.DbImage(key).get_name()
                if name == "":
                    name = str(key)

                # result
                print("{}: {:.2f}%".format(name, certainty_percentage))
        else:
            print("No result found, please take a new picture.")

        print("Processing this result took {:.2f}s.".format(time.time() - start))
        print("")


if __name__ == "__main__":
    main()

