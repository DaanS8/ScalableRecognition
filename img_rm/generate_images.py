# Source: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_sift_intro/py_sift_intro.html
import cv2 as cv
import numpy as np
from past.builtins import xrange

MIN_MATCH_COUNT = 10

img_1 = cv.imread('200890_db.jpg', cv.IMREAD_GRAYSCALE)
img_2 = cv.imread('200890.jpg', cv.IMREAD_GRAYSCALE)

sift = cv.SIFT_create()
kp_1, des_1 = sift.detectAndCompute(img_1, None)
kp_2, des_2 = sift.detectAndCompute(img_2, None)

img_1_kp = cv.drawKeypoints(img_1, kp_1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_2_kp = cv.drawKeypoints(img_2, kp_2, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imwrite('./keypoint_detection.jpg', np.concatenate((img_1_kp, img_2_kp), axis=1))

matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
matches = matcher.knnMatch(des_1, des_2, k=2)

# Apply ratio test
good = []
matchesMask = [[0, 0] for i in xrange(len(matches))]
for i, (m, n) in enumerate(matches):
    if m.distance < 0.5 * n.distance:
        good.append([m])
        matchesMask[i] = [1, 0]

draw_params = dict(matchColor=(52, 235, 211),
                   singlePointColor=(0, 0, 255),
                   matchesMask=matchesMask,
                   flags=0)
# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img_1, kp_1, img_2, kp_2, matches, None, **draw_params)
cv.imwrite('./matching.jpg', img3)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp_1[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_2[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    h, w = img_1.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, M)
    img_2 = cv.polylines(cv.cvtColor(img_2, cv.COLOR_GRAY2RGB), [np.int32(dst)], True, (0, 0, 255), 3, cv.LINE_AA)
else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                   singlePointColor=None,
                   matchesMask=matchesMask,  # draw only inliers
                   flags=2)

good = [i[0] for i in good]
img4 = cv.drawMatches(img_1, kp_1, img_2, kp_2, good, None, **draw_params)
cv.imwrite('./RANSAC.jpg', img4)
