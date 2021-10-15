"""
Change the parameters of the program here.
"""
import cv2 as cv

# all images (query & db) are resized before calculating their des. Their original aspect ratio is maintained.
MAX_IMAGE_SIZE = 1080

# Branches of tree (K > 1)
K = 10
# Depth of tree (L > 6)
L = 6
# nb of leaf nodes = K**L

# Halves needed disk space during runtime. Requires recalculating all des at the end.
SAVE_DISK_SPACE_DURING_RUNNING = False


# How much RAM/memory is available for the program.
# In pycharm you can increase this under Help>Change memory settings.
# In a window machine, you can increase the cache size by following the steps on:
# https://www.windowscentral.com/how-change-virtual-memory-size-windows-10
#
# If possible provide (at least) '2.1 x size of all des' as available memory.
# If not enough ram the first k-means will use only a portion of the data.
# It will also store checkpoints on disk and reload them when needed during the processing of every branch.
# Due to disk access this will be slower than using all data, but the end result will have almost the same accuracy.
# Minimum memory size = 2.1 * size all des / K
MAX_MEMORY_USAGE_GB = 60

# How many of the first results of the initial scoring should be tested with geometric verification?
NB_OF_IMAGES_CONSIDERED = 5


"""DEBUG PARAMETERS"""


# During offline fase, resize and grayscale all db images?
# Recommended: True
RESIZE_IMAGES = False

# Store all kp & des of all db images in calc/? Speeds up online fase but requires lots of disk space.
# Only applicable if SAVE_DISK_SPACE_DURING_RUNNING = True
# Recommended: True
PRE_CALC_DES = True

# Criteria of kmeans algorithm
# opencv docs: https://docs.opencv.org/4.5.2/d1/d5c/tutorial_py_kmeans_opencv.html
# Recomended:
# (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 5, 0.5)
# 5
CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 2, 1)
ATTEMPTS_KMEANS = 2

ATTEMPTS_TREE_LEVEL = 10
ATTEMPTS_TREE_BRANCH = 25
