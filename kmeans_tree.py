import numpy as np
import utils
import numpy_indexed as npi
import cv2 as cv
from multiprocessing import Pool
from functools import partial
from collections import defaultdict


# helpers
def _build_tree_level_enough_cache(ids, des, k, criteria, attempts):
    _, indices, clusters = cv.kmeans(des, k, None, criteria, attempts, cv.KMEANS_PP_CENTERS)
    return npi.group_by(indices, ids)[1], npi.group_by(indices, des)[1], clusters


def _check_level_ok(res, k, l, level):
    if res is None:
        return False
    for i, _, _ in res:
        for j in i:
            if len(j) < k**(l-1-level):
                return False
    return True


def _cluster_nbs(level, index, k, l):  # could be cleaner, but works fine like this
    indeces = list()
    base = 0
    times = 1
    for i in range(l):
        if i >= level:
            indeces.extend(list(np.arange(times) + base + k**(i-level) * index))
            times = k**(i-level+1)
        base += k**i
    return indeces


def _leaf_nbs(level, index, k, l):
    return np.arange(k**(l-level)) + index * k**(l-level)


class KMeansTree:
    def __init__(self, k, l, criteria, attempts, vector_dimensions=128, file="tree.p"):
        self.k = k
        self.l = l
        self.file = file
        self.tree = utils.get_pickled(file)
        self.criteria = criteria
        self.attempts = attempts
        if self.tree is None:
            self.tree = (np.zeros([np.sum(np.power(k, np.arange(l))), k, vector_dimensions], dtype=np.float32),
                         np.zeros(k ** l, dtype="float32, object, object"))

    def store(self):
        utils.pickle_data(self.tree, self.file)

    def build_node_from_given_data(self, des, level=0, index=0):
        # take given node
        node = self.tree[0][index + np.sum(np.power(self.k, np.arange(level)))]

        # check if node not already build
        if node[0, 0] == 0:
            print("Node at level {} and index {} already build.".format(level, index))
            return None

        # calculate clusters
        clusters = cv.kmeans(des, self.k, None, self.criteria, self.attempts, cv.KMEANS_PP_CENTERS)[2]

        # update clusters
        node = clusters

        # store result!
        self.store()
        return clusters

    def build_branch(self, ids, des, level=0, index=0, attempts_level=10):
        # check if given node not already build
        if self.tree[0][index + np.sum(np.power(self.k, np.arange(level))), 0, 0] == 0:
            print("Node at level {} and index {} already build.".format(level, index))
            return None

        initial_level = level
        print("Building in branch: level {}, index {}, nb of des to process {:_}.".format(level, index, len(ids)))

        # build first level
        ids, des, clusters = _build_tree_level_enough_cache(ids, des, self.k, self.criteria, self.attempts)

        all_clusters = [clusters]
        level += 1

        # build all other levels
        while level < self.l:  # run till the depth of the tree
            print("Building in branch: level {}.".format(level))
            res = None
            tries = attempts_level
            while not _check_level_ok(res, self.k, self.l, level):  # try a few times to successfully build a tree level
                if tries != attempts_level:
                    print("trying level again.")
                with Pool() as p:
                    res = p.starmap(partial(_build_tree_level_enough_cache,
                                               k=self.k, criteria=self.criteria, attempts=self.attempts), zip(ids, des))
                if tries == 0:  # couldn't build tree level
                    raise RuntimeError("Unable to build branch at level {}. "
                                       "Probable cause is too few descriptors for the amount of leaves.".format(level))
                tries -= 1

            # if building level successful, setup data for next level
            ids, des = list(), list()
            for i, d, clusters in res:
                ids.extend(i)
                des.extend(d)
                all_clusters.append(clusters)
            del res
            level += 1

        # update clusters
        if initial_level > 0:
            for index, cluster in zip(_cluster_nbs(initial_level, index, self.k, self.l), clusters):
                self.tree[0][index] = cluster
        else:
            self.tree = (clusters, self.tree[1])

        # update leaves
        for leaf_nb, leaf_ids in zip(_leaf_nbs(initial_level, index, self.k, self.l), ids):
            unique, counts = np.unique(leaf_ids, return_counts=True)
            self.tree[1][leaf_nb] = np.array(0, dtype=np.float32), unique, counts.astype(np.uint32)

        # store result!
        self.store()

    def finalise(self, N=None):
        # check if not already done!
        if self.tree[1][-1][2].dtype == np.float32:
            print("Tree already finalised.")
            return None

        # Count
        if N is None:
            N = 0
            for leaf_nb in np.arange(self.k ** self.l):
                N += np.sum(self.tree[1][leaf_nb][2])

        all_items = defaultdict(lambda: np.array(0, dtype=np.float32))

        # Entropy
        for leaf_nb in np.arange(self.k ** self.l):
            leaf = self.tree[1][leaf_nb]

            w = np.log(N / np.sum(leaf[2], dtype=np.uint32))
            leaf[2] = np.multiply(w, leaf[2], dtype=np.float32)
            leaf[0] = w

            for item, score in zip(leaf[1], leaf[2]):
                all_items[item] += score

        # Normalisation
        print("Normalising scores.")
        for leaf_nb in np.arange(self.k ** self.l):
            leaf = self.tree[1][leaf_nb]
            sizes = np.array([all_items[item] for item in leaf[1]], dtype=np.float32)
            leaf[2] = np.divide(leaf[2], sizes, dtype=np.float32)

        # store result!
        self.store()
