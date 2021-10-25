import numpy as np
import utils
import numpy_indexed as npi
import cv2 as cv
from multiprocessing import Pool
from functools import partial
from collections import defaultdict


# helpers
def _build_tree_level_enough_cache(ids, des, k, criteria, attempts):
    try:
        _, indices, clusters = cv.kmeans(des, k, None, criteria, attempts, cv.KMEANS_PP_CENTERS)
        return npi.group_by(indices, ids)[1], npi.group_by(indices, des)[1], clusters
    except Exception as e:
        print("Error while building level", e)
        return None, None, None


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


def _process_node_initial_scoring(centroids, des, index):
    return utils.get_closest_indexes(centroids, des) + index


def _get_closest_leafs(tree, current_des, K, L):
    """
    For every descriptor search its closest (approximate) leave by traversing the tree.

    First level: For every des calculate which centroid is closest. Group all des that are closest to the same centroid.
    Next level:
        For every centroid take all des that traversed the tree to that centroid.
        For all those des, calculate which centroid it's closest to 'below' the current centroid.
        Group all des that are closest to the same centroid.
    Repeat L times, so that the closest leaf node for every des is found.

    The concept is pretty simple, but how to keep track of the indices is tricky to understand.
    """
    current_indices = utils.get_closest_indexes(tree.get_centroids(0), current_des)
    current_indices, current_des = npi.group_by(current_indices, current_des)
    l = 1
    current_indices += 1
    while l < L:
        new_centroids = tree.get_centroids([current_indices])
        current_indices *= K
        current_indices += 1
        des_next, indices_next = list(), list()
        for centroids_parent, des_parent, indices_parent in zip(new_centroids, current_des, current_indices):  # TODO: try multiprocessing
            indices_child = _process_node_initial_scoring(centroids_parent, des_parent, indices_parent)
            indices_child_grouped, des_child_grouped = npi.group_by(indices_child, des_parent)
            indices_next.extend(indices_child_grouped)
            des_next.extend(des_child_grouped)
        current_indices = np.array(indices_next, np.uint32)
        current_des = des_next
        l += 1
    return current_indices - np.sum([np.power(K, np.arange(L))]), [len(i) for i in current_des]



class KMeansTree:
    def __init__(self, file="tree.p", nb_images=-1, k=10, l=6, criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 5, 0.5), attempts=5, vector_dimensions=128):
        self.file = file
        self.tree = utils.get_pickled(file)
        if self.tree is None:  # if the file doesn not exist already
            if nb_images == -1:
                raise ValueError("nb_images not mentioned.")
            self.tree = (np.zeros([np.sum(np.power(k, np.arange(l))), k, vector_dimensions], dtype=np.float32),
                         np.zeros(k ** l, dtype="float32, object, object"),
                         (k, l, criteria, attempts, nb_images))
            self.criteria = criteria
            self.attempts = attempts
            self.k = k
            self.l = l
            self.N = nb_images
            self.store()
        else:
            self.k, self.l, self.criteria, self.attempts, self.N = self.tree[2]

    def get_k(self):
        return self.k

    def get_l(self):
        return self.l

    def get_centroids(self, indices):
        if isinstance(indices, list):
            indices = tuple(indices)
        return self.tree[0][indices]

    def get_leaves(self, leaf_indices):
        if isinstance(leaf_indices, list):
            leaf_indices = tuple(leaf_indices)
        return self.tree[1][leaf_indices]

    def store(self):
        utils.pickle_data(self.tree, self.file)

    def build_node_from_given_data(self, des, level=0, index=0):
        # take given node
        tree_index = index + np.sum(np.power(self.k, np.arange(level)))
        node = self.tree[0][tree_index]
        # check if node not already build
        if node[0, 0] != 0:
            print("Node at level {} and index {} already build.".format(level, index))
            return None
        print("Building node at level {} and index {}.".format(level, index))

        # calculate clusters
        clusters = cv.kmeans(des, self.k, None, self.criteria, self.attempts, cv.KMEANS_PP_CENTERS)[2]

        # update clusters
        self.tree[0][tree_index] = clusters

        # store result!
        self.store()
        return clusters

    def build_branch(self, ids_in, des_in, level=0, index=0, attempts_level=10):
        # check if given node not already build
        if self.tree[0][index + np.sum(np.power(self.k, np.arange(level))), 0, 0] != 0:
            print("Node at level {} and index {} already build.".format(level, index))
            return None

        initial_level = level
        print("Building in branch: level {}, index {}, nb of des to process {:_}.".format(level, index, len(ids_in)))

        # build first level
        ids, des, clusters = _build_tree_level_enough_cache(ids_in, des_in, self.k, self.criteria, self.attempts)

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
                if i is None:
                    raise ValueError("An id list is None, this means an error occurred.")
                ids.extend(i)
                des.extend(d)
                all_clusters.append(clusters)
            del res
            level += 1

        # update clusters
        print("Updating clusters.")
        if initial_level > 0:
            for j, cluster in zip(_cluster_nbs(initial_level, index, self.k, self.l), all_clusters):
                self.tree[0][j] = cluster
        else:
            self.tree = (all_clusters[0], self.tree[1], self.tree[2])

        # update leaves
        print("Updating leaves.")
        for leaf_nb, leaf_ids in zip(_leaf_nbs(initial_level, index, self.k, self.l), ids):
            unique, counts = np.unique(leaf_ids, return_counts=True)
            self.tree[1][leaf_nb] = np.array(0, dtype=np.float32), unique, counts.astype(np.uint32)
        # store result!
        print("Storing results.")
        self.store()

    def finalise(self):
        # check if not already done!
        if self.tree[1][-1][2].dtype == np.float32:
            print("Tree already finalised.")
            return None

        all_items = defaultdict(lambda: np.array(0, dtype=np.float32))

        # Entropy
        for leaf_nb in np.arange(self.k ** self.l):
            leaf = self.tree[1][leaf_nb]

            w = np.log(self.N / len(leaf[2]))
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

    def initial_scoring(self, des):
        """
        Perform the initial scoring.
        """
        leaf_indices, nb_of_des = _get_closest_leafs(self, des, self.k, self.l)
        leaves = self.get_leaves(leaf_indices)

        weights = [i[0] for i in leaves]
        values = np.multiply(nb_of_des, weights, dtype=np.float32)  # weighting
        values /= np.sum(values)  # L-1 normalisation

        # Calculate the score for every db index on every leaf node
        indexes, scores = list(), list()
        for leave, value in zip(leaves, values):
            indexes.extend(leave[1])
            scores.extend(np.subtract(leave[2] + value, np.abs(leave[2] - value), dtype=np.float32))

        # sum all scores for every index
        indexes, scores = npi.group_by(indexes, scores)  # this should be way faster!!!
        return indexes, [np.sum(i) for i in scores]
