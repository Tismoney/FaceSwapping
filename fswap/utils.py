from collections import OrderedDict
import numpy as np


FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 35)),
    ("jaw", (0, 17))
])

def swap_pts_fmt(pts):
    if isinstance(pts, list):
        return np.array(pts, np.int32)
    elif isinstance(pts, np.ndarray):
        if len(pts.shape) == 2:
            return [(int(x), int(y)) for x, y in pts]
        elif len(pts.shape) == 3:
            return [[(int(x), int(y)) for x, y in pts_] for pts_ in pts]


def indexing_trtiangles(triangles, points):
    pts1 = triangles[:, 0:2]
    pts2 = triangles[:, 2:4]
    pts3 = triangles[:, 4:6]

    ind_pts1 = [np.where((pt == points).all(axis=1))[0][0] for pt in pts1]
    ind_pts2 = [np.where((pt == points).all(axis=1))[0][0] for pt in pts2]
    ind_pts3 = [np.where((pt == points).all(axis=1))[0][0] for pt in pts3]

    triangles_indxs = [(i, j, k) for i, j, k, tr in zip(ind_pts1, ind_pts2, ind_pts3, triangles)]
    return triangles_indxs         
