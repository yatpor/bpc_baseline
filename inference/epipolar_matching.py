import numpy as np
from scipy.optimize import linear_sum_assignment


def epipolar_error(pt1, pt2, F):
    """Compute symmetric epipolar distance between two points."""
    pt1_h = np.array([pt1[0], pt1[1], 1.0])
    pt2_h = np.array([pt2[0], pt2[1], 1.0])

    l2 = F @ pt1_h
    l1 = F.T @ pt2_h

    l2 /= np.linalg.norm(l2[:2])
    l1 /= np.linalg.norm(l1[:2])

    d1 = abs(np.dot(l1, pt1_h))
    d2 = abs(np.dot(l2, pt2_h))
    return 0.5 * (d1 + d2)


def epipolar_error_full(pt1, pt2, pt3, F12, F13, F23):
    """Compute the total epipolar error across three cameras."""
    e12 = epipolar_error(pt1, pt2, F12)
    e13 = epipolar_error(pt1, pt3, F13)
    e23 = epipolar_error(pt2, pt3, F23)
    return e12 + e13 + e23


def compute_cost_matrix(dets1, dets2, dets3, F12, F13, F23):
    """Compute NxMxP cost matrix for bounding box matches."""
    N, M, P = len(dets1), len(dets2), len(dets3)
    cost = np.zeros((N, M, P), dtype=np.float32)

    for i in range(N):
        pt1 = dets1[i]['bb_center']
        for j in range(M):
            pt2 = dets2[j]['bb_center']
            for k in range(P):
                pt3 = dets3[k]['bb_center']
                cost[i, j, k] = epipolar_error_full(pt1, pt2, pt3, F12, F13, F23)
    return cost


def match_objects(cost_matrix, threshold):
    """Match objects using the Hungarian algorithm."""
    N, M, P = cost_matrix.shape
    matched = []
    flattened = cost_matrix.reshape(N * M, P)
    row_idx, col_idx = linear_sum_assignment(flattened)

    for r, c in zip(row_idx, col_idx):
        val = flattened[r, c]
        if val < threshold:
            i = r // M
            j = r % M
            k = c
            matched.append((i, j, k))
    return matched
