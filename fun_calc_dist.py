from py_libs import *


def calc_closest_point(receiver, path, return_params=False):
    pr = np.array(receiver).astype(float).reshape(-1, 1)
    if len(path) > 1:
        dists_along_path = []
        for position in path:
            dist_to_pos = float(norm(np.array(receiver) - np.array(position)))
            dists_along_path.append(dist_to_pos)
        # print()
        # print('a', np.where(dists_along_path == np.sort(dists_along_path)[0]))
        closest = np.where(dists_along_path == np.sort(dists_along_path)[0])[0][0]
        closest_a = np.clip(closest + 1, 0, len(path)-1)
        closest_b = np.clip(closest - 1, 0, len(path)-1)
        closes = [closest_a, closest_b]
        # print('b', closest, closest_a, closest_b)
        dists = []
        ts = []
        pps = []
        # print(path)
        for cls in closes:
            p0 = np.array(path[closest]).astype(float).reshape(-1, 1)
            p1 = np.array(path[cls]).astype(float).reshape(-1, 1)
            # print(cls, closest)
            # print(p0, p1)
            if np.sum((p1 - p0) ** 2) == 0:
                if np.sum((p0 - pr) * (p1 - p0)) == 0:
                    t = 0
                else:
                    print("ERROR: UNCALCULABLE t")
                    t = np.inf
            else:
                t = np.sum((pr - p0) * (p1 - p0)) / np.sum((p1 - p0) ** 2)
            pp = p0 + t * (p1 - p0)
            dist = norm(pp - pr)
            dists.append(dist)
            ts.append(t)
            pps.append(pp)
            # print(ts)
        if any([0<=t<=1 for t in ts]):
            # print('valid t')
            ts = [t if t > 0 else np.inf for t in ts]
            idx = ts.index(min(ts))
        else:
            idx = dists.index(min(dists))
        pp = pps[idx]
        t = ts[idx]
        cls = closes[idx]
    else:
        pp = np.array(path[0]).reshape(-1, 1)
        closest = 0
        cls = 0
        t = -1
    if return_params:
        return pp, closest, cls, t
    else:
        return pp


def calc_dist(receiver, source, path):
    pr = np.array(receiver).astype(float).reshape(-1, 1)
    pp = calc_closest_point(receiver, path)
    ps = np.array(source).astype(float).reshape(-1, 1)
    pr -= ps
    pp -= ps
    # print(pp)
    # print(pr)
    # print(pp.T @ np.array([[1], [1j]]))
    # print(pr.T @ np.array([[1], [1j]]))
    # print(np.rad2deg(np.angle(pp.T @ np.array([[1], [1j]]))))
    # print(np.rad2deg(np.angle(pr.T @ np.array([[1], [1j]]))))
    lin_dist = norm(pp - pr)
    ang_dist = (np.angle((pp.T @ np.array([[1], [1j]]))) - np.angle((pr.T @ np.array([[1], [1j]])))).item()
    return lin_dist, ang_dist

# calc_dist([0.2, 0.3], [(0, 0), (1, 1), (1.5, 2), (2, 2.2)])