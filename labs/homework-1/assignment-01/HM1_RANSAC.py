import numpy as np
from utils import draw_save_plane_with_points, normalize


if __name__ == "__main__":

    np.random.seed(0)
    # load data, total 130 points inlcuding 100 inliers and 30 outliers
    # to simplify this problem, we provide the number of inliers and outliers here

    noise_points = np.loadtxt("HM1_ransac_points.txt")

    # RANSAC
    # Please formulate the palnace function as:  A*x+B*y+C*z+D=0
    expected_outliers = 27  # the number of outliers in the data
    w = 1-expected_outliers/noise_points.shape[0]  # the probability that a point is an inlier
    n = 3  # number of points needed to fit a plane
    p = 0.999  # the probability of at least one hypothesis does not contain any outliers
    k = int(np.ceil(np.log(1-p)/np.log(1-w**n)))
    sample_time = k  # the minimal time that can guarantee the probability of at least one hypothesis does not contain any outliers is larger than 99.9%
    print("sample_time: ", sample_time)
    distance_threshold = 0.05

    # sample points group
    hypothesises_idx = np.random.choice(noise_points.shape[0], size=(sample_time, 3), replace=False)  # shape: (sample_time, 130)
    hypothesises = noise_points[hypothesises_idx].reshape(sample_time, 3, 3)  # shape: (sample_time, 3, 3)

    # estimate the plane with sampled points group
    vector1 = hypothesises[:, 1, :] - hypothesises[:, 0, :]  # shape: (sample_time, 3)
    vector2 = hypothesises[:, 2, :] - hypothesises[:, 0, :]  # shape: (sample_time, 3)
    normal_vectors = np.cross(vector1, vector2)  # shape: (sample_time, 3, 3)
    Ds = -np.sum(normal_vectors * hypothesises[:, 0, :], axis=1)  # shape: (sample_time,)
    A_B_C_Ds = np.concatenate((normal_vectors, Ds.reshape(-1, 1)), axis=1)  # shape: (sample_time, 4)

    # A_B_C_Ds[0] = np.array([0.2522, -0.7429, -0.1434, 0.6033])
    # evaluate inliers (with point-to-plance distance < distance_threshold)
    distance_all = np.abs(np.sum(A_B_C_Ds[:, None, :3] * noise_points, axis=2) + A_B_C_Ds[:, 3, None]) / \
        np.linalg.norm(A_B_C_Ds[:, :3], axis=1)[:, None]  # shape: (sample_time, 130)
    inlier_idx = distance_all < distance_threshold  # shape: (sample_time, 130)
    inliner_num = np.sum(inlier_idx, axis=1)  # shape: (sample_time,)
    print("inliner_num: ", inliner_num)
    best_hypothesis_idx = np.argmax(inliner_num)  # one integer
    best_hypothesis_inliners_num = inliner_num[best_hypothesis_idx]  # one integer
    best_hypothesis_inliners_idx = inlier_idx[best_hypothesis_idx]  # shape: (inlier_num,)
    best_hypothesis_inliners = noise_points[best_hypothesis_inliners_idx]  # shape: (inlier_num, 3)

    # minimize the sum of squared perpendicular distances of all inliers with least-squared method
    # 0 = Ap, p=[A, B, C, D].T
    A = np.concatenate((best_hypothesis_inliners[:, :2], np.ones((best_hypothesis_inliners_num, 1))), axis=1)  # shape: (inlier_num, 3)
    y = -best_hypothesis_inliners[:, 2]  # shape: (inlier_num,)
    p = np.linalg.lstsq(A, y, rcond=None)[0]  # shape: (3,)
    pf = np.array([p[0], p[1], 1, p[2]])  # shape: (4,)

    # draw the estimated plane with points and save the results
    # check the utils.py for more details
    # pf: [A,B,C,D] contains the parameters of palnace function  A*x+B*y+C*z+D=0
    pf = normalize(pf)
    draw_save_plane_with_points(pf, noise_points, "result/HM1_RANSAC_fig.png")
    np.savetxt("result/HM1_RANSAC_plane.txt", pf)
