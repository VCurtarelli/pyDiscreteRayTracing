from class_Environment import Environment
from funs_generate_env_fields import generate_temp_field, generate_sal_field
import numpy as np
import copy
import matplotlib.pyplot as plt
import pickle as pkl


def calculate_eofs(profiles, threshold=0.995, return_all=False):
    n_profiles = profiles.shape[1]
    avg_profile = np.mean(profiles, axis=1)
    covariance_matrix = np.zeros((avg_profile.size, avg_profile.size))
    for k in range(n_profiles):
        deviation = (profiles[:, k] - avg_profile).reshape(-1, 1)
        delta_cov = deviation @ deviation.T
        covariance_matrix += delta_cov / n_profiles

    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    eigenvalues = np.real(eigenvalues)
    eigsort = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[eigsort]
    eigenvectors = eigenvectors[:, eigsort]
    threshold_count = 0
    while True:
        threshold_count += 1
        variability = sum(np.abs(eigenvalues[:threshold_count]))/sum(np.abs(eigenvalues))
        if variability > threshold:
            break
    eigenbase = eigenvectors[:, :threshold_count]

    cum_eigenvalues = np.cumsum(np.abs(eigenvalues)) / np.sum(np.abs(eigenvalues))
    t_covariance_matrix = covariance_matrix
    for k in range(avg_profile.size):
        t_covariance_matrix[:, k] /= t_covariance_matrix[0, k]
    if return_all:
        return avg_profile, eigenbase, eigenvectors, eigenvalues
    else:
        return avg_profile, eigenbase


def main():
    threshold = 0.9995
    with open('input/select_block_velocity_field.dat', 'rb') as f:
        data = pkl.load(f)
        velocity_profiles = data['velocity']
        depths = data['depth']
    ny = depths.size
    n_profiles = velocity_profiles.shape[1]

    slowness_profiles = 1/velocity_profiles

    mean_profile, eof_base, eigenvectors, eigenvalues = calculate_eofs(slowness_profiles, threshold=threshold, return_all=True)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    for i in range(n_profiles):
        axs[0].plot(velocity_profiles[:, i], depths)
    for i in range(ny):
        if i < eof_base.shape[1]:
            axs[1].plot(eof_base[:, i], depths, linewidth=4/(1+3*i/eof_base.shape[1]))
        else:
            axs[1].plot(eigenvectors[:, i], depths, alpha=0.05, linestyle='--')

    [ax.invert_yaxis() for ax in axs]
    # axs[1].set_xlim(-100, 100)

    data = {'slowness': slowness_profiles,
            'depth': depths,
            'mean': mean_profile,
            'eigenbase': eof_base,
            'threshold': threshold,
            'eigenvectors': eigenvectors,
            'eigenvalues': eigenvalues,}

    with open('midput/slowness_field_eofs.dat', 'wb') as f:
        pkl.dump(data, f)
    plt.show()


main()