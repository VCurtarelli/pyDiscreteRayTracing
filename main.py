from fun_calc_travel_time import calc_travel_time
from fun_plot_curves import plot_curves
from py_params import *
from fun_receiver_search import receiver_search
from fun_ray_tracing import ray_tracing
from py_params import D_laplacian
from fun_calc_velocity_field import calc_velocity_field
from fun_cost_function import cost_function, gradient

## --------------------
# SOURCE-TO-RECEIVERS ANGLES ESTIMATION
ray_angles = []
times = []
J = np.zeros([n_receivers, num_cells])

for pos_receiver in pos_receivers:
    theta_0 = np.angle(complex(pos_receiver[0], pos_receiver[1]))  # initial guess for source-to-receiver ray's emission angle
    angles, _ = receiver_search(theta_0, velocity_field, num_cells_x, num_cells_y, width, height, pos_source,
                                    pos_receiver)
    ray_angles.append(angles[-1])

for idx in range(n_receivers):
    pos_receiver = pos_receivers[idx]
    angle = ray_angles[idx]

    time, Lengths = calc_travel_time(angle, velocity_field, num_cells_x, num_cells_y, width, height, pos_source,)
    times.append(time)
    # print("receiver {}: {:.2f}ms".format(idx, 1000 * time))
    J[idx, :] = Lengths.reshape(-1)

obs_times = times

run_iterative_J = True
if run_iterative_J:
    est_velocity_field = 1000*np.ones([num_cells_y, num_cells_x])     # test velocity field
    while True:
        fig, axs = plt.subplots(1, 3, sharey=True, figsize=(16,9))
        est_times = []
        ray_angles = []
        ## --------------------
        # CONSTRUCT CURRENT ITERATION J MATRIX
        z = 1/est_velocity_field.reshape(-1, 1)            # test velocity field, matrix form
        J = np.zeros([n_receivers, num_cells])
        for idx in range(n_receivers):
            pos_receiver = pos_receivers[idx]
            theta_0 = np.angle(complex(pos_receiver[0], pos_receiver[1]))
            angles, _ = receiver_search(theta_0, est_velocity_field, num_cells_x, num_cells_y, width, height, pos_source,pos_receiver)
            angle = angles[-1]
            ray_angles.append(angle)

            time, Lengths = calc_travel_time(angle, est_velocity_field, num_cells_x, num_cells_y, width, height, pos_source,)
            J[idx, :] = Lengths.reshape(-1)
        # print(J)
        U, s, Vh = svd(J)
        V = Vh.T
        s[s < 0.05*s[0]] = 0
        S_ = np.diag(s)
        S = np.zeros_like(J)
        rank_J = (s[s!=0]).size
        V2 = V[:, rank_J:]
        S[:min(n_receivers, num_cells), :min(n_receivers, num_cells)] = S_
        G = (np.eye(num_cells) - V2 @ pinv(D_laplacian @ V2) @ D_laplacian) @ V @ pinv(S) @ U.T
        z = G @ obs_times
        est_velocity_field = 1/z.reshape(num_cells_y, num_cells_x)

        ray_angles = []
        for idx in range(n_receivers):
            pos_receiver = pos_receivers[idx]
            theta_0 = np.angle(complex(pos_receiver[0], pos_receiver[1]))  # initial guess for source-to-receiver ray's emission angle
            angles, _ = receiver_search(theta_0, est_velocity_field, num_cells_x, num_cells_y, width, height, pos_source,
                                            pos_receiver)
            ray_angles.append(angles[-1])
            angle = ray_angles[idx]

            _, lengths, _, _ = ray_tracing(angle, est_velocity_field, num_cells_x, num_cells_y, width, height,
                                                  pos_source, )
            time, _ = calc_travel_time(angle, est_velocity_field, num_cells_x, num_cells_y, width, height,
                                             pos_source, )
            est_times.append(time)
        print(np.array(obs_times))
        print(np.array(est_times))

        print(cost_function(J, z, obs_times))
        print(gradient(J, z, obs_times))


        plot_curves(est_velocity_field, width, height, num_cells_x, num_cells_y, n_receivers, pos_source, pos_receivers,ax=axs[0])
        # plot_curves(velocity_field, width, height, num_cells_x, num_cells_y, n_receivers, pos_source, pos_receivers, ax=axs[1], show_field=np.abs(velocity_field-est_velocity_field))
        plot_curves(velocity_field, width, height, num_cells_x, num_cells_y, n_receivers, pos_source, pos_receivers, ax=axs[1], show_field=velocity_field)

        plot_curves(velocity_field, width, height, num_cells_x, num_cells_y, n_receivers, pos_source, pos_receivers, ax=axs[2], show_field=np.abs(velocity_field-est_velocity_field))
        plot_curves(est_velocity_field, width, height, num_cells_x, num_cells_y, n_receivers, pos_source, pos_receivers, ax=axs[2], show_field=False, color = np.array([180, 0, 180]) / 255)
        axs[0].set_xlabel('Estimate')
        axs[1].set_xlabel('Real')
        axs[2].set_xlabel('Comparison')

        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()
        plt.show()