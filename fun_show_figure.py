from py_libs import *
# import fast_sweeping as sf


def show_figure(pairs: list,
                pos_receivers: list[tuple[float, float]],
                pos_sources: list[tuple[float, float]],
                show_path: bool = False,
                legend: bool = False,
                title: str = None,
                comp_field = False):

    fig, axs = plt.subplots(2, int(np.ceil(len(pairs)/2)), sharey=True, sharex=True, figsize=(10,3))
    axs = axs.reshape(-1,)
    standard_field = pairs[0].field
    if not comp_field:
        vmin = np.amin(standard_field)
        vmax = np.amax(standard_field)
    else:
        vmin = -20
        vmax = 20
    all_rays = []
    for idx, pair in enumerate(pairs):
        environment, rays = pair, pair.rays
        if comp_field:
            show_field = environment.field - standard_field
        else:
            show_field = None
        environment.plot_curves(rays, pos_sources, pos_receivers, axs[idx], show_path=show_path, legend=legend,
                                vs=(vmin, vmax), show_field=show_field
                                )
        axs[idx].set_xlabel(environment.field_name)
        axs[idx].invert_yaxis()
    # axs[-1].set_xlabel('Comparison')
    if title is not None:
        fig.suptitle(title)
    # fig.delaxes(axs[3])
    manager = plt.get_current_fig_manager()
    manager.window.wm_geometry("950x800+485+40")
    # manager.full_screen_toggle()
    plt.show()
