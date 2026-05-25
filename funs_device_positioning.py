import numpy as np
import matplotlib.pyplot as plt

def position_devices(mode, env_params, dev_params):
    def _s_layer_r_layer(_env_params, _dev_params):  # Source: Layer | Receiver:Layer
        _nx = _env_params.num_cells_x
        _ny = _env_params.num_cells_y
        _ns = _dev_params.n_sources
        _nr = _dev_params.n_receivers

        _pos_sx = []
        _pos_sy = []
        _pos_rx = []
        _pos_ry = []

        # Sources on bottom
        _pos_sx.append(np.linspace(0, _nx - 1, _ns))
        _pos_sy.append(np.array([_ny - 1 for _ in range(_ns)]))

        # Receivers on surface
        _pos_rx.append(np.linspace(0, _nx - 1, _nr))
        _pos_ry.append(np.array([0 for _ in range(_nr)]))

        return _pos_sx, _pos_sy, _pos_rx, _pos_ry

    def _s_layer_r_blob(_env_params, _dev_params):  # Source: Layer | Receiver:Blob
        _nx = _env_params.num_cells_x
        _ny = _env_params.num_cells_y
        _ns = _dev_params.n_sources
        _nr = _dev_params.n_receivers
        _n_travels = _dev_params.n_travels
        _h = _env_params.height
        _travels_dpth = _dev_params.travels_dpth

        _pos_sx = []
        _pos_sy = []
        _pos_rx = []
        _pos_ry = []

        # Sources on bottom
        _pos_sx.append(np.linspace(0, _nx - 1, _ns))
        _pos_sy.append(np.array([_ny-1-0 for _ in range(_ns)]))

        # Receivers in layer on surface
        for travel_idx in range(_n_travels):
            travel_idx = travel_idx / (_n_travels - 1)
            _pos_rx.append(np.linspace(0, _nx - 1, _nr))
            _pos_ry.append(np.array([travel_idx * (_travels_dpth / _h * (_ny - 1)) for _ in range(_nr)]))

        return _pos_sx, _pos_sy, _pos_rx, _pos_ry

    def _s_layer_r_layer_diagonal(_env_params, _dev_params):  # Source: Layer | Receiver:Layer+Diagonal
        _nx = _env_params.num_cells_x
        _ny = _env_params.num_cells_y
        _ns = _dev_params.n_sources
        _nr = _dev_params.n_receivers
        _n_travels = _dev_params.n_travels
        _h = _env_params.height
        _travels_dpth = _dev_params.travels_dpth

        _pos_sx = []
        _pos_sy = []
        _pos_rx = []
        _pos_ry = []

        # Sources on bottom
        _pos_sx.append(np.linspace(0, _nx - 1, _ns))
        _pos_sy.append(np.array([_ny - 1 for _ in range(_ns)]))

        # Receivers on surface
        _nrt = _nr
        _nrs = _nr
        _pos_rx.append(np.linspace(0, _nx - 1, _nrt))
        _pos_ry.append(np.array([0 for _ in range(_nrt)]))
        # Receivers on diagonal
        _pos_rx.append(np.linspace(0, _nx - 1, _nrt))
        _pos_ry.append(np.linspace(0, (_ny - 1 - 0) * _travels_dpth / _h, _nrs + 1)[1:])

        return _pos_sx, _pos_sy, _pos_rx, _pos_ry

    def _s_layer_r_layer_appendix(_env_params, _dev_params):  # Source: Layer | Receiver:Layer+Column
        _nx = _env_params.num_cells_x
        _ny = _env_params.num_cells_y
        _ns = _dev_params.n_sources
        _nr = _dev_params.n_receivers
        _n_travels = _dev_params.n_travels
        _h = _env_params.height
        _travels_dpth = _dev_params.travels_dpth

        _pos_sx = []
        _pos_sy = []
        _pos_rx = []
        _pos_ry = []

        # Sources on bottom
        _pos_sx.append(np.linspace(0, _nx - 1, _ns))
        _pos_sy.append(np.array([_ny - 1 for _ in range(_ns)]))

        # Receivers on surface
        _pos_rx.append(np.linspace(0, _nx - 1, _nr))
        _pos_ry.append(np.array([0 for _ in range(_nr)]))
        # Receivers on side
        _pos_rx.append(np.array([int(_nx//2) for _ in range(_nr)]))
        _pos_ry.append(np.linspace(0, (_ny - 1 - 0) * _travels_dpth/_h, _nr + 1)[1:])

        return _pos_sx, _pos_sy, _pos_rx, _pos_ry

    def _s_layer_vertical_r_layer(_env_params, _dev_params):  # Source: Layer+Vertical | Receiver:Layer
        _nx = _env_params.num_cells_x
        _ny = _env_params.num_cells_y
        _ns = _dev_params.n_sources
        _nr = _dev_params.n_receivers
        _n_travels = _dev_params.n_travels
        _h = _env_params.height

        _pos_sx = []
        _pos_sy = []
        _pos_rx = []
        _pos_ry = []

        # Sources on bottom
        _nsb = _ns
        _nss = _n_travels
        _pos_sx.append(np.linspace(0, _nx - 1, _nsb))
        _pos_sy.append(np.array([_ny - 1 for _ in range(_nsb)]))
        # Sources on side
        _pos_sx.append(np.array([int(_nx//2) for _ in range(_nss)]))
        _pos_sy.append(np.linspace(0, _ny - 1, _nss+2)[1:-1])

        # Receivers on surface
        _pos_rx.append(np.linspace(0, _nx - 1, _nr))
        _pos_ry.append(np.array([0 for _ in range(_nr)]))

        return _pos_sx, _pos_sy, _pos_rx, _pos_ry

    def _s_layer_vertical_r_layer_vertical(_env_params, _dev_params):  # Source: Layer | Receiver:Layer+Short-column
        _nx = _env_params.num_cells_x
        _ny = _env_params.num_cells_y
        _ns = _dev_params.n_sources
        _nr = _dev_params.n_receivers
        _n_travels = _dev_params.n_travels
        _h = _env_params.height

        _pos_sx = []
        _pos_sy = []
        _pos_rx = []
        _pos_ry = []

        # Sources on bottom
        _nsb = _ns
        _nss = _n_travels
        _pos_sx.append(np.linspace(0, _nx - 1, _nsb))
        _pos_sy.append(np.array([_ny - 1 for _ in range(_nsb)]))
        # Sources on side
        _pos_sx.append(np.array([_nx - 1 for _ in range(_nss)]))
        _pos_sy.append(np.linspace(0, _ny - 1, _nss+2)[1:-1])

        # Receivers on surface
        _nrt = _nr
        _nrs = _n_travels
        _pos_rx.append(np.linspace(0, _nx - 1, _nrt))
        _pos_ry.append(np.array([0 for _ in range(_nrt)]))
        # Receivers on side
        _pos_rx.append(np.array([0 for _ in range(_nrs)]))
        _pos_ry.append(np.linspace(0, _ny - 1, _nrs+2)[1:-1])

        return _pos_sx, _pos_sy, _pos_rx, _pos_ry

    def _s_layer_r_layer_vertical(_env_params, _dev_params):  # Source: Layer+Short-column | Receiver:Layer+Short-column
        _nx = _env_params.num_cells_x
        _ny = _env_params.num_cells_y
        _ns = _dev_params.n_sources
        _nr = _dev_params.n_receivers
        _n_travels = _dev_params.n_travels
        _h = _env_params.height

        _pos_sx = []
        _pos_sy = []
        _pos_rx = []
        _pos_ry = []

        # Sources on bottom
        _pos_sx.append(np.linspace(0, _nx - 1, _ns))
        _pos_sy.append(np.array([_ny - 1 for _ in range(_ns)]))

        # Receivers on surface
        _nrt = _nr
        _nrs = _n_travels
        _pos_rx.append(np.linspace(0, _nx - 1, _nrt))
        _pos_ry.append(np.array([0 for _ in range(_nrt)]))
        # Receivers on side
        _pos_rx.append(np.array([int(_nx//2) for _ in range(_nrs)]))
        _pos_ry.append(np.linspace(0, _ny - 1, _nrs + 2)[1:-1])

        return _pos_sx, _pos_sy, _pos_rx, _pos_ry

    def _s_layer_r_bracket(_env_params, _dev_params):  # Source: Layer+Short-column | Receiver:Layer+Short-column
        _nx = _env_params.num_cells_x
        _ny = _env_params.num_cells_y
        _ns = _dev_params.n_sources
        _nr = _dev_params.n_receivers
        _n_travels = _dev_params.n_travels
        _h = _env_params.height
        _travels_dpth = _dev_params.travels_dpth

        _pos_sx = []
        _pos_sy = []
        _pos_rx = []
        _pos_ry = []

        # Sources on bottom
        _pos_sx.append(np.linspace(0, _nx - 1, _ns))
        _pos_sy.append(np.array([_ny - 1 for _ in range(_ns)]))

        # Receivers on surface
        _pos_rx.append(np.linspace(0, _nx - 1, _nr))
        _pos_ry.append(np.array([0 for _ in range(_nr)]))
        # Receivers on left-side
        _pos_rx.append(np.array([0 for _ in range(_nr)]))
        _pos_ry.append(np.linspace(0, (_ny - 1 - 0) * _travels_dpth/_h, _nr+1)[1:])
        # Receivers on right-side
        _pos_rx.append(np.array([_nx - 1 for _ in range(_nr)]))
        _pos_ry.append(np.linspace(0, (_ny - 1 - 0) * _travels_dpth/_h, _nr+1)[1:])

        return _pos_sx, _pos_sy, _pos_rx, _pos_ry

    def _s_layer_r_cap(_env_params, _dev_params):  # Source: Layer+Column | Receiver:Layer+Column
        _nx = _env_params.num_cells_x
        _ny = _env_params.num_cells_y
        _ns = _dev_params.n_sources
        _nr = _dev_params.n_receivers
        _n_travels = _dev_params.n_travels
        _h = _env_params.height

        _pos_sx = []
        _pos_sy = []
        _pos_rx = []
        _pos_ry = []

        # Sources on bottom
        _ns = _ns
        _pos_sx.append(np.linspace(0, _nx - 1, _ns))
        _pos_sy.append(np.array([_ny - 1 for _ in range(_ns)]))

        # Receivers on surface
        _nrt = _nr
        _nrs = _n_travels
        _pos_rx.append(np.linspace(0, _nx - 1, _nrt))
        _pos_ry.append(np.array([0 for _ in range(_nrt)]))
        # Receivers on left-side
        _pos_rx.append(np.array([0 for _ in range(_nrs)]))
        _pos_ry.append(np.linspace(0, _ny - 1, _nrs+2)[1:-1])
        # Receivers on right-side
        _pos_rx.append(np.array([_nx - 1 for _ in range(_nrs)]))
        _pos_ry.append(np.linspace(0, _ny - 1, _nrs+2)[1:-1])

        return _pos_sx, _pos_sy, _pos_rx, _pos_ry

    def _s_cup_r_layer(_env_params, _dev_params):  # Source: Cup | Receiver: Layer
        _nx = _env_params.num_cells_x
        _ny = _env_params.num_cells_y
        _ns = _dev_params.n_sources
        _nr = _dev_params.n_receivers
        _n_travels = _dev_params.n_travels
        _h = _env_params.height

        _pos_sx = []
        _pos_sy = []
        _pos_rx = []
        _pos_ry = []

        # Sources on bottom
        _nsb = _ns
        _nss = _n_travels
        _pos_sx.append(np.linspace(0, _nx - 1, _nsb))
        _pos_sy.append(np.array([_ny - 1 for _ in range(_nsb)]))
        # Sources on left-side
        _pos_sx.append(np.array([0 for _ in range(_nss)]))
        _pos_sy.append(np.linspace(0, _ny - 1, _nss+2)[1:-1])
        # Sources on right-side
        _pos_sx.append(np.array([_nx - 1 for _ in range(_nss)]))
        _pos_sy.append(np.linspace(0, _ny - 1, _nss+2)[1:-1])

        # Receivers on surface
        _pos_rx.append(np.linspace(0, _nx - 1, _nr))
        _pos_ry.append(np.array([0 for _ in range(_nr)]))

        return _pos_sx, _pos_sy, _pos_rx, _pos_ry

    def _s_layer_sparse_r_layer(_env_params, _dev_params):  # Source: Layer+Sparse | Receiver:Layer
        _nx = _env_params.num_cells_x
        _ny = _env_params.num_cells_y
        _ns = _dev_params.n_sources
        _nr = _dev_params.n_receivers
        _n_travels = _dev_params.n_travels
        _h = _env_params.height

        _pos_sx = []
        _pos_sy = []
        _pos_rx = []
        _pos_ry = []

        # Sources on bottom
        _nsb = _ns
        _nss = _n_travels
        _pos_sx.append(np.linspace(0, _nx - 1, _nsb))
        _pos_sy.append(np.array([_ny - 1 for _ in range(_nsb)]))
        # Sources on center
        rng = np.random.default_rng(2)
        _pos_sx.append(rng.integers(1, _nx-2, _nss))
        _pos_sy.append(np.linspace(0, _ny - 1, _nss+2)[1:-1])

        # Receivers on surface
        _pos_rx.append(np.linspace(0, _nx - 1, _nr))
        _pos_ry.append(np.array([0 for _ in range(_nr)]))

        return _pos_sx, _pos_sy, _pos_rx, _pos_ry

    def _s_layer_r_layer_sparse(_env_params, _dev_params):  # Source: Layer | Receiver:Layer+Sparse
        _nx = _env_params.num_cells_x
        _ny = _env_params.num_cells_y
        _ns = _dev_params.n_sources
        _nr = _dev_params.n_receivers
        _n_travels = _dev_params.n_travels
        _h = _env_params.height

        _pos_sx = []
        _pos_sy = []
        _pos_rx = []
        _pos_ry = []

        # Sources on bottom
        _pos_sx.append(np.linspace(0, _nx - 1, _ns))
        _pos_sy.append(np.array([_ny - 1 for _ in range(_ns)]))

        # Receivers on surface
        _nrt = _nr
        _nrs = _n_travels
        _pos_rx.append(np.linspace(0, _nx - 1, _nrt))
        _pos_ry.append(np.array([0 for _ in range(_nrt)]))
        # Sources on center
        rng = np.random.default_rng(2)
        _pos_rx.append(rng.integers(1, _nx-2, _nrs))
        _pos_ry.append(np.linspace(0, _ny - 1, _nrs+2)[1:-1])

        return _pos_sx, _pos_sy, _pos_rx, _pos_ry


    match mode:
        case 'layer/layer':
            psx, psy, prx, pry = _s_layer_r_layer(env_params, dev_params)
        case 'layer/blob':
            psx, psy, prx, pry = _s_layer_r_blob(env_params, dev_params)
        case 'layer/layer+diagonal':
            psx, psy, prx, pry = _s_layer_r_layer_diagonal(env_params, dev_params)
        case 'layer/layer+appendix':
            psx, psy, prx, pry = _s_layer_r_layer_appendix(env_params, dev_params)
        case 'layer/bracket':
            psx, psy, prx, pry = _s_layer_r_bracket(env_params, dev_params)
        case 'layer+column/layer+column':
            psx, psy, prx, pry = _s_layer_vertical_r_layer_vertical(env_params, dev_params)
        case 'layer/cap':
            psx, psy, prx, pry = _s_layer_r_cap(env_params, dev_params)
        case 'layer+column/layer':
            psx, psy, prx, pry = _s_layer_vertical_r_layer(env_params, dev_params)
        case 'layer/layer+column':
            psx, psy, prx, pry = _s_layer_r_layer_vertical(env_params, dev_params)
        case 'layer+vertical/layer+vertical':
            psx, psy, prx, pry = _s_layer_vertical_r_layer_vertical(env_params, dev_params)
        case 'layer+vertical/layer':
            psx, psy, prx, pry = _s_layer_vertical_r_layer(env_params, dev_params)
        case 'layer/layer+vertical':
            psx, psy, prx, pry = _s_layer_r_layer_vertical(env_params, dev_params)
        case 'layer+sparse/layer':
            psx, psy, prx, pry = _s_layer_sparse_r_layer(env_params, dev_params)
        case 'layer/layer+sparse':
            psx, psy, prx, pry = _s_layer_r_layer_sparse(env_params, dev_params)
        case 'cup/layer':
            psx, psy, prx, pry = _s_cup_r_layer(env_params, dev_params)
        case _:
            raise NotImplementedError("The mode {} is not implemented.".format(mode))
    pname = mode
    psx = np.concatenate(psx)
    psy = np.concatenate(psy)
    prx = np.concatenate(prx)
    pry = np.concatenate(pry)

    nx = env_params.num_cells_x
    ny = env_params.num_cells_y
    w = env_params.width
    h = env_params.height
    cw = w / nx
    ch = h / ny
    psx = cw * (0.5 + np.around(psx))
    psy = ch * (0.5 + np.around(psy))
    prx = cw * (0.5 + np.around(prx))
    pry = ch * (0.5 + np.around(pry))

    psx, psy = tuple(np.array(list(set(zip(list(psx), list(psy))))).T)
    prx, pry = tuple(np.array(list(set(zip(list(prx), list(pry))))).T)
    return psx, psy, prx, pry, pname


def export_pos_devices(pos_sources, pos_receivers, direc, width, height):
    header = 'x_pos,y_pos'
    pos_sources = [(ps[0], ps[1]) for ps in pos_sources]
    pos_receivers = [(pr[0], pr[1]) for pr in pos_receivers]
    txt_sources = '\n'.join([header] + ['{},{}'.format(*pos) for pos in pos_sources])
    txt_receivers = '\n'.join([header] + ['{},{}'.format(*pos) for pos in pos_receivers])

    with open(direc + '/pos_sources.csv', 'w') as f:
        f.write(txt_sources)
        f.close()
    with open(direc + '/pos_receivers.csv', 'w') as f:
        f.write(txt_receivers)
        f.close()
    pass
