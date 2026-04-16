import numpy as np
import matplotlib.pyplot as plt

def position_devices(mode, env_params, dev_params):
    def _s_layer_r_layer(_env_params, _dev_params):
        _nx = _env_params.num_cells_x
        _ny = _env_params.num_cells_y
        _ns = _dev_params.n_sources
        _nr = _dev_params.n_receivers
        _ofs = _dev_params.offset

        _pos_sx = []
        _pos_sy = []
        _pos_rx = []
        _pos_ry = []

        _pos_sx.append(np.linspace(_ofs, _nx - 1 - _ofs, _ns))
        _pos_sy.append(np.array([_ny - 1 - _ofs for _ in range(_ns)]))
        _pos_rx.append(np.linspace(_ofs, _nx - 1 - _ofs, _nr))
        _pos_ry.append(np.array([_ofs for _ in range(_nr)]))

        return _pos_sx, _pos_sy, _pos_rx, _pos_ry

    def _s_layer_r_blob(_env_params, _dev_params):
        _nx = _env_params.num_cells_x
        _ny = _env_params.num_cells_y
        _ns = _dev_params.n_sources
        _nr = _dev_params.n_receivers
        _ofs = _dev_params.offset
        _n_travels = _dev_params.n_travels
        _h = _env_params.height

        _pos_sx = []
        _pos_sy = []
        _pos_rx = []
        _pos_ry = []

        _pos_sx.append(np.linspace(_ofs, _nx - 1 - _ofs, _ns))
        _pos_sy.append(np.array([_ny-1-_ofs for _ in range(_ns)]))
        for travel_idx in range(_n_travels):
            travel_idx = travel_idx / (_n_travels - 1)
            _pos_rx.append(np.linspace(_ofs, _nx - 1 - _ofs, _nr))
            _pos_ry.append(np.array([travel_idx * (500 / _h * (_ny - 1)) for _ in range(_nr)]))

        return _pos_sx, _pos_sy, _pos_rx, _pos_ry

    def _s_layer_r_blob_column(_env_params, _dev_params):
        _nx = _env_params.num_cells_x
        _ny = _env_params.num_cells_y
        _ns = _dev_params.n_sources
        _nr = _dev_params.n_receivers
        _ofs = _dev_params.offset
        _n_travels = _dev_params.n_travels
        _h = _env_params.height

        _pos_sx = []
        _pos_sy = []
        _pos_rx = []
        _pos_ry = []

        _pos_sx.append(np.linspace(_ofs, _nx - 1 - _ofs, _ns))
        _pos_sy.append(np.array([_ny - 1 - _ofs for _ in range(_ns)]))

        for travel_idx in range(_n_travels):
            travel_idx = travel_idx / (_n_travels-1)
            _pos_rx.append(np.linspace(_ofs, _nx - 1 - _ofs, _nr))
            _pos_ry.append(np.array([travel_idx * (500 / _h * (_ny-1)) for _ in range(_nr)]))
        _pos_rx.append(np.array([_nx//2 for _ in range(_ns)]))
        _pos_ry.append(np.linspace(500 / _h * (_ny-1), 1000 / _h * (_ny-1), _ns))

        return _pos_sx, _pos_sy, _pos_rx, _pos_ry

    def _s_layer_shallowcolumn_r_blob(_env_params, _dev_params):
        _nx = _env_params.num_cells_x
        _ny = _env_params.num_cells_y
        _ns = _dev_params.n_sources
        _nr = _dev_params.n_receivers
        _ofs = _dev_params.offset
        _n_travels = _dev_params.n_travels
        _h = _env_params.height

        _pos_sx = []
        _pos_sy = []
        _pos_rx = []
        _pos_ry = []

        _pos_sx.append(np.linspace(_ofs, _nx - 1 - _ofs, _ns))
        _pos_sy.append(np.array([_ny - 1 - _ofs for _ in range(_ns)]))
        _pos_sx.append(np.array([_ofs for _ in range(_n_travels)]))
        _pos_sy.append(np.array([travel_idx/(_n_travels-1) * (500 / _h * (_ny-1)) for travel_idx in range(_n_travels)]))
        for travel_idx in range(_n_travels):
            travel_idx = travel_idx / (_n_travels-1)
            _pos_rx.append(np.linspace(1+_ofs, _nx - 1 - _ofs, _nr))
            _pos_ry.append(np.array([travel_idx * (500 / _h * (_ny-1)) for _ in range(_nr)]))

        return _pos_sx, _pos_sy, _pos_rx, _pos_ry


    match mode:
        case 'layer/layer':
            psx, psy, prx, pry = _s_layer_r_layer(env_params, dev_params)
        case 'layer/blob':
            psx, psy, prx, pry = _s_layer_r_blob(env_params, dev_params)
        case 'layer/column+blob':
            psx, psy, prx, pry = _s_layer_r_blob_column(env_params, dev_params)
        case 'layer+shallowcolumn/blob':
            psx, psy, prx, pry = _s_layer_shallowcolumn_r_blob(env_params, dev_params)
        case _:
            raise NotImplementedError("The mode {} is not implemented.".format(mode))
    pname = mode
    psx = np.concatenate(psx)
    psy = np.concatenate(psy)
    prx = np.concatenate(prx)
    pry = np.concatenate(pry)

    _nx = env_params.num_cells_x
    _ny = env_params.num_cells_y
    _w = env_params.width
    _h = env_params.height
    cw = _w / _nx
    ch = _h / _ny
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
