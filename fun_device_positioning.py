import numpy as np

# from main import w, nx, h, ny


def position_devices(mode, params):
    def _s_layer_r_layer(_params):
        _nx, _ny, _w, _h, _ns, _nr, _ofs, _n_travels = _params

        _pos_sx = []
        _pos_sy = []
        _pos_rx = []
        _pos_ry = []

        _pos_sx.append(np.linspace(_ofs, _nx - 1 - _ofs, _ns))
        _pos_sy.append(np.array([_ofs for _ in range(_ns)]))
        _pos_rx.append(np.linspace(_ofs, _nx - 1 - _ofs, _nr))
        _pos_ry.append(np.array([_ny - 1 - _ofs for _ in range(_nr)]))

        return _pos_sx, _pos_sy, _pos_rx, _pos_ry

    def _s_layer_r_blob(_params):
        _nx, _ny, _w, _h, _ns, _nr, _ofs, _n_travels = _params

        _pos_sx = []
        _pos_sy = []
        _pos_rx = []
        _pos_ry = []

        _pos_sx.append(np.linspace(_ofs, _nx - 1 - _ofs, _ns))
        _pos_sy.append(np.array([_ofs for _ in range(_ns)]))
        for travel_idx in range(_n_travels):
            travel_idx = travel_idx / (_n_travels - 1)
            _pos_rx.append(np.linspace(_ofs, _nx - 1 - _ofs, _nr))
            _pos_ry.append(np.array([_ny - 1 - travel_idx * (500 / _h * (_ny - 1)) for _ in range(_nr)]))

        return _pos_sx, _pos_sy, _pos_rx, _pos_ry

    def _s_layer_r_blob_column(_params):
        _nx, _ny, _w, _h, _ns, _nr, _ofs, _n_travels = _params

        _pos_sx = []
        _pos_sy = []
        _pos_rx = []
        _pos_ry = []

        _pos_sx.append(np.linspace(_ofs, _nx - 1 - _ofs, _ns))
        _pos_sy.append(np.array([_ofs for _ in range(_ns)]))

        for travel_idx in range(_n_travels):
            travel_idx = travel_idx / (_n_travels-1)
            _pos_rx.append(np.linspace(_ofs, _nx - 1 - _ofs, _nr))
            _pos_ry.append(np.array([_ny - 1 - travel_idx * (500 / _h * (_ny-1)) for _ in range(_nr)]))
        _pos_rx.append(np.array([_nx//2 for _ in range(_ns)]))
        _pos_ry.append(np.linspace(_ny//4, (3*_ny)//4, _ns))

        return _pos_sx, _pos_sy, _pos_rx, _pos_ry

    def _s_layer_shallowcolumn_r_blob(_params):
        _nx, _ny, _w, _h, _ns, _nr, _ofs, _n_travels = _params

        _pos_sx = []
        _pos_sy = []
        _pos_rx = []
        _pos_ry = []

        _pos_sx.append(np.linspace(_ofs, _nx - 1 - _ofs, _ns))
        _pos_sy.append(np.array([_ofs for _ in range(_ns)]))
        _pos_sx.append(np.array([_ofs for _ in range(_n_travels)]))
        _pos_sy.append(np.array([_ny - 1 - travel_idx/(_n_travels-1) * (500 / _h * (_ny-1)) for travel_idx in range(_n_travels)]))
        for travel_idx in range(_n_travels):
            travel_idx = travel_idx / (_n_travels-1)
            _pos_rx.append(np.linspace(1+_ofs, _nx - 1 - _ofs, _nr))
            _pos_ry.append(np.array([_ny - 1 - travel_idx * (500 / _h * (_ny-1)) for _ in range(_nr)]))

        return _pos_sx, _pos_sy, _pos_rx, _pos_ry


    match mode:
        case 0:
            psx, psy, prx, pry = _s_layer_r_layer(params)
            pname = 'layer/layer'
        case 1:
            psx, psy, prx, pry = _s_layer_r_blob(params)
            pname = 'layer-blob'
        case 2:
            psx, psy, prx, pry = _s_layer_r_blob_column(params)
            pname = 'layer-column/blob'
        case 3:
            psx, psy, prx, pry = _s_layer_shallowcolumn_r_blob(params)
            pname = 'layer-shallowcolumn/blob'
        case _:
            raise NotImplementedError

    psx = np.concatenate(psx)
    psy = np.concatenate(psy)
    prx = np.concatenate(prx)
    pry = np.concatenate(pry)

    _nx, _ny, _w, _h, _, _, _, _ = params
    cw = _w / _nx
    ch = _h / _ny
    psx = cw * (0.5 + np.around(psx))
    psy = ch * (0.5 + np.around(psy))
    prx = cw * (0.5 + np.around(prx))
    pry = ch * (0.5 + np.around(pry))

    psx, psy = tuple(np.array(list(set(zip(list(psx), list(psy))))).T)
    prx, pry = tuple(np.array(list(set(zip(list(prx), list(pry))))).T)

    return psx, psy, prx, pry, pname
