import os

import numpy as np

def encode(number, symbols_list):
    symbols = [a for a in symbols_list]
    base = len(symbols)
    rep = []
    sign = False
    if number < 0:
        sign = True
        number = -number
    # for i in range(int(np.ceil(len(number_bin)/6))):
    #     digit = int(number_bin[6*i:6*(i+1)], 2)
    #     print(digit)
    while True:
        digit = number % base
        rep.insert(0, symbols[digit])
        number //= base
        # print(number)
        if number == 0:
            break
    rep = ('-' if sign else '') + ''.join(rep)
    # print(rep)
    return rep

def encode64(number, symbols_list='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ=*'):
    return 'ts-' + encode(number, symbols_list)

def encode16(number, symbols_list='0123456789abcdef'):
    return 'h' + encode(number, symbols_list)


def mhash(arg):
    b = int(sum([int(str(ord(_c))) for _a in arg for _c in str(_a)]))
    return b


def export_params(parameters: dict, direc, name, filename):
    parameters_text = '\n'.join([key + ': ' + str(parameters[key]) for key in parameters.keys()])
    with open(direc+name+'/'+filename+'.txt', 'w') as f:
        f.write(parameters_text)


def make_folders_and_hash(n_rays: int, parameters: dict, name=None) -> tuple[str, str, dict]:
    hash_val = mhash(parameters.values())
    code = encode16(hash_val)
    direc = 'Results/'
    if name is None:
        name = code
    os.makedirs(direc, exist_ok=True)
    os.makedirs(direc + name, exist_ok=True)

    parameters['code'] = code
    parameters['n_rays'] = n_rays
    # export_params(parameters, direc, name, 'simulation parameters')
    return code, direc, parameters
