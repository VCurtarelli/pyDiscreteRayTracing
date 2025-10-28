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


