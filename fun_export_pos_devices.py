def export_pos_devices(pos_sources, pos_receivers, direc, width, height):
    header = 'x_pos,y_pos'
    pos_sources = [(ps[0], height-ps[1]) for ps in pos_sources]
    pos_receivers = [(pr[0], height-pr[1]) for pr in pos_receivers]
    txt_sources = '\n'.join([header] + ['{},{}'.format(*pos) for pos in pos_sources])
    txt_receivers = '\n'.join([header] + ['{},{}'.format(*pos) for pos in pos_receivers])

    with open(direc + '/pos_sources.csv', 'w') as f:
        f.write(txt_sources)
        f.close()
    with open(direc + '/pos_receivers.csv', 'w') as f:
        f.write(txt_receivers)
        f.close()
    pass
