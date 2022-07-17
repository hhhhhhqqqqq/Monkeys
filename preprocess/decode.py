import numpy as np
import h5py


# decode chan_names
def decode_chan_names(chan_names, matdata):
    a = []
    for i in range (192):
        ref = chan_names[0][i]
        res = list(np.array(matdata[ref]).reshape(6))
        for i in range (6):
            res[i] = chr(res[i])
        a.append((res[0]+res[1]+res[2]+res[3]+res[4]+res[5]))
    return a


# decode cursor_pos
def decode_cursor_pos(cursor_pos):
    return (np.array(cursor_pos).transpose())


# decode finger_pos
def decode_finger_pos(finger_pos):
    return (np.array(finger_pos).transpose())


# decode spikes
def decode_spikes(spikes, matdata):
    x = []
    for i in range (192):
        x.append([])
    for i in range(192):
        for m in range(5):
            ref = spikes[m][i]
            res = np.array(matdata[ref]).squeeze()
            if(res.any(0)==0):
                x[i].append(np.array([]))
            else:
                x[i].append(res)

    return x


# decode t
def decode_t(t):
    return np.array(t).transpose().squeeze()


# deocde target_pos
def decode_target_pose(target_pos):
    return np.array(target_pos).transpose()


# decode wf
def decode_wf(wf, matdata):
    x = []
    for i in range (192):
        x.append([])
    for i in range(192):
        for m in range(5):
            ref = wf[m][i]
            res = np.array(matdata[ref])
            x[i].append(res)
    return x


def decode_all(path):
    matdata = h5py.File(path)
    # print('matdata key is: ', matdata.keys())
    chan_names = matdata['chan_names']
    cursor_pos = matdata['cursor_pos']
    finger_pos = matdata['finger_pos']
    spikes = matdata['spikes']
    t = matdata['t']
    target_pos = matdata['target_pos']
    wf = matdata['wf']
    return decode_chan_names(chan_names, matdata), decode_cursor_pos(cursor_pos), \
           decode_finger_pos(finger_pos), decode_spikes(spikes, matdata), \
           decode_t(t), decode_target_pose(target_pos), decode_wf(wf, matdata)


if __name__ == '__main__':
    path = 'loco_20170215_02.mat'
    chan_names, _, finger_pos, spikes, t, _, _ = decode_all(path)
    print(chan_names)
    print(finger_pos)
    print(spikes[0][0].shape)
    print(t)

