import numpy as np
import scipy
from scipy import interpolate
import neurokit2 as nk


def interpolation_ppg(imgt, time_ns, ppg, normalize=True):
    itp_ppgs = list()
    ppg_idx1 = 0
    ppg_idx2 = 1
    ppg_idx3 = 2
    err = 0
    for img_idx in range(1, len(imgt)-1):
        diff1 = 0
        diff2 = 0
        if time_ns[ppg_idx3] - imgt[img_idx] > imgt[img_idx] - time_ns[ppg_idx1]:
            while imgt[img_idx] - time_ns[ppg_idx1] < 0:
                if ppg_idx1 == 0:
                    break
                ppg_idx1 -= 1
                ppg_idx2 -= 1
                ppg_idx3 -= 1
            diff1 = (imgt[img_idx] - time_ns[ppg_idx1]) / \
                (time_ns[ppg_idx2] - time_ns[ppg_idx1])
            diff2 = (time_ns[ppg_idx2] - imgt[img_idx]) / \
                (time_ns[ppg_idx2] - time_ns[ppg_idx1])
            new_ppg = ppg[ppg_idx2] * diff1 + ppg[ppg_idx1] * diff2
            itp_ppgs.append(new_ppg)
        else:
            if time_ns[ppg_idx3] - imgt[img_idx] < imgt[img_idx] - time_ns[ppg_idx2]:
                ppg_idx1 += 1
                ppg_idx2 += 1
                ppg_idx3 += 1
            diff1 = (time_ns[ppg_idx3] - imgt[img_idx]) / \
                (time_ns[ppg_idx3] - time_ns[ppg_idx2])
            diff2 = (imgt[img_idx] - time_ns[ppg_idx2]) / \
                (time_ns[ppg_idx3] - time_ns[ppg_idx2])
            if diff1 < 0 or diff2 < 0:
                diff1 = (imgt[img_idx] - time_ns[ppg_idx1]) / \
                    (time_ns[ppg_idx2] - time_ns[ppg_idx1])
                diff2 = (time_ns[ppg_idx2] - imgt[img_idx]) / \
                    (time_ns[ppg_idx2] - time_ns[ppg_idx1])
                new_ppg = ppg[ppg_idx2] * diff1 + ppg[ppg_idx1] * diff2
            else:
                new_ppg = ppg[ppg_idx2] * diff1 + ppg[ppg_idx3] * diff2
            itp_ppgs.append(new_ppg)
        if ppg_idx2 + 2 < len(time_ns):
            time_delay = time_ns[ppg_idx2+2]-time_ns[ppg_idx2] - 15 * err
    #         print(err, time_delay)
            if time_delay < 40:
                step = 2
                err = 0
            elif time_delay < 65:
                step = 1
                err += 1
            else:
                err += 1
            ppg_idx1 += step
            ppg_idx2 += step
            ppg_idx3 += step
        if ppg_idx3 >= len(time_ns):
            ppg_idx3 -= 1
            break
    if abs(imgt[len(imgt)-1] - time_ns[-1]) > abs(imgt[len(imgt)-1] - time_ns[ppg_idx3]):
        end_idx = ppg_idx3
    elif abs(imgt[len(imgt)-1] - time_ns[-1]) > abs(imgt[len(imgt)-1] - time_ns[ppg_idx2]):
        end_idx = ppg_idx2
    else:
        end_idx = -1
    itp_ppgs.append(ppg[end_idx])

    if normalize:
        itp_ppgs = nk.standardize(itp_ppgs)
    return itp_ppgs


def detrend(X, detLambda=10):
    t = X.shape[0]
    l = t/detLambda  # lambda
    I = np.identity(t)
    # this works better than spdiags in python
    D2 = scipy.sparse.diags([1, -2, 1], [0, 1, 2], shape=(t-2, t)).toarray()
    detrendedX = (I-np.linalg.inv(I+l**2*(np.transpose(D2).dot(D2)))).dot(X)
    return detrendedX


def sawtooth(ppg, sampling_rate, video_time, ppg_time):

    signal, _ = nk.ppg_process(ppg, sampling_rate=sampling_rate)
    peak = signal['PPG_Peaks'] == True

    video_time = np.arange(ppg.shape[0])/1000
    video_time = video_time[::40]
    vt_ind = np.arange(video_time.shape[0])
    interp_func = interpolate.interp1d(video_time, vt_ind)

    # ppg time is always short than video time?
    vt_indi = interp_func(ppg_time[peak])
    vt_pk_indi = np.round(vt_indi).astype(int)
    vt_pk = video_time[vt_pk_indi]  # peak times in video time
    # valley times in video time
    vt_vy = vt_pk[:-1] + (vt_pk[1:] - vt_pk[:-1])*0.66

    pks = [1]*vt_pk.shape[0]
    vys = [0]*vt_vy.shape[0]

    vt_pv = np.stack((vt_pk[:-1], vt_vy), axis=1).reshape(-1)
    # peak and valley times in video time
    vt_pv = np.append(vt_pv, vt_pk[-1])
    # peak and valley times in video time
    vt_pv = np.append(self.video_time[0], vt_pv)
    # peak and valley times in video time
    vt_pv = np.append(vt_pv, self.video_time[-1])

    v_pv = np.stack((pks[:-1], vys), axis=1).reshape(-1)
    # peak and valley values in video time
    v_pv = np.append(v_pv, pks[-1])
    v_pv = np.append(vys[0], v_pv)
    # peak and valley values in video time
    v_pv = np.append(v_pv, vys[0])

    # ideal peak anv valley function
    interp_func = interpolate.interp1d(vt_pv, v_pv)
    # [:vlen] # sawtooth ppg at video time cut
    ppg_st = interp_func(self.video_time)

    return ppg_st
