import numpy as np
import scipy
from scipy import interpolate
import neurokit2 as nk


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
