from fct import detrend
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from scipy import interpolate
import numpy as np
import cv2
import torchvision
import torch
import random
import math
import neurokit2 as nk
import os
import skvideo
import skvideo.io


class V4V(Dataset):

    def __init__(self, video_path, video_fns, time_depth, isTrain=True):
        if isTrain == True:
            overlap = 0
            hflip = True
            rand_shift = True
            rand_shift = False
        else:
            overlap = 0
            hflip = False
            rand_shift = False

        self.vidAll = True
        self.isTrain = isTrain
        self.hflip = hflip
        self.random_shift = rand_shift

        # Image config
        self.depth = time_depth
        self.height = 128
        self.width = 128
        self.channel = 3
        self.overlap = overlap
        # overlap, s.t., 0=< overlap < 1
        self.shift = int(self.depth*(1-overlap))

        # Gathtering each video's parameters
        self.video_fns = []
        self.video_path = video_path
        self.ppgs = []
        self.vlens = []
        self.nums = []
        self.num_samples = 0
        self.tname = []
        self.vids = []

        self.ppg_sts = []
        # For debugging
        self.ppg_raw = []
        self.gt_paths = []
        self.hrs = []

        for cnt, fn in enumerate(video_fns):
            video_path = os.path.join(self.video_path, fn)
            metadata = skvideo.io.ffprobe(video_path)
            tsplit = metadata['video']['tag']['@value'].split(':')
            time_length = int(tsplit[0])*60*60 + \
                int(tsplit[1])*60+float(tsplit[2])
            frame_rate = float(
                metadata['video']['@avg_frame_rate'].split('/')[0])

            base_dir = os.path.split(os.path.split(
                os.path.dirname(video_path))[0])[0]
            vlen = int(time_length*frame_rate)

         # print(cnt, fn, vlen)
         #  depth보다 frame 수가 적으면 불러오지 않음
            if vlen >= self.depth:
                # Load PPG signals
                splt = fn[:-4].split("_")
                tname = splt[0]+"-"+splt[1]+"-"+"BP.txt"
                gt_path = os.path.join(
                    base_dir, 'Ground truth', 'BP_raw_1KHz', tname)
                ppg = np.array(np.loadtxt(gt_path))

                self.ppg_time = np.arange(ppg.shape[0])/1000
                self.video_time = np.arange(vlen)/25

                interp_func = interpolate.interp1d(self.ppg_time, ppg)
                ppgi = interp_func(self.video_time)
                ppgi = nk.standardize(ppgi)[:vlen]

            if fn[:4] == "F001":
                ppgi = (-1)*(ppgi)

            min_max_scaler = MinMaxScaler()
            ppgi = min_max_scaler.fit_transform(ppgi.reshape(-1, 1))
            ppgi = min_max_scaler.fit_transform(detrend(ppgi))

            # Find Peaks
            signal, info = nk.ppg_process(ppg, sampling_rate=1000)
            peak = signal['PPG_Peaks'] == True

            video_time = np.arange(ppg.shape[0])/1000
            video_time = video_time[::40]
            vt_ind = np.arange(video_time.shape[0])
            interp_func = interpolate.interp1d(video_time, vt_ind)

            # ppg time is always short than video time?
            vt_indi = interp_func(self.ppg_time[peak])
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

            # Store data only if sawtooth ppg length == depth
            if ppg_st.shape[0] >= self.depth:
                self.ppgs.append(ppgi)
                self.ppg_sts.append(ppg_st)
                # For Debugging
                self.ppg_raw.append(ppg)
                self.gt_paths.append(gt_path)

                # depth보다 frame 수가 많거나 같은 경우만 fn 추가해줌
                self.video_fns.append(fn)
                self.vlens.append(vlen)
                print(
                    'video cnt: {} / {}, filename: {}, video len: {}'.format(cnt, len(video_fns), fn, vlen))

                if self.vidAll:
                    vid = skvideo.io.vread(video_path)
                    self.vids.append(vid)
                num_samples = math.floor(vlen/self.depth)
                self.num_samples += num_samples
                self.nums.append(self.num_samples-1)
                self.tname.append(tname[:-4])
            else:
                print('video cnt: {} / {}, filename: {}, video len: {} - short sawtooth data'.format(
                    cnt, len(video_fns), fn, vlen))
        else:
            print('video cnt: {} / {}, filename: {}, video len: {} - short video length'.format(
                cnt, len(video_fns), fn, vlen))

        self.crop = True
        dpath = "config/haarcascade_frontalface_alt.xml"
        if not os.path.exists(dpath):
            print("Cascade file not present!")
        self.face_cascade = cv2.CascadeClassifier(dpath)
        print("Initialization is Done!, total num_samples: ", self.num_samples)

        # for vid store
        self.old_sample_num = -1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # -------------------------------
        # Fill video with frames
        # -------------------------------
        # conv3d input: N x C x D x H X W
        idx_orig = idx
        sample_num = 0
        while idx > self.nums[sample_num]:
            sample_num += 1

        if idx > self.nums[sample_num-1]:
            idx = idx - self.nums[sample_num-1] - 1

        self.sample_num = sample_num  # for debugging

        shift = self.shift
        depth = self.depth
        height = self.height
        width = self.width
        channel = self.channel

      #  Temporal jitter
        video = torch.empty(channel, depth, height, width, dtype=torch.float)

        if self.random_shift:
            rand_offset = int(depth*(1-self.overlap)*0.5)
            rand_shift = random.randint(-rand_offset, rand_offset)
        else:
            rand_shift = 0
        if self.hflip:
            rand_flip = bool(random.getrandbits(1))
        else:
            rand_flip = False

        vlen = self.vlens[sample_num]

        if vlen >= self.depth:
            start_frame = idx * shift + rand_shift
            end_frame = idx * shift + depth + rand_shift
        else:
            start_frame = 0
            end_frame = vlen

        while start_frame < 0:
            start_frame += 1
            end_frame += 1

        if self.isTrain:
            #             while end_frame > vlen:
            # HERE Why >= not > ??
            while end_frame > vlen or end_frame > len(self.ppg_sts[sample_num]):
                start_frame -= 1
                end_frame -= 1
        else:
            if end_frame > vlen:
                end_frame = vlen

        fn = self.video_fns[sample_num]

        #    print('idx: ', idx_orig, ', sam#: ', sample_num, fn, ', vlen: ', vlen, ', s/e: ', start_frame, end_frame)

        if self.vidAll:
            vid = self.vids[sample_num]
        else:
            if self.old_sample_num == self.sample_num:
                vid = self.vid_store
            else:
                video_path = os.path.join(self.video_path, fn)
                vid = skvideo.io.vread(video_path)
                self.vid_store = vid

        self.old_sample_num = self.sample_num

        img = vid[start_frame]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected = list(self.face_cascade.detectMultiScale(gray, 1.1, 4))

        i = 1
        while len(detected) <= 0:
            img = vid[start_frame+i]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detected = list(
                self.face_cascade.detectMultiScale(gray, 1.1, 4))
            i += 1
        detected.sort(key=lambda a: a[-1] * a[-2])
        face_rect = detected[-1]
        x, y, w, h = face_rect

        for cnt, img in enumerate(vid[start_frame: end_frame]):
            if self.crop:
                img = img[y:y + h, x: x + w, :]
            img = cv2.resize(img, (height, width),
                             interpolation=cv2.INTER_CUBIC)
            img = ToTensor()(img)
            img = torch.sub(img, torch.mean(img, (1, 2)).view(3, 1, 1))
            if rand_flip:
                img = torchvision.transforms.functional.hflip(img)

            video[:, cnt, :, :] = img  # torch.Tensor(img).permute(2,0,1)

        target = self.ppgs[sample_num][start_frame:  end_frame]
        target = torch.tensor(target, dtype=torch.float)

        # For Debugging
        self.ppgi = self.ppgs[sample_num][start_frame:  end_frame]
        self.start_frame = start_frame
        self.end_frame = end_frame

        return video, target
