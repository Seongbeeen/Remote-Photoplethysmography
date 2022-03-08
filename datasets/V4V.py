from cProfile import label
from fct import detrend
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from face_detect import face_rect
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
min_max_scaler = MinMaxScaler()


class V4V(Dataset):

    def __init__(self, video_paths, label_paths, depth, isTrain=True):
        if isTrain == True:
            overlap = 0
            hflip = True
            rand_shift = True
            rand_shift = False
        else:
            overlap = 0
            hflip = False
            rand_shift = False

        self.isTrain = isTrain
        self.hflip = hflip
        self.random_shift = rand_shift

        # Image config
        self.depth = depth
        self.height = 128
        self.width = 128
        self.channel = 3
        self.crop = True
        self.overlap = overlap
        # overlap, s.t., 0=< overlap < 1
        self.shift = int(self.depth*(1-overlap))

        # Gathtering each video's parameters
        self.video_fns = []
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

        for cnt, (video_path, label_path) in enumerate(zip(video_paths, label_paths)):
            metadata = skvideo.io.ffprobe(video_path)
            tsplit = metadata['video']['tag']['@value'].split(':')
            time_length = int(tsplit[0])*60*60 + \
                int(tsplit[1])*60+float(tsplit[2])
            frame_rate = float(eval(metadata['video']['@avg_frame_rate']))
            vlen = int(time_length*frame_rate)

            if vlen >= self.depth:
                # Load PPG signals
                ppg = np.array(np.loadtxt(label_path))

                self.ppg_time = np.arange(ppg.shape[0])/1000
                self.video_time = np.arange(vlen)/25

                interp_func = interpolate.interp1d(self.ppg_time, ppg)
                ppgi = interp_func(self.video_time)
                ppgi = nk.standardize(ppgi)[:vlen]

                if label_path.split("/")[-1][:4] == "F001":
                    ppgi = (-1)*(ppgi)

                min_max_scaler = MinMaxScaler()
                ppgi = min_max_scaler.fit_transform(ppgi.reshape(-1, 1))
                ppgi = min_max_scaler.fit_transform(detrend(ppgi))

                # For Debugging
                self.gt_paths.append(label_path)

                # depth보다 frame 수가 많거나 같은 경우만 fn 추가해줌
                self.ppgs.append(ppgi)
                self.video_fns.append(video_path)
                self.vlens.append(vlen)
                print(
                    'video cnt: {} / {}, filename: {}, video len: {}'.format(cnt, len(video_paths), video_path.split("/")[-1], vlen))

                vid = skvideo.io.vread(video_path)
                self.vids.append(vid)
                num_samples = math.floor(vlen/self.depth)
                self.num_samples += num_samples
                self.nums.append(self.num_samples-1)
            # else:
            #     print('video cnt: {} / {}, filename: {}, video len: {} - short sawtooth data'.format(
            #         cnt, len(video_paths), video_path.split("/")[-1], vlen))
            else:
                print('video cnt: {} / {}, filename: {}, video len: {} - short video length'.format(
                    cnt, len(video_paths), video_path.split("/")[-1], vlen))

        print("Initialization is Done!, total num_samples: ", self.num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # -------------------------------
        # Fill video with frames
        # -------------------------------
        # conv3d input: N x C x D x H X W
        sample_num = 0
        while idx > self.nums[sample_num]:
            sample_num += 1

        if idx > self.nums[sample_num-1]:
            idx = idx - self.nums[sample_num-1] - 1

        self.sample_num = sample_num  # for debugging

      #  Temporal jitter
        video = torch.empty(self.channel, self.depth,
                            self.height, self.width, dtype=torch.float)

        if self.random_shift:
            rand_offset = int(self.depth*(1-self.overlap)*0.5)
            rand_shift = random.randint(-rand_offset, rand_offset)
        else:
            rand_shift = 0
        if self.hflip:
            rand_flip = bool(random.getrandbits(1))
        else:
            rand_flip = False

        vlen = self.vlens[sample_num]

        if vlen >= self.depth:
            start_frame = idx * self.shift + rand_shift
            end_frame = idx * self.shift + self.depth + rand_shift
        else:
            start_frame = 0
            end_frame = vlen

        while start_frame < 0:
            start_frame += 1
            end_frame += 1

        # if self.isTrain:
            #             while end_frame > vlen:
            # HERE Why >= not > ??
        while end_frame > vlen or end_frame > len(self.ppg_sts[sample_num]):
            start_frame -= 1
            end_frame -= 1
        # else:
        #     if end_frame > vlen:
        #         end_frame = vlen

        fn = self.video_fns[sample_num]
        vid = self.vids[sample_num]
        x, y, w, h = face_rect(vid[start_frame: end_frame])
        #    print('idx: ', idx_orig, ', sam#: ', sample_num, fn, ', vlen: ', vlen, ', s/e: ', start_frame, end_frame)

        for cnt, img in enumerate(vid[start_frame: end_frame]):
            if self.crop:
                img = img[y:y + h, x: x + w, :]
            img = cv2.resize(img, (self.height, self.width),
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


class V4V_vreader(Dataset):
    """
        Dataset class for PhysNet neural network.
    """

    def __init__(self, video_paths, label_paths, depth, isTrain=True):
        if isTrain == True:
            overlap = 0
            hflip = True
    #             rand_shift = True
            rand_shift = False
        else:
            overlap = 0
            hflip = False
            rand_shift = False

        self.hflip = hflip
        self.random_shift = rand_shift

        # Image config
        self.depth = depth
        self.height = 128
        self.width = 128
        self.channel = 3
        self.crop = True
        self.overlap = overlap
        # overlap, s.t., 0=< overlap < 1
        self.shift = int(self.depth*(1-overlap))

        # Gathtering each video's parameters
        self.ppgs = []
        self.vlens = []
        self.nums = []
        self.num_samples = 0
        self.tname = []
        self.vids = []
        for cnt, (video_path, label_path) in enumerate(zip(video_paths, label_paths)):
            metadata = skvideo.io.ffprobe(video_path)
            tsplit = metadata['video']['tag']['@value'].split(':')
            time_length = int(tsplit[0])*60*60 + \
                int(tsplit[1])*60+float(tsplit[2])

            frame_rate = float(eval(metadata['video']['@avg_frame_rate']))
            start = 0
            vlen = int(time_length*frame_rate)
            # depth보다 frame 수가 적으면 불러오지 않음
            if vlen >= self.depth:
                # Load PPG signals
                ppg = np.array(np.loadtxt(label_path))

                self.ppg_time = np.arange(ppg.shape[0])/1000
                self.video_time = np.arange(vlen)/25

                interp_func = interpolate.interp1d(self.ppg_time, ppg)
                ppgi = interp_func(self.video_time)
                ppgi = nk.standardize(ppgi)[:vlen]

                ppgi = min_max_scaler.fit_transform(ppgi.reshape(-1, 1))
                ppgi = min_max_scaler.fit_transform(detrend(ppgi))

                self.vids.append(video_path)
                self.vlens.append(vlen)
                self.ppgs.append(ppgi)
                #  num_samples = math.floor((vlen - start - self.depth)/self.shift)+1 #??? -ppg_offset
                num_samples = math.floor(vlen/self.depth)

                self.num_samples += num_samples
                self.nums.append(self.num_samples-1)
            else:
                print('video cnt: {} / {}, filename: {}, video len: {} - short video length'.format(
                    cnt, len(video_paths), video_path.split("/")[-1], vlen))
        print("Initialization is Done!, total num_samples: ", self.num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # -------------------------------
        # Fill video with frames
        # -------------------------------
        # conv3d input: N x C x D x H X W
        sample_num = 0
        while idx > self.nums[sample_num]:

            sample_num += 1
        if idx > self.nums[sample_num-1]:
            idx = idx - self.nums[sample_num-1] - 1
        self.sample_num = sample_num  # for debugging

        video = torch.empty(self.channel, self.depth,
                            self.height, self.width, dtype=torch.float)

        if self.random_shift:
            rand_offset = int(self.depth*(1-self.overlap)*0.5)
            rand_shift = random.randint(-rand_offset, rand_offset)
        else:
            rand_shift = 0
        if self.hflip:
            rand_flip = bool(random.getrandbits(1))
        else:
            rand_flip = False

        start_frame = idx * self.shift + rand_shift
        end_frame = idx * self.shift + self.depth + rand_shift

        start = 0
        vlen = self.vlens[sample_num]
        while start+start_frame < start:
            start_frame += 1
            end_frame += 1
        while start + end_frame >= vlen or start + end_frame >= len(self.ppgs[sample_num]) + start:
            start_frame -= 1
            end_frame -= 1
        # for face detection
        vid = skvideo.io.vreader(self.vids[sample_num])
        x, y, w, h = face_rect(vid, vreader=True)
        # for dataset
        vid = skvideo.io.vreader(self.vids[sample_num])
        for cnt, frame in enumerate(vid):
            if cnt >= start_frame and cnt < end_frame:

                if self.crop:
                    frame = frame[y:y + h, x: x + w, :]
                img = cv2.resize(frame, (self.height, self.width),
                                 interpolation=cv2.INTER_CUBIC)
                img = ToTensor()(img)
                img = torch.sub(img, torch.mean(img, (1, 2)).view(3, 1, 1))

                if rand_flip:
                    img = torchvision.transforms.functional.hflip(img)
                # torch.Tensor(img).permute(2,0,1)
                video[:, cnt-start_frame, :, :] = img
            if cnt == end_frame-1:
                break

        target = self.ppgs[sample_num][start_frame:  end_frame]
        target = torch.tensor(target, dtype=torch.float)

        return video, target
