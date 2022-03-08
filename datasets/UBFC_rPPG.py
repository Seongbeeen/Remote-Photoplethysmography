from fileinput import filename
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from face_detect import face_rect
from scipy import interpolate
import numpy as np
import cv2
import torchvision
import torch
import random
import math
import neurokit2 as nk
import skvideo
import skvideo.io
import pandas as pd


class UBFC_rPPG(Dataset):
    """
        Dataset class for PhysNet neural network.
    """

    def __init__(self, video_paths, label_paths, depth, isTrain=False):
        if isTrain == True:
            overlap = 0
            hflip = True
            rand_shift = True
            self.isVidGen = True
        else:
            overlap = 0
            hflip = False
            rand_shift = False
            self.isVidGen = True

        self.isTrain = isTrain
        self.hflip = hflip
        self.random_shift = rand_shift

        # Image config
        self.depth = depth
        self.height = 128
        self.width = 128
        self.channel = 3
        self.crop = True
        # overlap = overlap
        # overlap, s.t., 0=< overlap < 1
        self.shift = int(self.depth*(1-overlap))

        # Gathtering each video's parameters
        self.video_fns = []
        self.ppgs = []
        self.vlens = []
        self.nums = []
        self.num_samples = 0
        # self.tname = []
        self.vids = []

        # self.ppg_sts = []
        # For debugging
        self.ppg_raw = []
        self.gt_paths = []
        self.hrs = []

        for cnt, (video_path, label_path) in enumerate(zip(video_paths, label_paths)):
            metadata = skvideo.io.ffprobe(video_path)
            # frame_rate = int(eval(metadata['video']['@avg_frame_rate']))
            vlen = int(metadata['video']["@nb_frames"])
            if vlen >= self.depth:
                # Load PPG signals
                ppg = np.loadtxt(label_path)[0]
                self.ppgs.append(ppg)
                self.video_fns.append(video_path)
                filename = video_path.split("\\")[-1]
                print(
                    'video cnt: {} / {}, filename: {}, video len: {}'.format(cnt+1, len(video_paths), filename, vlen))

                self.vlens.append(vlen)
                vid = skvideo.io.vread(video_path)
                self.vids.append(vid[:vlen])

                if isTrain:
                    # Not considering overlap
                    num_samples = math.floor(vlen/self.depth)
                else:
                    # Not considering overlap
                    num_samples = math.ceil(vlen/self.depth)

                self.num_samples += num_samples
                self.nums.append(self.num_samples-1)
            else:
                print('video cnt: {} / {}, filename: {}, video len: {} - short video length'.format(
                    cnt+1, len(video_paths), video_path[14:], vlen))

        print("Initialization is Done!, total num_samples: ", self.num_samples)

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

        # Temporal jitter
        video = torch.empty(channel, depth, height, width, dtype=torch.float)

        if self.random_shift:
            rand_offset = int(depth*0.5)
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

        # if self.isTrain:
    #             while end_frame > vlen:
        # HERE Why >= not > ??
        while end_frame >= vlen or end_frame >= len(self.ppgs[sample_num]):
            start_frame -= 1
            end_frame -= 1
        # else:
            # if end_frame > vlen:
            # end_frame = vlen

        fn = self.video_fns[sample_num]

        # print('idx: ', idx_orig, ', sam#: ', sample_num, fn, ', vlen: ', vlen, ', s/e: ', start_frame, end_frame)

        vid = self.vids[sample_num]
        x, y, w, h = face_rect(
            self.vids[sample_num][start_frame: end_frame])
        for cnt, img in enumerate(vid[start_frame: end_frame]):
            if self.crop:
                img = img[y:y + h, x: x + w, :]
            img = cv2.resize(img, (height, width),
                             interpolation=cv2.INTER_CUBIC)
            img = ToTensor()(img)
            #   img = torch.sub(img, torch.mean(img, (1, 2)).view(3, 1, 1))
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


class UBFC_rPPG_vreader(Dataset):
    """
        Dataset class for PhysNet neural network.
    """

    def __init__(self, video_paths, label_paths, depth, isTrain=False):
        if isTrain == True:
            overlap = 0
            hflip = True
            rand_shift = True
            self.isVidGen = True
        else:
            overlap = 0
            hflip = False
            rand_shift = False
            self.isVidGen = True

        self.isTrain = isTrain
        self.hflip = hflip
        self.random_shift = rand_shift

        # Image config
        self.depth = depth
        self.height = 128
        self.width = 128
        self.channel = 3
        self.crop = True
        # overlap = overlap
        # overlap, s.t., 0=< overlap < 1
        self.shift = int(self.depth*(1-overlap))

        # Gathtering each video's parameters
        self.video_fns = []
        self.ppgs = []
        self.vlens = []
        self.nums = []
        self.num_samples = 0
        # self.tname = []
        self.vids = []

        # self.ppg_sts = []
        # For debugging
        self.ppg_raw = []
        self.gt_paths = []
        self.hrs = []

        for cnt, (video_path, label_path) in enumerate(zip(video_paths, label_paths)):
            metadata = skvideo.io.ffprobe(video_path)
            # frame_rate = int(eval(metadata['video']['@avg_frame_rate']))
            vlen = int(metadata['video']["@nb_frames"])
            if vlen >= self.depth:
                # Load PPG signals
                ppg = np.loadtxt(label_path)[0]
                self.ppgs.append(ppg)
                self.video_fns.append(video_path)
                filename = video_path.split("\\")[-1]
                print(
                    'video cnt: {} / {}, filename: {}, video len: {}'.format(cnt+1, len(video_paths), filename, vlen))

                self.vlens.append(vlen)
                self.vids.append(video_path)

                if isTrain:
                    # Not considering overlap
                    num_samples = math.floor(vlen/self.depth)
                else:
                    # Not considering overlap
                    num_samples = math.ceil(vlen/self.depth)

                self.num_samples += num_samples
                self.nums.append(self.num_samples-1)
            else:
                print('video cnt: {} / {}, filename: {}, video len: {} - short video length'.format(
                    cnt+1, len(video_paths), video_path[14:], vlen))

        print("Initialization is Done!, total num_samples: ", self.num_samples)

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

        # Temporal jitter
        video = torch.empty(channel, depth, height, width, dtype=torch.float)

        if self.random_shift:
            rand_offset = int(depth*0.5)
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

        # if self.isTrain:
    #             while end_frame > vlen:
        # HERE Why >= not > ??
        while end_frame >= vlen or end_frame >= len(self.ppgs[sample_num]):
            start_frame -= 1
            end_frame -= 1
        # else:
            # if end_frame > vlen:
            # end_frame = vlen

        fn = self.video_fns[sample_num]

        # print('idx: ', idx_orig, ', sam#: ', sample_num, fn, ', vlen: ', vlen, ', s/e: ', start_frame, end_frame)

        # for face detection
        vid = skvideo.io.vreader(self.vids[sample_num])
        x, y, w, h = face_rect(vid, vreader=True)
        # for dataset
        vid = skvideo.io.vreader(self.vids[sample_num])
        for cnt, frame in enumerate(vid):
            if cnt >= start_frame and cnt < end_frame:

                if self.crop:
                    img = frame[y:y + h, x: x + w, :]
                img = cv2.resize(img, (height, width),
                                 interpolation=cv2.INTER_CUBIC)
                img = ToTensor()(img)
                # img = torch.sub(img, torch.mean(img, (1, 2)).view(3, 1, 1))

                if rand_flip:
                    img = torchvision.transforms.functional.hflip(img)
                # torch.Tensor(img).permute(2,0,1)
                video[:, cnt-start_frame, :, :] = img
            if cnt == end_frame-1:
                break

        target = self.ppgs[sample_num][start_frame:  end_frame]
        target = torch.tensor(target, dtype=torch.float)
        # For Debugging
        self.ppgi = self.ppgs[sample_num][start_frame:  end_frame]
        self.start_frame = start_frame
        self.end_frame = end_frame

        return video, target
