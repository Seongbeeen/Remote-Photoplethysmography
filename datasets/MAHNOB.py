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
import os
import skvideo
import skvideo.io
# ECG 데이터 바로 불러올 때?


class MAHNOB(Dataset):
    """
         Dataset class for PhysNet neural network.
    """

    def __init__(self, video_path, video_fns, time_depth, isTrain=False, normalize=False):
        if isTrain == True:
            self.overlap = 0
            self.hflip = True
            self.rand_shift = True
            self.isVidGen = True
        else:
            self.overlap = 0
            self.hflip = False
            self.rand_shift = False
            self.isVidGen = True

        self.vidStore = True
        self.isTrain = isTrain

        # Image config
        self.channel = 3
        self.depth = time_depth
        self.height = 128
        self.width = 128
        # overlap, s.t., 0=< overlap < 1
        self.shift = int(self.depth*(1-self.overlap))

        # Gathtering each video's parameters
        self.video_fns = []
        self.video_path = video_path
        self.ppgs = []
        self.vlens = []
        self.nums = []
        self.num_samples = 0
        self.tname = []
        self.vids = []
        self.normalize = normalize

        # self.ppg_sts = []
        # For debugging
        self.ppg_raw = []
        self.gt_paths = []
        self.hrs = []
        #################################################################################
        for cnt, fn in enumerate(video_fns):
            video_path = os.path.join(self.video_path, fn)
            print(video_path)
            metadata = skvideo.io.ffprobe(video_path)
            folder_name = fn.split("/")[0]
            tmp = metadata['video']['@avg_frame_rate'].split("/")
            frame_rate = float(int(tmp[0])/int(tmp[1]))
            vlen = int(metadata['video']["@nb_frames"])

            if vlen >= self.depth:
                # Load PPG signals
                tname = "folder_name"+"/"+"folder_name"+".txt"
                gt_path = os.path.join(
                    self.video_path, folder_name, folder_name)+".txt"
                ppg = np.array(np.loadtxt(gt_path))
                sampling_rate = 256
                self.ppg_time = np.arange(ppg.shape[0])/sampling_rate
                self.video_time = np.arange(vlen)/frame_rate

                sig_time = int(len(self.ppg_time)/sampling_rate)
                vid_time = int(len(self.video_time)/frame_rate)

                # for synchronization
                if sig_time <= vid_time:
                    vid_time = sig_time
                else:
                    sig_time = vid_time

                self.ppg_time = self.ppg_time[:sig_time*sampling_rate]
                self.video_time = self.video_time[:vid_time*61]
                ppg = ppg[:sig_time*sampling_rate]
                vlen = len(self.video_time)

                interp_func = interpolate.interp1d(self.ppg_time, ppg)
                ppgi = interp_func(self.video_time)
                self.ppgs.append(ppgi)

                self.video_fns.append(fn)
                print(
                    'video cnt: {} / {}, filename: {}, video len: {}'.format(cnt+1, len(video_fns), fn, vlen))

                self.vlens.append(vlen)
                vid = skvideo.io.vread(video_path)
                self.vids.append(vid[:vlen])

                if isTrain:
                    num_samples = math.floor(vlen/self.depth)
                else:
                    num_samples = math.ceil(vlen/self.depth)

                self.num_samples += num_samples
                self.nums.append(self.num_samples-1)
                self.tname.append(tname[:-4])
            else:
                print('video cnt: {} / {}, filename: {}, video len: {} - short video length'.format(
                    cnt+1, len(video_fns), fn, vlen))

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

        # Random shift
        if self.random_shift:
            rand_offset = int(self.depth*0.5)
            rand_shift = random.randint(-rand_offset, rand_offset)
        else:
            rand_shift = 0
        # Random horiziontal flip
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
        while end_frame >= vlen or end_frame >= len(self.ppgs[sample_num]):
            start_frame -= 1
            end_frame -= 1
        # else:
            # if end_frame > vlen:
            # end_frame = vlen

        # fn = self.video_fns[sample_num]

        # print('idx: ', idx_orig, ', sam#: ', sample_num, fn, ', vlen: ', vlen, ', s/e: ', start_frame, end_frame)

        vid = self.vids[sample_num][start_frame: end_frame]
        x, y, w, h = face_rect(vid)

        for cnt, img in enumerate(vid):
            if self.crop:
                img = img[y:y + h, x: x + w, :]
            img = cv2.resize(img, (self.height, self.width),
                             interpolation=cv2.INTER_CUBIC)
            img = ToTensor()(img)
            if rand_flip:
                img = torchvision.transforms.functional.hflip(img)
            if self.normalize:
                img = torch.sub(img, torch.mean(img, (1, 2)).view(3, 1, 1))
            video[:, cnt, :, :] = img  # torch.Tensor(img).permute(2,0,1)

        target = torch.tensor(
            self.ppgs[sample_num][start_frame:  end_frame], dtype=torch.float)

        return video, target
