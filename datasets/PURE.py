from skimage import io
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from face_detect import face_rect
from fct import interpolation_ppg
from glob import glob
from sklearn.preprocessing import MinMaxScaler
import json
import numpy as np
import cv2
import torchvision
import torch
import random
import math
import neurokit2 as nk


class PURE(Dataset):
    def __init__(self, video_path, video_fns, time_depth, isTrain=True, ppg_offset=6, overlap=0, normalize=False):
        if isTrain == True:
            self.overlap = overlap
            self.hflip = True
            self.rand_shift = True

        else:
            self.overlap = overlap
            self.hflip = False
            self.rand_shift = False

        # overlap, s.t., 0=< overlap < 1
        self.height = 128
        self.width = 128
        self.channel = 3
        self.time_depth = time_depth

        self.shift = int(self.time_depth*(1-self.overlap))
        self.crop = True
        self.ppgs = []
        self.vfls = []
        self.nums = []
        self.num_samples = 0
        self.vids = []
        self.vlens = []
        self.normalize = normalize
        self.vdirs = video_fns

        for video_dir in video_fns:
            # Load video image list
            # get path of image file
            image_list = sorted(glob(video_path + video_dir + "/*"))
            start = 0
            end = len(image_list)

            self.vfls.append(image_list)
            vids = []
            for img_path in image_list:
                image = io.imread(img_path)
                vids.append(image)
            vlen = len(vids)
            if vlen >= self.time_depth:
                self.vids.append(vids)
                self.vlens.append(len(vids))

                print(f"Add {end} images")

                # Load PPG signals
                with open(video_path + video_dir + ".json", "r") as st_json:
                    data_json = json.load(st_json)
                time_ns = list()
                ppg = list()
                for dat in data_json['/FullPackage']:
                    time_ns.append(dat['Timestamp'])
                    ppg.append(dat['Value']['waveform'])
                time_ns = np.array(time_ns)
                ppg = np.array(ppg)
                imgt = list()

                for fname in image_list:
                    imgt.append(int(fname.split("\\")[-1][5:-4]))
                imgt = np.array(imgt)
                imgt2 = (imgt - imgt[0])/1e6
                time_ns2 = (time_ns - imgt[0])/1e6
                ppg = interpolation_ppg(imgt2, time_ns2, ppg, normalize=True)

                scaler = MinMaxScaler(feature_range=(-1, 1))
                scaler.fit(torch.Tensor(ppg).reshape(-1, 1))
                # ??? tensor??? ????????? ??? ??? ?????? ???????
                ppg = scaler.transform(torch.Tensor(
                    ppg).reshape(-1, 1)).squeeze()

                ppg = np.array(ppg)[start + ppg_offset:end]
                self.ppgs.append(ppg)

                num_samples = math.ceil(
                    (end - start - self.time_depth)/self.shift)+1

                self.num_samples += num_samples
                self.nums.append(self.num_samples-1)
        print("Initialization is Done!, total num_samples: ", self.num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # conv3d input: N x C x D x H X W
        sample_num = 0
        while idx > self.nums[sample_num]:
            sample_num += 1
        if idx > self.nums[sample_num-1]:
            idx = idx - self.nums[sample_num-1] - 1

        video = torch.empty(self.channel, self.time_depth,
                            self.height, self.width, dtype=torch.float)
        # Random shift
        if self.rand_shift:
            rand_offset = int(self.time_depth*(1-self.overlap)*0.5)
            rand_shift = random.randint(-rand_offset, rand_offset)
        else:
            rand_shift = 0
        # Random horiziontal flip
        if self.hflip:
            rand_flip = bool(random.getrandbits(1))
        else:
            rand_flip = False

        start_frame = idx * self.shift + rand_shift
        end_frame = idx * self.shift + self.time_depth + rand_shift
        # print(f"idx: {idx}, start_frame: {start_frame}, end_frame: {end_frame}")

        # range ???????????? rand_shift ??? ??????.
        start = 0
        end = len(self.vfls[sample_num])
        while start + start_frame < start:
            start_frame += 1
            end_frame += 1
        while start + end_frame >= end or start + end_frame >= len(self.ppgs[sample_num]) + start:

            start_frame -= 1
            end_frame -= 1

        x, y, w, h = face_rect(self.vids[sample_num][start_frame: end_frame])
        for cnt, img in enumerate(self.vids[sample_num][start_frame: end_frame]):
            if self.crop:
                img = img[y:y + h, x: x + w, :]
            img = cv2.resize(img, (self.height, self.width),
                             interpolation=cv2.INTER_CUBIC)
            img = ToTensor()(img)
            if rand_flip:
                img = torchvision.transforms.functional.hflip(img)

            if self.normalize:
                # spatial intensity norm for each channel
                img = torch.sub(img, torch.mean(img, (1, 2)).view(3, 1, 1))
            video[:, cnt, :, :] = img  # video -> C, D, H, W

        target = torch.tensor(
            self.ppgs[sample_num][start_frame:  end_frame], dtype=torch.float)
        return video, target


class PURE_vreader(Dataset):
    def __init__(self, video_path, video_fns, time_depth, isTrain=True, ppg_offset=6, overlap=0, normalize=False):
        if isTrain == True:
            self.overlap = overlap
            self.hflip = True
            self.rand_shift = True

        else:
            self.overlap = overlap
            self.hflip = False
            self.rand_shift = False

        # overlap, s.t., 0=< overlap < 1
        self.height = 128
        self.width = 128
        self.channel = 3
        self.time_depth = time_depth

        self.shift = int(self.time_depth*(1-self.overlap))
        self.crop = True
        self.ppgs = []
        self.vfls = []
        self.nums = []
        self.num_samples = 0
        self.vids = []
        self.vlens = []
        self.normalize = normalize
        self.vdirs = video_fns

        for video_dir in video_fns:
            # Load video image list
            # get path of image file
            image_list = sorted(glob(video_path + video_dir + "/*"))
            start = 0
            end = len(image_list)

            self.vfls.append(image_list)
            vids = []
            for img_path in image_list:
                image = io.imread(img_path)
                vids.append(image)
            vlen = len(vids)
            if vlen >= self.time_depth:
                self.vids.append(vids)
                self.vlens.append(len(vids))

                print(f"Add {end} images")

                # Load PPG signals
                with open(video_path + video_dir + ".json", "r") as st_json:
                    data_json = json.load(st_json)
                time_ns = list()
                ppg = list()
                for dat in data_json['/FullPackage']:
                    time_ns.append(dat['Timestamp'])
                    ppg.append(dat['Value']['waveform'])
                time_ns = np.array(time_ns)
                ppg = np.array(ppg)
                imgt = list()

                for fname in image_list:
                    imgt.append(int(fname.split("\\")[-1][5:-4]))
                imgt = np.array(imgt)
                imgt2 = (imgt - imgt[0])/1e6
                time_ns2 = (time_ns - imgt[0])/1e6
                ppg = interpolation_ppg(imgt2, time_ns2, ppg, normalize=True)

                scaler = MinMaxScaler(feature_range=(-1, 1))
                scaler.fit(torch.Tensor(ppg).reshape(-1, 1))
                # ??? tensor??? ????????? ??? ??? ?????? ???????
                ppg = scaler.transform(torch.Tensor(
                    ppg).reshape(-1, 1)).squeeze()

                ppg = np.array(ppg)[start + ppg_offset:end]
                self.ppgs.append(ppg)

                num_samples = math.ceil(
                    (end - start - self.time_depth)/self.shift)+1

                self.num_samples += num_samples
                self.nums.append(self.num_samples-1)
        print("Initialization is Done!, total num_samples: ", self.num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # conv3d input: N x C x D x H X W
        sample_num = 0
        while idx > self.nums[sample_num]:
            sample_num += 1
        if idx > self.nums[sample_num-1]:
            idx = idx - self.nums[sample_num-1] - 1

        video = torch.empty(self.channel, self.time_depth,
                            self.height, self.width, dtype=torch.float)
        # Random shift
        if self.rand_shift:
            rand_offset = int(self.time_depth*(1-self.overlap)*0.5)
            rand_shift = random.randint(-rand_offset, rand_offset)
        else:
            rand_shift = 0
        # Random horiziontal flip
        if self.hflip:
            rand_flip = bool(random.getrandbits(1))
        else:
            rand_flip = False

        start_frame = idx * self.shift + rand_shift
        end_frame = idx * self.shift + self.time_depth + rand_shift
        # print(f"idx: {idx}, start_frame: {start_frame}, end_frame: {end_frame}")

        # range ???????????? rand_shift ??? ??????.
        start = 0
        end = len(self.vfls[sample_num])
        while start + start_frame < start:
            start_frame += 1
            end_frame += 1
        while start + end_frame >= end or start + end_frame >= len(self.ppgs[sample_num]) + start:

            start_frame -= 1
            end_frame -= 1

        x, y, w, h = face_rect(self.vids[sample_num][start_frame: end_frame])
        for cnt, img in enumerate(self.vids[sample_num][start_frame: end_frame]):
            if self.crop:
                img = img[y:y + h, x: x + w, :]
            img = cv2.resize(img, (self.height, self.width),
                             interpolation=cv2.INTER_CUBIC)
            img = ToTensor()(img)
            if rand_flip:
                img = torchvision.transforms.functional.hflip(img)

            if self.normalize:
                # spatial intensity norm for each channel
                img = torch.sub(img, torch.mean(img, (1, 2)).view(3, 1, 1))
            video[:, cnt, :, :] = img  # video -> C, D, H, W

        target = torch.tensor(
            self.ppgs[sample_num][start_frame:  end_frame], dtype=torch.float)
        return video, target


class PURE_lite(Dataset):
    def __init__(self, video_path, video_fns, time_depth, ppg_offset=6, normalize=False):
        self.height = 128
        self.width = 128
        self.channel = 3
        self.time_depth = time_depth

        self.crop = True
        self.ppgs = []
        self.vfls = []
        self.nums = []
        self.num_samples = 0
        self.vids = []
        self.vlens = []
        self.normalize = normalize
        self.vdirs = video_fns

        for video_dir in video_fns:
            # Load video image list
            # get path of image file
            image_list = sorted(glob(video_path + video_dir + "/*"))
            start = 0
            end = len(image_list)

            self.vfls.append(image_list)
            vids = []
            for img_path in image_list:
                image = io.imread(img_path)
                vids.append(image)
            vlen = len(vids)
            if vlen >= self.time_depth:
                self.vids.append(vids)
                self.vlens.append(len(vids))

                print(f"Add {end} images")

                # Load PPG signals
                with open(video_path + video_dir + ".json", "r") as st_json:
                    data_json = json.load(st_json)
                time_ns = list()
                ppg = list()
                for dat in data_json['/FullPackage']:
                    time_ns.append(dat['Timestamp'])
                    ppg.append(dat['Value']['waveform'])
                time_ns = np.array(time_ns)
                ppg = np.array(ppg)
                imgt = list()

                for fname in image_list:
                    imgt.append(int(fname.split("\\")[-1][5:-4]))
                imgt = np.array(imgt)
                imgt2 = (imgt - imgt[0])/1e6
                time_ns2 = (time_ns - imgt[0])/1e6
                ppg = interpolation_ppg(imgt2, time_ns2, ppg, normalize=True)

                scaler = MinMaxScaler(feature_range=(-1, 1))
                scaler.fit(torch.Tensor(ppg).reshape(-1, 1))
                # ??? tensor??? ????????? ??? ??? ?????? ???????
                ppg = scaler.transform(torch.Tensor(
                    ppg).reshape(-1, 1)).squeeze()

                ppg = np.array(ppg)[start + ppg_offset:end]
                self.ppgs.append(ppg)

                num_samples = math.floor((end - start)/self.time_depth)

                self.num_samples += num_samples
                self.nums.append(self.num_samples-1)
        print("Initialization is Done!, total num_samples: ", self.num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # conv3d input: N x C x D x H X W
        sample_num = 0
        while idx > self.nums[sample_num]:
            sample_num += 1
        if idx > self.nums[sample_num-1]:
            idx = idx - self.nums[sample_num-1] - 1

        video = torch.empty(self.channel, self.time_depth,
                            self.height, self.width, dtype=torch.float)

        start_frame = idx * self.time_depth
        end_frame = (idx + 1) * self.time_depth
        # range ???????????? rand_shift ??? ??????.
        print(f"idx: {idx}, start_frame: {start_frame}, end_frame: {end_frame}")

        x, y, w, h = face_rect(self.vids[sample_num][start_frame: end_frame])
        for cnt, img in enumerate(self.vids[sample_num][start_frame: end_frame]):
            if self.crop:
                img = img[y:y + h, x: x + w, :]
            img = cv2.resize(img, (self.height, self.width),
                             interpolation=cv2.INTER_CUBIC)
            img = ToTensor()(img)

            if self.normalize:
                # spatial intensity norm for each channel
                img = torch.sub(img, torch.mean(img, (1, 2)).view(3, 1, 1))
            video[:, cnt, :, :] = img  # video -> C, D, H, W

        target = torch.tensor(
            self.ppgs[sample_num][start_frame:  end_frame], dtype=torch.float)
        return video, target
