import os
import os.path
import numpy as np
import random
import h5py
import torch
import torch.utils.data as udata
from numpy.random import RandomState
import PIL
from PIL import Image
from utils.build_120 import  initialization120, build_gemotry120
from utils.build_150 import  initialization150, build_gemotry150
from utils.build_90 import  initialization90, build_gemotry90
from utils.build_360 import  initialization360, build_gemotry360
from utils.buildsv_60 import  initialization60sv, buildsv_gemotry60
from utils.buildsv_90 import  initialization90sv, buildsv_gemotry90
from utils.buildsv_120 import  initialization120sv, buildsv_gemotry120


param360 = initialization360()
radon360,iradon360,fbp360,op_norm360 = build_gemotry360(param360)


paramsv60 = initialization60sv()
radonsv60,iradonsv60,fbpsv60,op_normsv60 = buildsv_gemotry60(paramsv60)


paramsv90 = initialization90sv()
radonsv90,iradonsv90,fbpsv90,op_normsv90 = buildsv_gemotry90(paramsv90)

paramsv120 = initialization120sv()
radonsv120,iradonsv120,fbpsv120,op_normsv120 = buildsv_gemotry120(paramsv120)


param120 = initialization120()
radon120,iradon120,fbp120,op_norm120 = build_gemotry120(param120)

param150 = initialization150()
radon150,iradon150,fbp150,op_norm150 = build_gemotry150(param150)

param90 = initialization90()
radon90,iradon90,fbp90,op_norm90 = build_gemotry90(param90)



def image_get_minmax():
    return 0.0, 1.0


def proj_get_minmax():
    return 0.0, 4.0

def normalize1(data):
    data = data
    data = data.astype(np.float32)
    data = data
    data = torch.from_numpy(np.transpose(np.expand_dims(data, 2), (2, 0, 1)))
    return data

def testnormalize(data, minmax):
    data_min, data_max = minmax
    data = np.clip(data, data_min, data_max)
    data = data * 255.0
    data = data.astype(np.float32)
    data = np.expand_dims(np.transpose(np.expand_dims(data, 2), (2, 0, 1)),0)
    return data

def normalize(data, minmax):
    data_min, data_max = minmax
    data = np.clip(data, data_min, data_max)
    data = (data - data_min) / (data_max - data_min)
    data = data * 255
    data = data.astype(np.float32)
    data = np.transpose(np.expand_dims(data, 2), (2, 0, 1))
    return data


def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1]
        if vflip: img = img[::-1, :]
        return img

    return [_augment(a) for a in args]



class MARTrainDataset(udata.Dataset):  # 继承父类为Dataset,继承其属性
    def __init__(self, dir, patchSize, length, mask, task, ang=None):  # 64 32*800
        super().__init__()
        self.dir = dir
        self.train_mask = mask
        self.patch_size = patchSize
        self.sample_num = length
        self.txtdir = os.path.join(self.dir, 'train_640geo_dir.txt')
        self.mat_files = open(self.txtdir, 'r').readlines()  # 000195_02_02/202/gt.h5/
        self.file_num = len(self.mat_files)  # 1000*90
        self.rand_state = RandomState(66)
        self.start = 0
        self.end = int(self.file_num * 0.9)  # 1000*90*0.9   # 90% of training data
        self.task = task  # 列表[0,1,2]
        self.ang = ang


    def __len__(self):  # 用于返回列表的长度
        return self.sample_num


    def __getitem__(self, idx):
        gt_dir = self.mat_files[idx % self.end]
        random_mask = random.randint(0, 79)
        file_dir = gt_dir[:-6]
        data_file = file_dir + str(random_mask) + '.h5'
        abs_dir = os.path.join(self.dir, 'train_640geo/', data_file)
        gt_absdir = os.path.join(self.dir, 'train_640geo/', gt_dir[:-1])
        gt_file = h5py.File(gt_absdir, 'r')
        Xgt = gt_file['image'][()]
        gt_file.close()
        file = h5py.File(abs_dir, 'r')
        Xma = file['ma_CT'][()]
        XLI = file['LI_CT'][()]
        file.close()
        Xgt1 = normalize1(Xgt)
        Sgt = radon360(Xgt1)
        if self.task == 0:  # svct
            sv = self.ang
            if sv == 60:
                fbp = fbpsv60
                radon = radonsv60
                Sma = radon(Xgt1)
                Xma = np.squeeze(fbp(Sma).numpy())
            if sv == 90:
                fbp = fbpsv90
                radon = radonsv90
                Sma = radon(Xgt1)
                Xma = np.squeeze(fbp(Sma).numpy())
            if sv == 120:
                fbp = fbpsv120
                radon = radonsv120
                Sma = radon(Xgt1)
                Xma = np.squeeze(fbp(Sma).numpy())
            M = np.zeros((416, 416))  # 无金属
        if self.task == 1:  # mar
            Xma = Xma
            M512 = self.train_mask[:, :, random_mask]
            M = np.array(Image.fromarray(M512).resize((416, 416), PIL.Image.BILINEAR))
        if self.task == 2:  # lact
            dl = self.ang
            if dl == 90:
                fbp = fbp90
            if dl == 120:
                fbp = fbp120
            if dl == 150:
                fbp = fbp150
            np.random.seed(120)
            Sla = Sgt[:, 0:dl, :]
            Xma = np.squeeze(fbp(Sla).numpy())
            M = np.zeros((416, 416))

        # ###增强###
        Mask, Xma, XLI, Xgt = augment(M, Xma, XLI, Xgt)
        # ###归一化###
        Xma = normalize(Xma, image_get_minmax())  # *255
        Xgt = normalize(Xgt, image_get_minmax())
        XLI = normalize(XLI, image_get_minmax())

        non_Mask = 1 - Mask.astype(np.float32)  # nomask
        non_Mask = np.transpose(np.expand_dims(non_Mask, 2), (2, 0, 1))

        return torch.from_numpy(Xgt.copy()), torch.from_numpy(non_Mask.copy()), torch.from_numpy(XLI.copy()), torch.from_numpy(Xma.copy()), self.task


class MARValDataset(udata.Dataset):
    def __init__(self, dir, mask, task, ang=None):
        super().__init__()
        self.dir = dir
        self.train_mask = mask
        self.txtdir = os.path.join(self.dir, 'train_640geo_dir.txt')
        self.mat_files = open(self.txtdir, 'r').readlines()
        self.file_num = len(self.mat_files)
        self.rand_state = RandomState(66)
        self.start = int(self.file_num * 0.9)
        self.end = int(self.file_num)
        self.sample_num = self.end - self.start
        self.task = task  # 列表[0,1,2]

        self.ang = ang

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        gt_dir = self.mat_files[idx + self.start]
        random_mask = random.randint(80, 89)
        file_dir = gt_dir[:-6]
        data_file = file_dir + str(random_mask) + '.h5'
        abs_dir = os.path.join(self.dir, 'train_640geo/', data_file)
        gt_absdir = os.path.join(self.dir, 'train_640geo/', gt_dir[:-1])
        gt_file = h5py.File(gt_absdir, 'r')
        Xgt = gt_file['image'][()]
        gt_file.close()
        file = h5py.File(abs_dir, 'r')
        Xma = file['ma_CT'][()]
        XLI = file['LI_CT'][()]
        file.close()
        Xgt1 = normalize1(Xgt)
        Sgt = radon360(Xgt1)
        if self.task == 0:  # svct
            sv = self.ang
            if sv == 60:
                fbp = fbpsv60
                radon = radonsv60
                Sma = radon(Xgt1)
                Xma = np.squeeze(fbp(Sma).numpy())
            if sv == 90:
                fbp = fbpsv90
                radon = radonsv90
                Sma = radon(Xgt1)
                Xma = np.squeeze(fbp(Sma).numpy())
            if sv == 120:
                fbp = fbpsv120
                radon = radonsv120
                Sma = radon(Xgt1)
                Xma = np.squeeze(fbp(Sma).numpy())
            M = np.zeros((416, 416))  # 无金属
        if self.task == 1:  # mar
            Xma = Xma
            M512 = self.train_mask[:, :, random_mask]
            M = np.array(Image.fromarray(M512).resize((416, 416), PIL.Image.BILINEAR))
        if self.task == 2:  # lact
            dl = self.ang
            if dl == 90:
                fbp = fbp90
            if dl == 120:
                fbp = fbp120
            if dl == 150:
                fbp = fbp150
            np.random.seed(120)
            Sla = Sgt[:, 0:dl, :]
            Xma = np.squeeze(fbp(Sla).numpy())
            M = np.zeros((416, 416))  # 无金属

        ###归一化###
        Xma = normalize(Xma, image_get_minmax())  # *255
        Xgt = normalize(Xgt, image_get_minmax())
        XLI = normalize(XLI, image_get_minmax())

        non_Mask = 1 - M.astype(np.float32)  # nomask
        non_Mask = np.transpose(np.expand_dims(non_Mask, 2), (2, 0, 1))

        return torch.from_numpy(Xgt.copy()), torch.from_numpy(non_Mask.copy()), torch.from_numpy(XLI.copy()), torch.from_numpy(Xma.copy())



#
def test_image(data_path, imag_idx, mask_idx, task, ang=None):
    txtdir = os.path.join(data_path, 'test_640geo_dir.txt')
    mat_files = open(txtdir, 'r').readlines()
    gt_dir = mat_files[imag_idx]
    file_dir = gt_dir[:-6]
    data_file = file_dir + str(mask_idx) + '.h5'
    abs_dir = os.path.join(data_path, 'test_640geo/', data_file)
    gt_absdir = os.path.join(data_path, 'test_640geo/', gt_dir[:-1])
    gt_file = h5py.File(gt_absdir, 'r')
    Xgt = gt_file['image'][()]
    gt_file.close()
    file = h5py.File(abs_dir, 'r')
    Xma = file['ma_CT'][()]
    XLI = file['LI_CT'][()]
    Xgt1 = normalize1(Xgt)
    Sgt = radon360(Xgt1)
    file.close()
    test_mask = np.load(os.path.join(data_path, 'testmask.npy'))
    if task == 0:  # svct
        sv = ang
        if sv == 60:
            fbp = fbpsv60
            radon = radonsv60
            Sma = radon(Xgt1)
            Xma = np.squeeze(fbp(Sma).numpy())
        if sv == 90:
            fbp = fbpsv90
            radon = radonsv90
            Sma = radon(Xgt1)
            Xma = np.squeeze(fbp(Sma).numpy())
        if sv == 120:
            fbp = fbpsv120
            radon = radonsv120
            Sma = radon(Xgt1)
            Xma = np.squeeze(fbp(Sma).numpy())
        M = np.zeros((416, 416))  # 无金属
    if task == 1:  # mar
        Xma = Xma
        M512 = test_mask[:, :, mask_idx]
        M = np.array(Image.fromarray(M512).resize((416, 416), PIL.Image.BILINEAR))
    if task == 2:  # lact
        dl = ang
        if dl == 90:
            fbp = fbp90
        if dl == 120:
            fbp = fbp120
        if dl == 150:
            fbp = fbp150
        np.random.seed(120)
        Sla = Sgt[:, 0:dl, :]
        Xma = np.squeeze(fbp(Sla).numpy())
        M = np.zeros((416, 416))  # 无金属


    ###归一化###
    Xma = testnormalize(Xma, image_get_minmax())  # *255
    Xgt = testnormalize(Xgt, image_get_minmax())
    XLI = testnormalize(XLI, image_get_minmax())

    non_Mask = 1 - M.astype(np.float32)  # nomask
    non_Mask = np.expand_dims(np.transpose(np.expand_dims(non_Mask, 2), (2, 0, 1)), 0)

    return torch.Tensor(Xma).cuda(), torch.Tensor(Xgt).cuda(), torch.Tensor(XLI).cuda(), torch.Tensor(non_Mask).cuda()
