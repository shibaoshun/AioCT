import math
import torch
import re
import torch.nn as nn
import numpy as np

import  os
import glob 
import logging
from skimage.metrics import structural_similarity as skssim
from skimage.metrics import peak_signal_noise_ratio as skpsnr

def psnr(img1, img2, datarange):
    if isinstance(img1, torch.Tensor):
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    psnr = skpsnr(img1.cpu().detach().numpy(), img2.cpu().detach().numpy(), data_range=datarange)
    return psnr

def aver_psnr(img1, img2, datarange):
    PSNR = 0
    assert img1.size() == img2.size()
    for i in range(img1.size()[0]):
        for j in range(img1.size()[1]):
            PSNR += psnr(img1[i, j:j + 1, ...], img2[i, j:j + 1, ...], datarange=datarange)
    return PSNR / (img1.size()[0] * img1.size()[1])


# def aver_ssim(img1, img2, datarange):
#     '''used in the training'''
#     # from skimage.measure import compare_ssim as ski_ssim
#     SSIM = 0
#     img1 = img1.cpu().numpy().astype(np.float64)
#     img2 = img2.cpu().numpy().astype(np.float64)
#     for i in range(len(img1)):
#         for j in range(img1.shape[1]):
#             SSIM += skssim(img1[i, j, ...], img2[i, j, ...], gaussian_weights=True, win_size=11,
#                            data_range=datarange,
#                            sigma=1.5)
#     return SSIM / (len(img1) * img1.shape[1])


def aver_ssim(img1, img2,datarange):
    SSIM = 0
    img1 = img1.detach().cpu().numpy().astype(np.float32)   #img1.detach().cpu().numpy().astype(np.float64)
    img2 = img2.detach().cpu().numpy().astype(np.float32)   #img2.detach().cpu().numpy().astype(np.float64)
    for i in range(len(img1)):
        for j in range(img1.shape[1]):
            SSIM += skssim(img1[i, j, ...], img2[i, j, ...], gaussian_weights=True, win_size=11, data_range=datarange,
                           sigma=1.5)
    return SSIM / (len(img1) * img1.shape[1])


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*epoch(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])


def normalize(data):
    return data / 255.


def is_image(img_name):
    if img_name.endswith(".jpg") or img_name.endswith(".bmp") or img_name.endswith(".png"):
        return True
    else:
        return  False


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def init_logger(argdict):
	r"""Initializes a logging.Logger to save all the running parameters to a
	log file

	Args:
		argdict: dictionary of parameters to be logged
	"""
	from os.path import join
	logger = logging.getLogger(__name__)
	logger.setLevel(level=logging.INFO)
	fh = logging.FileHandler(join(argdict.log_dir, 'log.txt'), mode='a')
	formatter = logging.Formatter('%(asctime)s - %(message)s')
	fh.setFormatter(formatter)
	logger.addHandler(fh)
	# try:
	# 	logger.info("Commit: {}".format(get_git_revision_short_hash()))
	# except Exception as e:
	# 	logger.error("Couldn't get commit number: {}".format(e))
	logger.info("Arguments: ")
	for k in argdict.__dict__:
		logger.info("\t{}: {}".format(k, argdict.__dict__[k]))
	return logger

def init_logger_ipol():
	r"""Initializes a logging.Logger in order to log the results after
	testing a model

	Args:
		result_dir: path to the folder with the denoising results
	"""
	logger = logging.getLogger('testlog')
	logger.setLevel(level=logging.INFO)
	fh = logging.FileHandler('out.txt', mode='w')
	formatter = logging.Formatter('%(message)s')
	fh.setFormatter(formatter)
	logger.addHandler(fh)

	return logger

def init_logger_test(result_dir):
	r"""Initializes a logging.Logger in order to log the results after testing
	a model

	Args:
		result_dir: path to the folder with the denoising results
	"""
	from os.path import join

	logger = logging.getLogger('testlog')
	logger.setLevel(level=logging.INFO)
	fh = logging.FileHandler(join(result_dir, 'log.txt'), mode='a')
	formatter = logging.Formatter('%(asctime)s - %(message)s')
	fh.setFormatter(formatter)
	logger.addHandler(fh)

	return logger


