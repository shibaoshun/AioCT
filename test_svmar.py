import os
import os.path
import argparse
import numpy as np
import torch
import time
import h5py
from utils import utils_image
import utils.save_image as save_img
from utils.utils import init_logger
from utils.build_gemotry import initialization, build_gemotry
import odl
from scipy.interpolate import interp1d
import csv
from network import MaskAwareNet
import os
import os.path
import argparse
import numpy as np
import torch
import time
import h5py
import matplotlib
matplotlib.use('TkAgg')
from utils.utils import init_logger, aver_psnr, aver_ssim
import PIL
from PIL import Image
import utils.save_image as save_img
from utils.utils import init_logger
from network import MaskAwareNet
torch.cuda.set_device(4)

param = initialization()
ray_trafo = build_gemotry(param)
FBP_360 = odl.tomo.fbp_op(ray_trafo)
# op_modfp = odl_torch.OperatorModule(ray_trafo)
# op_modpT = odl_torch.OperatorModule(ray_trafo.adjoint)

parser = argparse.ArgumentParser(description="MaskAwareNet")
parser.add_argument("--model_dir", type=str, default="models/MaskAwareNet_best.pth", help='path to model file')
parser.add_argument("--data_path", type=str, default=r'/home/lthpc/code/wcw/shuju/360', help='path to test data')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument('--train_ps', type=int, default=416, help='patch size of training sample')
parser.add_argument('--embed_dim', type=int, default=32, help='dim of emdeding features')
parser.add_argument('--win_size', type=int, default=4, help='window size of self-attention')
parser.add_argument('--token_projection', type=str, default='linear', help='linear/conv token projection')
parser.add_argument('--token_mlp', type=str, default='leff', help='ffn/leff token mlp')
parser.add_argument("--save_path", type=str, default="save_results_3/", help='path to testing results')
parser.add_argument('--batchSize', type=int, default=1, help='testing input batch size')
parser.add_argument('--log_dir', default='logs3/test/', help='tensorboard logs')
opt = parser.parse_args()

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

# create path
try:
    os.makedirs(opt.log_dir)
except OSError:
    pass
  # Init loggers
logger = init_logger(opt)

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  " + path + "  ---")
    else:
        print("---  There exsits folder " + path + " !  ---")

out_dir = opt.save_path+'/MaskAwareNet/image/'
out_hudir = opt.save_path+'/MaskAwareNet/hu/'
mkdir(out_dir)
mkdir(out_hudir)

input_dir = opt.save_path+'/input/image/'
input_hudir = opt.save_path+'/input/hu/'
mkdir(input_dir)
mkdir(input_hudir)

gt_dir = opt.save_path+'/gt/image/'
gt_hudir = opt.save_path+'/gt/hu/'
mkdir(gt_dir)
mkdir(gt_hudir)

def tohu(X):           # display window as [-175HU, 275HU]
    CT = (X - 0.192) * 1000 / 0.192
    CT_win = CT.clamp_(-175, 275)
    CT_winnorm = (CT_win +175) / (275+175)
    return CT_winnorm
def image_get_minmax():
    return 0.0, 1.0
def proj_get_minmax():
    return 0.0, 4.0

# ######180#################
def SVimage_get_minmax_05():
    return 0.0, 0.5
########90#################
# def SVimage_get_minmax_02():
#     return 0.0, 0.2
########60#################
# def SVimage_get_minmax_015():
#     return 0.0, 0.15

def normalize(data, minmax):
    data_min, data_max = minmax
    data = np.clip(data, data_min, data_max)
    data = (data - data_min) / (data_max - data_min)
    data = data.astype(np.float32)
    data = data*255.0
    data = np.expand_dims(np.transpose(np.expand_dims(data, 2), (2, 0, 1)), 0)
    return data

def normalize_SVX(data, minmax):
    data_min, data_max = minmax
    data = np.clip(data, data_min, data_max)
    # data = (data - data_min) / (data_max - data_min)
    data = data / 0.5 #######180
    # data = data / 0.3 #######120
    # data = data / 0.2 #######90
    # data = data / 0.15  #######60
    data = data * 255.0
    data = data.astype(np.float32)
    data = np.expand_dims(np.transpose(np.expand_dims(data, 2), (2, 0, 1)), 0)
    return data


def interpolate_projection(proj, metalTrace):
    # projection linear interpolation
    # Input:
    # proj:         uncorrected projection
    # metalTrace:   metal trace in projection domain (binary image)
    # Output:
    # Pinterp:      linear interpolation corrected projection
    Pinterp = proj.copy()
    for i in range(Pinterp.shape[0]):
        mslice = metalTrace[i]
        pslice = Pinterp[i]

        metalpos = np.nonzero(mslice==1)[0]
        nonmetalpos = np.nonzero(mslice==0)[0]
        pnonmetal = pslice[nonmetalpos]
        pslice[metalpos] = interp1d(nonmetalpos,pnonmetal)(metalpos)
        Pinterp[i] = pslice

    return Pinterp

def uniform_extract(x):  ####metal_sv
    x1 = np.asarray(x)
    # #######180#########
    x1[1:361:2,:] = 0

    ######120#########
    # x1[1:361:3, :] = 0  # 90
    # x1[2:362:3, :] = 0  # 90
    # x1[3:363:3, :] = 0  # 90


    #
    ######90#########
    # x1[1:361:4, :] = 0  # 90
    # x1[2:362:4, :] = 0  # 90
    # x1[3:363:4, :] = 0  # 90

    # #########60#########
    # x1[1:361:6, :] = 0  # 60
    # x1[2:362:6, :] = 0  # 60
    # x1[3:363:6, :] = 0  # 60
    # x1[4:364:6, :] = 0  # 60
    # x1[5:365:6, :] = 0  # 60
    return x1


test_mask = np.load(os.path.join(opt.data_path, 'testmask.npy'))
def test_image(data_path, imag_idx, mask_idx):
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
    Sgt = np.asarray(ray_trafo(Xgt))
    file = h5py.File(abs_dir, 'r')
    Xma = file['ma_CT'][()]
    Sma = file['ma_sinogram'][()]
    Tr = file['metal_trace'][()]
    file.close()

    S_svma = uniform_extract(Sma)  # 下采样
    X_svma = np.asarray(FBP_360(S_svma))

    M512 = test_mask[:, :, mask_idx]
    M = np.array(Image.fromarray(M512).resize((416, 416), PIL.Image.BILINEAR))
    Mask = M.astype(np.float32)
    no_mask = 1 - Mask
    SLI_svma = interpolate_projection(S_svma, Tr)
    XLI_svma = np.asarray(FBP_360(SLI_svma))

    X_svma = normalize_SVX(X_svma, SVimage_get_minmax_05())  ################归一化###########
    X_GT = normalize(Xgt, image_get_minmax())
    XLI_svma = normalize(XLI_svma, image_get_minmax())
    S_svma = normalize(S_svma, proj_get_minmax())
    S_GT = normalize(Sgt, proj_get_minmax())
    SLI_svma = normalize(SLI_svma, proj_get_minmax())

    x = np.full((360, 641), 1)
    metal_sv = uniform_extract(x)
    metal_sv = metal_sv.astype(np.float32)
    metal_sv = np.transpose(np.expand_dims(metal_sv, 2), (2, 0, 1))

    Tr = 1 - Tr.astype(np.float32)
    Tr = np.transpose(np.expand_dims(Tr, 2), (2, 0, 1))
    no_mask = np.transpose(np.expand_dims(no_mask, 2), (2, 0, 1))
    return torch.Tensor(X_svma).cuda(), torch.Tensor(X_GT).cuda(), torch.Tensor(XLI_svma).cuda(), torch.Tensor(no_mask).cuda()

def main():
    # Build model
    print('Loading model ...\n')
    # model = MaskAwareNet(opt).cuda()
    # model.load_state_dict(torch.load(opt.model_dir))
    model = MaskAwareNet(opt).cuda()
    # 加载多卡保存的模型，并移除参数名中的module.前缀
    state_dict = torch.load(opt.model_dir)
    # 新建无module.前缀的参数字典
    new_state_dict = {}
    for k, v in state_dict.items():
        # 移除"module."前缀（若存在）
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    # 加载处理后的参数
    model.load_state_dict(new_state_dict)
    model.eval()
    time_test = 0
    count = 0
    psnr_per_epoch = 0
    ssim_per_epoch = 0
    rmsect_dudo = 0
    rmse_inter = 0
    list1 = []
    list2 = []

    list1_1 = []
    list2_2 = []


    for imag_idx in range(200):
       print("imag_idx:",imag_idx)
       for mask_idx in range(10):
            Xma, X, XLI, M = test_image(opt.data_path, imag_idx, mask_idx)
            with torch.no_grad():
                if opt.use_GPU:
                    torch.cuda.synchronize()
                start_time = time.time()
                # import matplotlib.pyplot as plt
                # artfict1 = torch.clamp(Xma / 255.0, 0, 0.5)
                # artfict2 = X
                # artfict3 = Xma - X
                #
                # plt.imsave('ART1.png', artfict1.cpu().numpy().squeeze(), cmap="gray")
                # plt.imsave('ART2.png', artfict2.cpu().numpy().squeeze(), cmap="gray")
                # plt.imsave('ART3.png', artfict3.cpu().numpy().squeeze(), cmap="gray")
                Xout, Eout, sne = model(Xma, 0)


                end_time = time.time()
                dur_time = end_time - start_time
                time_test += dur_time
            Xoutclip = torch.clamp(Xout / 255.0, 0, 0.5)
            Xgtclip = torch.clamp(X / 255.0, 0, 0.5)
            Xmaclip = torch.clamp(Xma / 255.0, 0, 0.5)
            Xoutnorm = Xoutclip / 0.5
            Xgtnorm = Xgtclip / 0.5
            Xmanorm = Xmaclip / 0.5
            Xoutnorm1 = Xoutnorm * M
            Xgtnorm1 = Xgtnorm * M
            Xmanorm1 = Xmanorm * M
            Xouthu = tohu(Xoutclip)
            Xgthu = tohu(Xgtclip)
            Xmahu = tohu(Xmaclip)
            rmsectdudo = torch.sqrt(torch.mean(((Xoutclip - Xgtclip) * M) ** 2))* 1000 / 0.192
            rmsect_dudo += rmsectdudo
            rmse = torch.sqrt(torch.mean(((Xoutclip - Xgtclip) * M) ** 2))
            rmse_inter += rmse
            idx = imag_idx * 10 + mask_idx + 1
            Xnorm = [Xoutnorm1, Xmanorm1, Xgtnorm1]
            Xhu = [Xouthu, Xmahu, Xgthu]
            dir = [out_dir, input_dir, gt_dir]
            hudir = [out_hudir, input_hudir, gt_hudir]
            save_img.imwrite(idx, dir, Xnorm)
            save_img.imwrite(idx, hudir,Xhu)
            psnr_iter = aver_psnr(Xoutnorm * M, Xgtnorm * M, 1)
            psnr_per_epoch += psnr_iter.item()
            ssim_iter = aver_ssim(Xoutnorm * M, Xgtnorm * M, 1)
            ssim_per_epoch += ssim_iter.item()

            if mask_idx == 1 or mask_idx == 7 or mask_idx == 2 or mask_idx == 6:
                list1.append(psnr_iter)
                list1_1.append(ssim_iter)
            if mask_idx == 0 or mask_idx == 8 or mask_idx == 4 or mask_idx == 9 or mask_idx == 3 or mask_idx == 5:
                list2.append(psnr_iter)
                list2_2.append(ssim_iter)


            print(" PSNR %f" % psnr_iter)
            print(" SSIM %f" % ssim_iter)
            print(" RMSE %f" % rmse)
            print(" RMSEHU %f" % rmsectdudo)
            print('Times: ', dur_time)
            logger.info("\t image:{} mask:{}  psnr:{:.4f}  ssim:{:.4f}  rmse:{:.4f}  rmsehu:{:.4f}   time:{:.4f}"
                .format(imag_idx + 1, mask_idx + 1, psnr_iter, ssim_iter, rmse, rmsectdudo, dur_time))
            count += 1

    total1 = sum(list1)
    total2 = sum(list2)


    total1_1 = sum(list1_1)
    total2_2 = sum(list2_2)


    lasm1 = total1 / 800
    lasm1_1 = total1_1 / 800
    lasm2 = total2 / 1200
    lasm2_2 = total2_2 / 1200



    print(100 * '*')
    print('total1.PSNR={:.2f}, total1.SSIM={:.4f}'.format(lasm1, lasm1_1))
    print('total2.PSNR={:.2f}, total2.SSIM={:.4f}'.format(lasm2, lasm2_2))

    print('Avg.PSNR={:.2f}, Avg.SSIM={:.4f}, Avg.RMSE={:.4f}'.format(psnr_per_epoch / count, ssim_per_epoch / count, rmsect_dudo / count))

    logger.info(
        "\t deeplesion_test_200_10: large to small  psnr:{:.4f}/ssim:{:.4f} psnr:{:.4f}/ssim:{:.4f}"
        .format(lasm1, lasm1_1, lasm2, lasm2_2))
    logger.info("\t deeplesion_test_200_10: -average- avg_psnr:{:.4f}  avg_ssim:{:.4f}"
                .format(psnr_per_epoch / count, ssim_per_epoch / count))
    print(100 * '*')
if __name__ == "__main__":
    main()
