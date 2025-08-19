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
from models.network import MaskAwareNet

torch.cuda.set_device(0)


parser = argparse.ArgumentParser(description="MaskAwareNet")
parser.add_argument("--model_dir", type=str, default="models/MaskAwareNet_best.pth", help='path to model file')
parser.add_argument("--data_path", type=str, default=r'/media/asus/代码/dataset_ct/360', help='path to test data')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument('--train_ps', type=int, default=416, help='patch size of training sample')
parser.add_argument('--embed_dim', type=int, default=32, help='dim of emdeding features')
parser.add_argument('--win_size', type=int, default=4, help='window size of self-attention')
parser.add_argument('--token_projection', type=str, default='linear', help='linear/conv token projection')
parser.add_argument('--token_mlp', type=str, default='leff', help='ffn/leff token mlp')
parser.add_argument("--save_path", type=str, default="save_results_1/", help='path to testing results')
parser.add_argument('--batchSize', type=int, default=1, help='testing input batch size')
parser.add_argument('--log_dir', default='logs/test/', help='tensorboard logs')
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


def image_get_minmax():
    return 0.0, 1.0

def normalize(data, minmax):
    data_min, data_max = minmax
    data = np.clip(data, data_min, data_max)
    data = data * 255.0
    data = data.astype(np.float32)
    data = np.expand_dims(np.transpose(np.expand_dims(data, 2), (2, 0, 1)),0)
    return data


def tohu(X):
    CT = (X - 0.192) * 1000 / 0.192
    CT_win = CT.clamp_(-175, 275)
    CT_winnorm = (CT_win + 175) / (275+175)
    return CT_winnorm

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
    file = h5py.File(abs_dir, 'r')
    Xma = file['ma_CT'][()]
    XLI = file['LI_CT'][()]
    file.close()
    M512 = test_mask[:, :, mask_idx]
    M = np.array(Image.fromarray(M512).resize((416, 416), PIL.Image.BILINEAR))
    Xma = normalize(Xma, image_get_minmax())
    Xgt = normalize(Xgt, image_get_minmax())
    XLI = normalize(XLI, image_get_minmax())
    Mask = M.astype(np.float32)
    Mask = np.expand_dims(np.transpose(np.expand_dims(Mask, 2), (2, 0, 1)), 0)
    non_mask = 1 - Mask
    return torch.Tensor(Xma).cuda(), torch.Tensor(Xgt).cuda(), torch.Tensor(XLI).cuda(), torch.Tensor(non_mask).cuda()




def main():
    # Build model
    print('Loading model ...\n')
    model = MaskAwareNet(opt).cuda()
    state_dict = torch.load(opt.model_dir)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
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
                Xout, Eout = model(Xma, 0)
                end_time = time.time()
                dur_time = end_time - start_time
                time_test += dur_time
            Xoutclip = torch.clamp(Xout / 255.0, 0, 0.5)
            Xgtclip = torch.clamp(X / 255.0, 0, 0.5)
            Xmaclip = torch.clamp(Xma / 255.0, 0, 0.5)
            Xoutnorm = Xoutclip / 0.5
            Xgtnorm = Xgtclip / 0.5
            Xmanorm = Xmaclip / 0.5
            Xoutnorm1 = Xoutnorm
            Xgtnorm1 = Xgtnorm
            Xmanorm1 = Xmanorm
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

