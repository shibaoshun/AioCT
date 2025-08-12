import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import os.path
import argparse
import numpy as np
import torch
torch.cuda.set_device(0)
import time
import matplotlib
matplotlib.use('TkAgg')
import utils.save_image as save_img
from utils.utils import init_logger, aver_psnr, aver_ssim
from Dataset import test_image
from models.network import MaskAwareNet





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
parser.add_argument("--save_path", type=str, default="save_results", help='path to testing results')
parser.add_argument('--batchSize', type=int, default=1, help='testing input batch size')
parser.add_argument('--log_dir', default='logs/test/', help='tensorboard logs')
parser.add_argument('--task', type=int, default='0', help='type')
parser.add_argument('--ang', type=int, default='60', help='type')

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


save_path = os.path.join(f"{opt.save_path}_{opt.task}_{opt.ang}")


out_dir = save_path+'/MaskAwareNet/image/'
out_hudir = save_path+'/MaskAwareNet/hu/'
mkdir(out_dir)
mkdir(out_hudir)

input_dir = save_path+'/input/image/'
input_hudir = save_path+'/input/hu/'
mkdir(input_dir)
mkdir(input_hudir)

gt_dir = save_path+'/gt/image/'
gt_hudir = save_path+'/gt/hu/'
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
    from thop import profile
    model.eval()
    time_test = 0
    count = 0
    psnr_per_epoch = 0
    ssim_per_epoch = 0
    rmsect_dudo = 0
    rmse_inter = 0

    task = opt.task
    ang = opt.ang


    for imag_idx in range(200):
        print("imag_idx:", imag_idx)
        Xma, X, XLI, M = test_image(opt.data_path, imag_idx, 0, task, ang)
        with torch.no_grad():
            if opt.use_GPU:
                torch.cuda.synchronize()
            start_time = time.time()
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
            rmsectdudo = torch.sqrt(torch.mean(((Xoutclip - Xgtclip) * M) ** 2)) * 1000 / 0.192
            rmsect_dudo += rmsectdudo
            rmse = torch.sqrt(torch.mean(((Xoutclip - Xgtclip) * M) ** 2))
            rmse_inter += rmse
            idx = imag_idx + 1

            Xnorm = [Xoutnorm1, Xmanorm1, Xgtnorm1]
            Xhu = [Xouthu, Xmahu, Xgthu]
            dir = [out_dir, input_dir, gt_dir]
            hudir = [out_hudir, input_hudir, gt_hudir]
            save_img.imwrite(idx, dir, Xnorm)
            save_img.imwrite(idx, hudir, Xhu)
            psnr_iter = aver_psnr(Xoutnorm * M, Xgtnorm * M, 1)
            psnr_per_epoch += psnr_iter.item()
            ssim_iter = aver_ssim(Xoutnorm * M, Xgtnorm * M, 1)
            ssim_per_epoch += ssim_iter.item()

            print(" PSNR %f" % psnr_iter)
            print(" SSIM %f" % ssim_iter)
            print(" RMSE %f" % rmse)
            print(" RMSEHU %f" % rmsectdudo)
            print('Times: ', dur_time)
            logger.info("\t image:{}   psnr:{:.4f}  ssim:{:.4f}  rmse:{:.4f}  rmsehu:{:.4f}   time:{:.4f}"
                        .format(imag_idx + 1, psnr_iter, ssim_iter, rmse, rmsectdudo, dur_time))
            count += 1

    print(100 * '*')
    print(
        'task：{}, angle:{}, Avg.PSNR={:.2f}, Avg.SSIM={:.4f}, Avg.RMSE={:.5f}'.format(task, ang,
                                                                                      psnr_per_epoch / count,
                                                                                      ssim_per_epoch / count,
                                                                                      rmsect_dudo / count))

    logger.info("\t deeplesion_test_200_10: task:{}  angle:{} -average- avg_psnr:{:.4f}  avg_ssim:{:.4f}"
                .format(task, ang, psnr_per_epoch / count, ssim_per_epoch / count))
    print(100 * '*')



if __name__ == "__main__":
    main()

