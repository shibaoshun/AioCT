#!/usr/bin/env python
# -*- coding:utf-8 -*-
# IJCAI 2022
# Adaptive Convolutional Dictionary Network for CT Metal Artifact Reduction

from __future__ import print_function
import argparse
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from tensorboardX import SummaryWriter
from network import MaskAwareNet
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from Dataset import MARTrainDataset, MARValDataset
from math import ceil
from loss import VGGLoss, airnet_uncertainty_loss, ContrastiveLoss
from utils.utils import init_logger, aver_psnr, aver_ssim







def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)

def main(opt):

    _modes = ['train', 'val']  # model有训练有测试
    #—————————————————— create path————————————————————————
    try:
        os.makedirs(opt.log_dir)
    except OSError:
        pass
    try:
        os.makedirs(opt.model_dir)
    except OSError:
        pass
    writer = SummaryWriter(opt.log_dir)
    logger = init_logger(opt)
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)  # 函数返回参数1和参数2之间的任意整数， 闭区间
    print("Random Seed: ", opt.manualSeed)  # 随机种子无限接近想要的值
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    model = MaskAwareNet(opt).cuda()
    model = torch.nn.DataParallel(model)
    cri_vgg = VGGLoss()
    cri_eun = airnet_uncertainty_loss()
    cri_cra = ContrastiveLoss()

    print_network(model)#打印参数

    optimizer = optim.AdamW(model.parameters(), betas=(0.9, 0.999), lr=opt.lr)
    #——————————————————— load dataset 加载数据集———————————————
    train_mask = np.load(os.path.join(opt.data_path, 'trainmask.npy'))
    train_dataset1 = MARTrainDataset(opt.data_path, opt.patchSize, int(opt.batchnum), train_mask,0,60)
    train_dataset2 = MARTrainDataset(opt.data_path, opt.patchSize, int(opt.batchnum), train_mask,0,90)
    train_dataset3 = MARTrainDataset(opt.data_path, opt.patchSize, int(opt.batchnum), train_mask,0,120)
    train_dataset4 = MARTrainDataset(opt.data_path, opt.patchSize, int(opt.batchnum), train_mask,1)
    train_dataset5 = MARTrainDataset(opt.data_path, opt.patchSize, int(opt.batchnum), train_mask,2,90)
    train_dataset6 = MARTrainDataset(opt.data_path, opt.patchSize, int(opt.batchnum), train_mask,2,120)
    train_dataset7 = MARTrainDataset(opt.data_path, opt.patchSize, int(opt.batchnum), train_mask,2,150)

    train_dataset = ConcatDataset([train_dataset1,train_dataset2,train_dataset3,train_dataset4,train_dataset5,train_dataset6,train_dataset7])

    val_dataset1 = MARValDataset(opt.data_path, train_mask,0,60)
    val_dataset2 = MARValDataset(opt.data_path, train_mask,0,90)
    val_dataset3 = MARValDataset(opt.data_path, train_mask,0,120)
    val_dataset4 = MARValDataset(opt.data_path, train_mask,1)
    val_dataset5 = MARValDataset(opt.data_path, train_mask,2,90)
    val_dataset6 = MARValDataset(opt.data_path, train_mask,2,120)
    val_dataset7 = MARValDataset(opt.data_path, train_mask,2,150)
    val_dataset = ConcatDataset([val_dataset1, val_dataset2, val_dataset3, val_dataset4, val_dataset5, val_dataset6,val_dataset7])

    datasets = {'train': train_dataset, 'val': val_dataset}

    #————————————————— train and val data ——————————————————————————————
    batch_size = {'train': opt.batchSize, 'val': 1}
    data_loader = {phase: DataLoader(datasets[phase], batch_size=batch_size[phase], shuffle=True, num_workers=int(opt.workers), pin_memory=True) for phase in _modes}  # train800 val 100

    num_data = {phase: len(datasets[phase]) for phase in _modes}
    num_iter_epoch = {phase: ceil(num_data[phase] / batch_size[phase]) for phase in _modes}

    #———————————— Resume training or start a new————————————
    yuxunlian = 0
    if yuxunlian:
        resumef = os.path.join(opt.log_dir, 'ckpt.pth')
        if os.path.isfile(resumef):
            checkpoint = torch.load(resumef)
            print("> Resuming previous training")
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            new_epoch = opt.epochs
            new_milestone = opt.milestone
            opt = checkpoint['opt']
            training_params = checkpoint['training_params']
            current_lr = training_params['current_lr']
            start_epoch = training_params['start_epoch']
            best_psnr = training_params['best_psnr']
            opt.epochs = new_epoch
            opt.milestone = new_milestone
            print("=> loaded checkpoint '{}' (epoch {})".format(resumef, start_epoch))

            print("=> loaded parameters :")
            print("==> checkpoint['optimizer']['param_groups']")
            print("\t{}".format(checkpoint['optimizer']['param_groups']))
            print("==> checkpoint['training_params']")
            for k in checkpoint['training_params']:
                print("\t{}, {}".format(k, checkpoint['training_params'][k]))
            argpri = vars(checkpoint['opt'])
            print("==> checkpoint['opt']")
            for k in argpri:
                print("\t{}, {}".format(k, argpri[k]))

            opt.resume_training = False
        else:
            raise Exception("Couldn't resume training with checkpoint {}".format(resumef))
    else:
        current_lr = opt.lr
        training_params = {}
        start_epoch = 0
        best_psnr = 0
        training_params['step'] = 0
        training_params['current_lr'] = current_lr

    # —————————————————      training   ——————————————————————————————
    step = 0  # 迭代次数为0
    #——————————————————————调整学习率————————————————————————————
    for epoch in range(start_epoch, opt.epochs): # epoch=[0--300)=0-299
        time_start = time.time()
        if epoch > opt.milestone[3]:
            current_lr = opt.lr / 16
        elif  opt.milestone[2] < epoch and epoch < opt.milestone[3] or epoch == opt.milestone[3]:
            current_lr = opt.lr / 8
        elif opt.milestone[1] < epoch and epoch < opt.milestone[2] or epoch == opt.milestone[2]:
            current_lr = opt.lr / 4
        elif opt.milestone[0] < epoch and epoch < opt.milestone[1] or epoch == opt.milestone[1]:
            current_lr = opt.lr / 2
        else:
            current_lr = opt.lr
        # # set learning rate
        training_params['current_lr'] = current_lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)

        # ———————————————————— train stage————————————————————————————————————
        mse_per_epoch = {x: 0 for x in _modes}  # {train:0 val:0}
        psnr_per_epoch = {x: 0 for x in _modes}
        ssim_per_epoch = {x: 0 for x in _modes}
        tic = time.time()  # 获取当前时间
        phase = 'train'
        optimizer.zero_grad()  # 意思是把梯度置零，也就是把loss关于weight的导数变成0

        for ii, data in enumerate(data_loader[phase]):
            GT, mask, XLI, Xma, task = [x.cuda() for x in data]

            model.train()
            optimizer.zero_grad()
            Xout, Eout, sne, un = model(Xma,1)
            newXgt = mask * GT

            contrastive_prompt_loss = 0.005 * cri_cra(sne, GT)
            lossE = cri_eun(Eout * mask, (Xma - GT) * mask, task, un) * 0.0001
            lossX2 = F.mse_loss(Xout * mask, newXgt)
            lossvgg = cri_vgg(Xout * mask, newXgt) * 0.1
            loss = lossE + lossX2 + lossvgg + contrastive_prompt_loss
            # back propagation
            loss.backward()
            optimizer.step()
            model.eval()
            mse_iter = loss.item()
            mse_per_epoch[phase] += mse_iter

            Xoutclip = torch.clamp(Xout / 255.0, 0, 0.5)
            Xgtclip = torch.clamp(GT / 255.0, 0, 0.5)  # 数值在0-0.5
            rmseu = torch.sqrt(torch.mean(((Xoutclip - Xgtclip) * mask) ** 2))  # RMSE

            train_psnr = aver_psnr(Xoutclip * mask, Xgtclip * mask, 0.5)
            train_ssim = aver_ssim(Xoutclip * mask, Xgtclip * mask, 0.5)

            psnr_per_epoch[phase] += train_psnr.item()
            ssim_per_epoch[phase] += train_ssim.item()
            ##########################可视化########################可视化loss 每100个图像
            if ii % 100 == 0:
                template = '[Epoch:{:>2d}/{:<2d}] {:s}:{:0>5d}/{:0>5d}, Loss={:5.2e}, LossE={:5.2e}, LossX2={:5.2e}, Lossvgg={:5.2e}, Losscra={:5.2e}, lr={:.2e}'
                print(template.format(epoch + 1, opt.epochs, phase, ii, num_iter_epoch[phase], mse_iter, lossE, lossX2, lossvgg, contrastive_prompt_loss, current_lr))
                log_str = 'rmseu={:5.4f},psnr={:4.2f}, ssim= {:5.4f}'
                print(log_str.format(rmseu.item(), train_psnr.item(), train_ssim.item()))
            writer.add_scalar('Train Loss Iter', mse_iter, step)  # 可视化loss
            step += 1
            ##########################可视化########################可视化psnr loss 每10个图形
            if training_params['step'] % opt.save_every == 0:  # opt.save_every=10
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), training_params['step'])
                writer.add_scalar('PSNR on training data', train_psnr.item(), training_params['step'])
            training_params['step'] += 1
        mse_per_epoch[phase] /= (ii + 1)
        psnr_per_epoch[phase] /= (ii + 1)
        ssim_per_epoch[phase] /= (ii + 1)
        print('{:s}: Loss={:+.2e} PSNR={:4.2f} SSIM={:5.4f}'.format(phase, mse_per_epoch[phase] , psnr_per_epoch[phase],ssim_per_epoch[phase]))
        print('-' * 100)
        del GT, mask, XLI, Xma, task, Xout, Eout, sne, un, newXgt, contrastive_prompt_loss, lossE, lossX2, lossvgg, loss
        torch.cuda.empty_cache()
        #—————————————————————————— evaluation stage——————————————————————————
        model.eval()

        phase = 'val'
        with torch.no_grad():
            for ii, data in enumerate(data_loader[phase]):
                GT, mask, XLI, Xma = [x.cuda() for x in data]

                with torch.set_grad_enabled(False):
                    Xout, Eout, SNE = model(Xma,0)
                newXgt = mask * GT
                Xoutclip = torch.clamp(Xout / 255.0, 0, 0.5)
                Xgtclip = torch.clamp(GT / 255.0, 0, 0.5)

                psnr_iter = aver_psnr(Xoutclip * mask, Xgtclip * mask, 0.5)
                ssim_iter = aver_ssim(Xoutclip * mask, Xgtclip * mask, 0.5)

                psnr_per_epoch[phase] += psnr_iter
                ssim_per_epoch[phase] += ssim_iter
                Xout.clamp_(0.0, 255.0)
                mse_iter = F.mse_loss(mask * Xout, newXgt)
                mse_per_epoch[phase] += mse_iter
                del GT, mask, XLI, Xma, Xout, Eout, SNE, newXgt, Xoutclip, Xgtclip
                torch.cuda.empty_cache()
                if ii % 10 == 0:
                        log_str = '[Epoch:{:>2d}/{:<2d}] {:s}:{:0>3d}/{:0>3d}, mae={:.2e}, psnr={:4.2f}, ssim= {:5.4f}'
                        print(log_str.format(epoch + 1, opt.epochs, phase, ii + 1, num_iter_epoch[phase], mse_iter, psnr_iter,
                                             ssim_iter))
            psnr_per_epoch[phase] /= (ii + 1)
            ssim_per_epoch[phase] /= (ii + 1)
            mse_per_epoch[phase] /= (ii + 1)
            time_end = time.time()
            cur_time = time_end - time_start
            print('{:s}: mse={:.3e}, PSNR={:4.2f}, ssim= {:5.4f}'.format(phase, mse_per_epoch[phase], psnr_per_epoch[phase],ssim_per_epoch[phase]))
            logger.info("\t{:s}: current_epoch:{}  psnr_val:{:.4f} ssim_val:{:.4f} time:{:.2f}  best_psnr:{:.4f}"
                        .format(phase, epoch + 1, psnr_per_epoch[phase],ssim_per_epoch[phase], cur_time, best_psnr))
            print('-' * 100)
        torch.cuda.empty_cache()
        #————————————————————————————— save best model————————————————————————
        training_params['start_epoch'] = epoch + 1
        if psnr_per_epoch[phase] > best_psnr:
            best_psnr = psnr_per_epoch[phase]
            training_params['best_psnr'] = best_psnr
            model_filename = 'MaskAwareNet_best.pth'
            torch.save(model.state_dict(), os.path.join(opt.model_dir, model_filename))  # save best modelformer

        #—————————————————————————— save current modelformer and checkpoint————————————————————
        # save modelformer and checkpoint
        save_dict = {'epoch': epoch + 1,
                     'step': step + 1,
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'training_params': training_params,
                     'opt': opt}
        torch.save(save_dict, os.path.join(opt.log_dir, 'ckpt.pth'))  # save current modelformer
        # —————————————————————————— save 5 modelformer and checkpoint————————————————————
        if (epoch + 1) % opt.save_every_epochs == 0:
            torch.save(save_dict, os.path.join(opt.log_dir, 'ckpt_e{}.pth'.format(epoch + 1)))

        # # —————————————————————————— save 20 model————————————————————
        if (epoch + 1) % opt.save_every_epochs == 0:
            torch.save(model.state_dict(), os.path.join(opt.model_dir, 'MaskAwareNet_state_%d.pth' % (epoch + 1)))  #save  model
        del save_dict
        writer.add_scalar('MSE_epoch', mse_per_epoch[phase], epoch + 1)
        writer.add_scalar('val PSNR epoch', psnr_per_epoch[phase], epoch + 1)
        writer.add_scalar('val SSIM epoch', ssim_per_epoch[phase], epoch + 1)
        writer.add_scalar('Learning rate', current_lr, epoch + 1)
        toc = time.time()  # 获取当前时间戳
        print('This epoch take time {:.2f}'.format(toc - tic))
    writer.close()
    print('Reach the maximal epochs! Finish training')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MaskAwareNet")
    parser.add_argument("--data_path", type=str, default=r"/home/lthpc/code/wcw/shuju/360", help='txt path to training data')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--batchSize', type=int, default=2, help='input batch size')
    parser.add_argument('--patchSize', type=int, default=416, help='the height / width of the input image to network')
    parser.add_argument('--epochs', type=int, default=100, help='total number of training epochs')
    parser.add_argument('--batchnum', type=int, default=1000, help='the number of batch')
    parser.add_argument("--milestone", type=int, default=[40, 80, 120, 160], nargs='+',
                        help="When to decay learning rate")
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
    parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
    parser.add_argument('--log_dir', default='logs/', help='tensorboard logs')
    parser.add_argument('--model_dir', default='models/', help='saving model')
    parser.add_argument('--manualSeed', type=int, default=99,help='manual seed')
    ############################################################################
    parser.add_argument('--train_ps', type=int, default=416, help='patch size of training sample')
    parser.add_argument('--embed_dim', type=int, default=32, help='dim of emdeding features')
    parser.add_argument('--win_size', type=int, default=4, help='window size of self-attention')
    parser.add_argument('--token_projection', type=str, default='linear', help='linear/conv token projection')
    parser.add_argument('--token_mlp', type=str, default='leff', help='ffn/leff token mlp')
    ############################################################################
    parser.add_argument("--save_every", type=int, default=10,
                        help="Number of training steps to log psnr ")
    parser.add_argument("--resume_training", "--r", action='store_true',
                        help="resume training from a previous checkpoint")
    parser.add_argument("--save_every_epochs", type=int, default=20,
                        help="Number of training epochs to save state")
    opt = parser.parse_args()
    main(opt)


