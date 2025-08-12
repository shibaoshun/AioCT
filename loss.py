import torch
import torch.nn as nn
from torch.nn import functional as F

torch.cuda.set_device(0)

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss


##############
class CharbonnierLoss2(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss2, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss


import torchvision
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        #vgg_pretrained_features = torchvision.models.vgg19(pretrained=True)
        vgg_pretrained_features = torchvision.models.vgg19()
        pre_file = torch.load('./utils/vgg19-dcbb9e9d.pth')
        vgg_pretrained_features.load_state_dict(pre_file)

        for param in vgg_pretrained_features.parameters():
            param.requires_grad = False
        vgg19_model_new = list(vgg_pretrained_features.features.children())
        vgg19_model_new[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.feature_extractor = nn.Sequential(*vgg19_model_new)
        vgg_pretrained_features = self.feature_extractor
        #vgg_pretrained_features.conv1 = nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        # self.criterion = nn.L1Loss()
        self.criterion = nn.L1Loss(reduction='sum')
        self.criterion2 = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            # print(x_vgg[i].shape, y_vgg[i].shape)
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

    # def forward2(self, x, y):
    #     x_vgg, y_vgg = self.vgg(x), self.vgg(y)
    #     loss = 0
    #     for i in range(len(x_vgg)):
    #         # print(x_vgg[i].shape, y_vgg[i].shape)
    #         loss += self.weights[i] * self.criterion2(x_vgg[i], y_vgg[i].detach())
    #     return loss


class airnet_uncertainty_loss(nn.Module):
    def __init__(self):
        super(airnet_uncertainty_loss, self).__init__()

    def forward(self, restored, clean_patch_1, de_id, un):
        pred_device = restored.device
        uncertainty_bn_dict = [torch.tensor(0, device=pred_device) for _ in range(3)]

        # 初始化每个类别的列表
        category_lists = {
            0: ([], [], []),  # denoise_15
            1: ([], [], []),  # denoise_25
            2: ([], [], [])  # denoise_50
        }

        # 将数据分类到相应的类别列表中
        for p, t, y, idx in zip(restored, clean_patch_1, un, de_id):
            category_lists[idx.item()][0].append(t)
            category_lists[idx.item()][1].append(p)
            category_lists[idx.item()][2].append(y)

        total_loss = 0
        un_hq_list = []
        un_lq_list = []
        num = 0
        eps = 1e-6
        # 计算每个类别的损失
        for idx in range(3):
            lq_list, list_, un_list = category_lists[idx]
            if lq_list:

                un_map = torch.mean(torch.stack(un_list), dim=0) + eps
                un_num = torch.mean(un_map)
                s = 1.0 / un_map
                num += 1
                for u, v in zip(lq_list, list_):
                    un_hq_list.append(torch.mul(v, s))
                    un_lq_list.append(torch.mul(u, s))

                # old_loss_dict[idx] = F.l1_loss(torch.stack(lq_list), torch.stack(list_), reduction='mean')
                uncertainty_bn_dict[idx] = 2 * torch.log(un_num)
                total_loss += F.l1_loss(torch.stack(un_lq_list), torch.stack(un_hq_list), reduction='mean') + uncertainty_bn_dict[idx]
                # category_losses[idx] = F.l1_loss(torch.stack(un_lq_list), torch.stack(un_hq_list),
                #                                  reduction='mean').detach() + uncertainty_bn_dict[idx].detach()
                # uncertainty_loss_dict[idx] = F.l1_loss(torch.stack(un_lq_list), torch.stack(un_hq_list),
                #                                        reduction='mean').detach()

        # print(total_loss)
        return total_loss / num


class ContrastiveLoss(torch.nn.Module):
    """对比损失函数：拉近正样本与目标的距离，推远负样本与目标的距离"""

    def __init__(self, margin=0.5):
        """
        初始化对比损失函数

        参数:
            margin: 负样本损失的权重系数，控制推远的力度
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, sne, GT):
        """
        前向传播计算对比损失

        参数:
            sne: 包含正样本和负样本的列表，第一个元素为正样本，其余为负样本
            GT: 目标真实值

        返回:
            loss: 计算得到的对比损失值
        """
        # 确保sne至少包含一个正样本和一个负样本
        assert len(sne) >= 2, "sne必须至少包含一个正样本和一个负样本"

        # 提取正样本
        I_r_positive = sne[0]

        # 计算正样本损失（拉近与目标的距离）
        pos_loss = F.mse_loss(I_r_positive, GT)

        # 提取负样本列表
        I_r_negative_list = sne[1:]

        # 计算负样本损失（推远与目标的距离）
        total_neg_loss = 0
        for I_r_negative in I_r_negative_list:
            neg_loss = F.mse_loss(I_r_negative, GT)
            total_neg_loss += neg_loss

        # 平均负样本损失
        avg_neg_loss = total_neg_loss / len(I_r_negative_list)

        # 计算最终损失
        loss = -avg_neg_loss

        return loss