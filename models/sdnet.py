import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicDownsampling(torch.nn.Module):
    # dynamic filtering without downsampling
    def __init__(self, kernel_size):
        super(DynamicDownsampling, self).__init__()
        self.kernel_size = kernel_size

    def kernel_normalize(self, kernel):
        # kernel: [B, H, W, k*k]
        return F.softmax(kernel, dim=-1)

    def forward(self, x, kernel):
        # x: [B, C, H, W]
        # kernel: [B, k*k, H, W]
        # return: [B, C, H, W]

        b, _, h, w = kernel.shape
        kernel = kernel.permute(0, 2, 3, 1).contiguous()  # [B, H, W, k*k]
        kernel = self.kernel_normalize(kernel)
        kernel2 = kernel.permute(0, 3, 1, 2)

        kernel = kernel.unsqueeze(dim=1)  # [B, 1, H, W, k*k]

        num_pad = (self.kernel_size - 1) // 2
        x = F.pad(x, (num_pad, num_pad, num_pad, num_pad), mode="replicate")
        x = x.unfold(2, self.kernel_size, 1)
        x = x.unfold(3, self.kernel_size, 1)  # [B, C, H, W, k, k]
        x = x.contiguous().view(b, -1, h, w, self.kernel_size * self.kernel_size)  # [B, C, H, W, k*k]
        x = x * kernel
        x = torch.sum(x, -1)  # [B, C, H, W]

        return x


class SDNet(nn.Module):
    def __init__(self):
        super(SDNet, self).__init__()
        '1'
        self.Conv_0 = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1,  dilation=1, bias=True)
        self.Conv_1 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_2 = nn.ReLU(inplace=True)
        self.Conv_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_5 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_6 = nn.ReLU(inplace=True)
        self.Conv_7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_9 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_10 = nn.ReLU(inplace=True)
        self.Conv_11 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_13 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_14 = nn.ReLU(inplace=True)
        self.Conv_15 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)


        '2'
        self.Conv_17 = torch.nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=0, dilation=1, bias=True)
        self.Conv_18 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_19 = nn.ReLU(inplace=True)
        self.Conv_20 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_22 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_23 = nn.ReLU(inplace=True)
        self.Conv_24 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_26 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_27 = nn.ReLU(inplace=True)
        self.Conv_28 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)


        self.Conv_30 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_31 = nn.ReLU(inplace=True)
        self.Conv_32 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)


        '3'
        self.Conv_34 = torch.nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=0, dilation=1, bias=True)
        self.Conv_35 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_36 = nn.ReLU(inplace=True)
        self.Conv_37 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_39 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_40 = nn.ReLU(inplace=True)
        self.Conv_41 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_43 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_44 = nn.ReLU(inplace=True)
        self.Conv_45 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_47 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_48 = nn.ReLU(inplace=True)
        self.Conv_49 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        '4'
        self.Conv_51 = torch.nn.Conv2d(256, 512,  kernel_size=2, stride=2, padding=0, dilation=1, bias=True)
        self.Conv_52 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_53 = nn.ReLU(inplace=True)
        self.Conv_54 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_56 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_57 = nn.ReLU(inplace=True)
        self.Conv_58 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_60 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_61 = nn.ReLU(inplace=True)
        self.Conv_62 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_64 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_65 = nn.ReLU(inplace=True)
        self.Conv_66 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        '5'
        self.ConvTranspose_69 =torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0, dilation=1, bias=True)
        self.Conv_70 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_71 = nn.ReLU(inplace=True)
        self.Conv_72 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_74 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_75 = nn.ReLU(inplace=True)
        self.Conv_76 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_78 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_79 = nn.ReLU(inplace=True)
        self.Conv_80 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_82 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_83 = nn.ReLU(inplace=True)
        self.Conv_84 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)


        '6'
        self.ConvTranspose_87 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, dilation=1,bias=True)
        self.Conv_88 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_89 = nn.ReLU(inplace=True)
        self.Conv_90 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_92 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_93 = nn.ReLU(inplace=True)
        self.Conv_94 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_96 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_97 = nn.ReLU(inplace=True)
        self.Conv_98 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_100 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_101 = nn.ReLU(inplace=True)
        self.Conv_102 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)


        '7'
        self.ConvTranspose_105 = torch.nn.ConvTranspose2d(128,64, kernel_size=2, stride=2, padding=0, dilation=1, bias=True)
        self.Conv_106 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_107 = nn.ReLU(inplace=True)
        self.Conv_108 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_110 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_111 = nn.ReLU(inplace=True)
        self.Conv_112 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_114 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_115 = nn.ReLU(inplace=True)
        self.Conv_116 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_118 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_119 = nn.ReLU(inplace=True)
        self.Conv_120 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_123 = torch.nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.downsampling = DynamicDownsampling(kernel_size=3)

    def forward(self, x, z):
        temp4 = self.Conv_3(self.ReLU_2(self.Conv_1(self.Conv_0(x))))
        Add_4 = temp4 + self.Conv_0(x)
        temp8 = self.Conv_7(self.ReLU_6(self.Conv_5(Add_4)))
        Add_8 = temp8 + Add_4
        temp12 = self.Conv_11(self.ReLU_10(self.Conv_9(Add_8)))
        Add_12 = temp12  + Add_8
        temp16 = self.Conv_15(self.ReLU_14(self.Conv_13(Add_12)))
        Add_16 = temp16 + Add_12

        temp21 = self.Conv_20(self.ReLU_19(self.Conv_18(self.Conv_17(Add_16))))
        Add_21 = temp21 + self.Conv_17(Add_16)
        temp25 = self.Conv_24(self.ReLU_23(self.Conv_22(Add_21)))
        Add_25 = temp25 + Add_21
        temp29 = self.Conv_28(self.ReLU_27(self.Conv_26(Add_25)))
        Add_29 = temp29 + Add_25
        temp33 = self.Conv_32(self.ReLU_31(self.Conv_30(Add_29)))
        Add_33 = temp33 + Add_29

        temp38 = self.Conv_37(self.ReLU_36(self.Conv_35(self.Conv_34(Add_33))))
        Add_38 = temp38 + self.Conv_34(Add_33)
        temp42 = self.Conv_41(self.ReLU_40(self.Conv_39(Add_38)))
        Add_42 = temp42 + Add_38
        temp46 = self.Conv_45(self.ReLU_44(self.Conv_43(Add_42)))
        Add_46 = temp46 + Add_42
        temp50 = self.Conv_49(self.ReLU_48(self.Conv_47(Add_46)))
        Add_50 = temp50 + Add_46

        temp55 = self.downsampling(self.ReLU_65(self.downsampling(self.Conv_51(Add_50), z)), z)
        Add_55 = temp55 + self.Conv_51(Add_50)
        temp59 = self.downsampling(self.ReLU_65(self.downsampling(Add_55, z)), z)
        Add_59 = temp59 + Add_55
        temp63 = self.downsampling(self.ReLU_65(self.downsampling(Add_59, z)), z)
        Add_63 = temp63 + Add_59
        output = self.downsampling(self.ReLU_65(self.downsampling(Add_63, z)), z)
        Add_67 = output + Add_63

        Add_68 = Add_67+self.Conv_51(Add_50)

        temp73 = self.Conv_72(self.ReLU_71(self.Conv_70(self.ConvTranspose_69(Add_68))))
        Add_73 = temp73 + self.ConvTranspose_69(Add_68)
        temp77 = self.Conv_76(self.ReLU_75(self.Conv_74(Add_73)))
        Add_77 = temp77 + Add_73
        temp81 = self.Conv_80(self.ReLU_79(self.Conv_78(Add_77)))
        Add_81 = temp81 + Add_77
        temp85 = self.Conv_84(self.ReLU_83(self.Conv_82(Add_81)))
        Add_85 = temp85 + Add_81

        Add_86 = Add_85 + self.Conv_34(Add_33)

        temp91 = self.Conv_90(self.ReLU_89(self.Conv_88(self.ConvTranspose_87(Add_86))))
        Add_91 = temp91 + self.ConvTranspose_87(Add_86)
        temp95 = self.Conv_94(self.ReLU_93(self.Conv_92(Add_91)))
        Add_95 = temp95 + Add_91
        temp99 = self.Conv_98(self.ReLU_97(self.Conv_96(Add_95)))
        Add_99 = temp99 + Add_95
        temp103 = self.Conv_102(self.ReLU_101(self.Conv_100(Add_99)))
        Add_103 = temp103 + Add_99

        Add_104 = Add_103 + self.Conv_17(Add_16)

        temp109 = self.Conv_108(self.ReLU_107(self.Conv_106(self.ConvTranspose_105(Add_104))))
        Add_109 = temp109 + self.ConvTranspose_105(Add_104)
        temp113 = self.Conv_112(self.ReLU_111(self.Conv_110(Add_109)))
        Add_113 = temp113 + Add_109
        temp117 = self.Conv_116(self.ReLU_115(self.Conv_114(Add_113)))
        Add_117 = temp117 + Add_113
        temp121 = self.Conv_120(self.ReLU_119(self.Conv_118(Add_117)))
        Add_121 = temp121 + Add_117


        Add_122 = Add_121 + self.Conv_0(x)
        out = self.Conv_123(Add_122)

        return out

# class PromptNet(nn.Module):
#     def __init__(self):
#         super(PromptNet, self).__init__()
#         self.Conv_0 = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
#         self.Conv_1 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
#         self.Conv_3 = torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
#         # Compute the flattened size
#         self.flatten_size = 64 * 416 * 416
#         # Add a fully connected layer
#         self.fc = nn.Linear(self.flatten_size, 1)
#
#     def forward(self, x):
#         c1 = self.Conv_0(x)
#         c2 = self.Conv_1(c1)
#         c3 = self.Conv_3(c2)
#         # Flatten the output of the last convolutional layer
#         flat = torch.flatten(c3, start_dim=1)
#         # Pass through fully connected layer
#         fc_out = self.fc(flat)
#         # Pass through linear layer
#         fc_out_reshaped = torch.reshape(fc_out, (1, 1, 1, 1))
#         y=x*fc_out_reshaped+fc_out_reshaped
#         return y
#

class F_ext(nn.Module):
    def __init__(self, in_nc=2, nf=64):
        super(F_ext, self).__init__()
        stride = 2
        pad = 0
        self.pad = nn.ZeroPad2d(1)
        self.conv1 = nn.Conv2d(in_nc, nf, 7, stride, pad, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, stride, pad, bias=True)
        self.conv3 = nn.Conv2d(nf, nf, 3, stride, pad, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1_out = self.act(self.conv1(self.pad(x)))
        conv2_out = self.act(self.conv2(self.pad(conv1_out)))
        conv3_out = self.act(self.conv3(self.pad(conv2_out)))
        out = torch.mean(conv3_out, dim=[2, 3], keepdim=False)

        return out

class PromptNet(nn.Module):
    def __init__(self, in_nc=1, nf=64):
        super(PromptNet, self).__init__()
        self.F_ext_net1 = F_ext(in_nc, nf)
        self.prompt_scale1 = nn.Linear(nf, 256, bias=True)  # Adjusted output size
        self.prompt_shift1 = nn.Linear(nf, 256, bias=True)  # Adjusted output size

    def forward(self, x, y):
        out_dec_level1 = x
        prompt1 = self.F_ext_net1(x)
        scale1 = self.prompt_scale1(prompt1)
        shift1 = self.prompt_shift1(prompt1)
        out_dec_level1 = y * scale1.view(-1, 256, 1, 1) + shift1.view(-1, 256, 1, 1) + y
        return out_dec_level1