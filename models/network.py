import torch
from models.sdnet import SDNet
import torch.nn as nn
from aiomodel import ShadowFormer
from models.TDDC import ShadowFormer1
class MaskAwareNet(nn.Module):
    def __init__(self, opt):
        super(MaskAwareNet, self).__init__()
        self.train_ps = opt.train_ps
        self.embed_dim = opt.embed_dim
        self.win_size = opt.win_size
        self.token_projection = opt.token_projection
        self.token_mlp = opt.token_mlp

        self.TDDC = ShadowFormer1(img_size=self.train_ps, embed_dim=self.embed_dim, win_size=self.win_size,
                               token_projection=self.token_projection, token_mlp=self.token_mlp, input_size=52)
        self.MF = ShadowFormer(img_size=self.train_ps, embed_dim=self.embed_dim, win_size=self.win_size, token_projection=self.token_projection, token_mlp=self.token_mlp)
        self.eanet = SDNet()

        self.var_conv = nn.Sequential(
            ShadowFormer1(img_size=self.train_ps, embed_dim=self.embed_dim, win_size=self.win_size,
                          token_projection=self.token_projection, token_mlp=self.token_mlp, input_size=416),
            nn.Conv2d(9, int(32 * 2), 3, 1, 1), nn.ELU(),
            nn.Conv2d(int(32 * 2), int(32 * 2), 3, 1, 1), nn.ELU(),
            nn.Conv2d(int(32 * 2), 1, 3, 1, 1), nn.Softplus()
        )


    def forward(self, x, state):   # ma    li和ma级联
        z1 = self.TDDC(x)
        EA = self.eanet(x, z1)
        if state:
            un = self.var_conv(EA)
        out, sne= self.MF(x, EA, state)   # 级联  估计伪影  文本
        if state:
            return out, EA, sne, un
        else:
            return out, EA, sne





