import torch
import torch.nn as nn
from layers import GCN, AvgReadout, Discriminator_E, Discriminator_I, Discriminator_J


class SHDGI(nn.Module):
    def __init__(self, n_in, n_h, n_f, activation):
        super(SHDGI, self).__init__()
        # n_in：输入特征数    n_h：输出特征数
        self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc_E = Discriminator_E(n_h)
        self.disc_I = Discriminator_I(n_h, n_f)
        self.disc_J = Discriminator_J(n_h, n_f)

    def forward(self, x, x_r, f, f_r, adj, sparse, msk, samp_bias1, samp_bias2):
        # h_1：输入x，得到h
        h_1 = self.gcn(x, adj, sparse)
        s = self.read(h_1, msk)
        s = self.sigm(s)
        h_2 = self.gcn(x_r, adj, sparse)
        ret_E = self.disc_E(s, h_1, h_2, samp_bias1, samp_bias2)
        ret_I = self.disc_I(h_1, f, f_r, samp_bias1, samp_bias2)
        ret_J = self.disc_J(s, h_1, f, f_r, samp_bias1, samp_bias2)
        ret = [ret_E, ret_I, ret_J]
        return ret

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)
        return h_1.detach(), c.detach()
