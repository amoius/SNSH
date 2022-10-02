import torch
import torch.nn as nn


class Discriminator_E(nn.Module):
    def __init__(self, n_h):
        super(Discriminator_E, self).__init__()
        # first_input_size\second_input_size\output_size
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, s, h_pl, h_mi, s_bias1=None, s_bias2=None):
        s_x = torch.unsqueeze(s, 1)
        s_x = s_x.expand_as(h_pl)
        # print(h_pl.shape)
        # print(s_x.shape)
        # print(s.shape)
        a = self.f_k(h_pl, s_x)
        # print(a.shape)
        sc_1 = torch.squeeze(a, 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, s_x), 2)
        # print(sc_1.shape)
        # print('in')
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits


class Discriminator_I(nn.Module):
    def __init__(self, n_h, n_f):
        super(Discriminator_I, self).__init__()
        # first_input_size\second_input_size\output_size
        self.f_k = nn.Bilinear(n_h, n_f, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # def forward(self, f, h_pl, h_mi, s_bias1=None, s_bias2=None):
    def forward(self, h, f, f_r, s_bias1=None, s_bias2=None):

        sc_1 = torch.squeeze(self.f_k(h, f), 2)
        sc_2 = torch.squeeze(self.f_k(h, f_r), 2)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits


class Discriminator_J(nn.Module):
    def __init__(self, n_h, n_f):
        super(Discriminator_J, self).__init__()
        # first_input_size\second_input_size\output_size
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        self.w_f = nn.Linear(n_f, n_h)
        self.w_s = nn.Linear(n_h, n_h)
        self.w_z = nn.Linear(n_h * 2, n_h)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, s, h, f_pl, f_mi, s_bias1=None, s_bias2=None):
        # s:[1, d], h:[1, n, d], f:[1, n, d_f]
        s_x = torch.unsqueeze(s, 1)
        s_x = s_x.expand_as(h)
        zf_1 = self.w_f(f_pl)
        zs = self.w_s(s_x)
        z_1 = self.w_z(torch.cat((zf_1, zs), 2))
        sc_1 = torch.squeeze(self.f_k(h, z_1), 2)

        zf_2 = self.w_f(f_mi)
        z_2 = self.w_z(torch.cat((zf_2, zs), 2))
        sc_2 = torch.squeeze(self.f_k(h, z_2), 2)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits
