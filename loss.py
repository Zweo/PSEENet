import torch
import torch.nn.functional as F


class MMD_loss(torch.nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self,
                        source,
                        target,
                        kernel_mul=2.0,
                        kernel_num=5,
                        fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)),
                                           int(total.size(0)),
                                           int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)),
                                           int(total.size(0)),
                                           int(total.size(1)))
        L2_distance = ((total0 - total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(
                L2_distance.data) / (n_samples**2 - n_samples)
        bandwidth /= kernel_mul**(kernel_num // 2)
        bandwidth_list = [
            bandwidth * (kernel_mul**i) for i in range(kernel_num)
        ]
        kernel_val = [
            torch.exp(-L2_distance / bandwidth_temp)
            for bandwidth_temp in bandwidth_list
        ]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(source,
                                           target,
                                           kernel_mul=self.kernel_mul,
                                           kernel_num=self.kernel_num,
                                           fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss


class MutiLoss(torch.nn.Module):
    def __init__(self, los='rbf', weight=[1, 1, 1], mode='auto', ww=None):
        super(MutiLoss, self).__init__()
        self.c1 = torch.nn.CrossEntropyLoss(weight=ww)
        self.c2 = torch.nn.CrossEntropyLoss(weight=ww)
        if los in ['rbf', 'linear']:
            self.c3 = MMD_loss(los)
        elif los == 'mse':
            self.c3 = torch.nn.MSELoss()
        elif los == 'cos':
            self.c3 = torch.nn.CosineEmbeddingLoss()
        self.w1, self.w2, self.w3 = weight
        self.mode = mode
        self.L_t_1 = []
        self.L_t_2 = []

    def setWeight(self, weight):
        assert len(weight) == 3
        self.w1, self.w2, self.w3 = weight

    def forward(self,
                clf_pred,
                clf_label,
                eog_pred,
                eog_label,
                f1,
                f2,
                tar=None):
        c1 = self.c1(clf_pred, clf_label)
        c2 = self.c2(eog_pred, eog_label)
        if tar is not None:
            c3 = self.c3(f1, f2, tar)
        else:
            c3 = self.c3(f1, f2)
        if self.mode == 'auto':
            if len(self.L_t_2) > 0:
                r = [self.L_t_1[j] / self.L_t_2[j] for j in range(3)]
                weight = F.softmax(torch.FloatTensor(r), dim=0)
                self.setWeight(weight)

        loss = self.w1 * c1 + self.w2 * c2 + self.w3 * c3

        if len(self.L_t_1) > 0:
            self.L_t_2 = self.L_t_1[:]
        self.L_t_1 = [c1, c2, c3]

        return loss
