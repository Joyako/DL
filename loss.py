# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes, device, feat_dim, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.device = device

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(device))
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss


class FocalLoss(nn.Module):
    """Focal Loss.
    Reference:
        Focal Loss for Dense Object Detection.
    The loss can be described as:
        Loss(x, class) = -alpha * (1 - softmax(x)[class])^gamma / log(softmax(x)[class])
        softmax = exp(x[class])}/ sum(exp(x[j])
    The losses are averaged across observations for each minibatch.

    Args:
        alpha(1D Tensor, Variable) : the scalar factor for this criterion
        gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                putting more focus on hard, misclassiﬁed examples
        size_average(bool): By default, the losses are averaged over observations for each minibatch.
                            However, if the field size_average is set to False, the losses are
                            instead summed for each minibatch.

    """

    def __init__(self, num_classes, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target, weight=None):
        if self.alpha is None:
            self.alpha = torch.Tensor(self.num_classes, 1).fill_(0.25)
        if weight is None:
            weight = torch.ones(self.num_classes, 1)

        N, C = input.size()
        P = F.softmax(input)

        class_mask = input.data.new(N, C).fill_(0)
        # class_mask = Variable(class_mask)
        ids = target.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if input.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda(1)
        alpha = self.alpha[ids.data.view(-1)]
        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()

        batch_loss = - alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class QuadraticKappaLoss(nn.Module):
    """
    Quadratic Kappa Loss.
    Reference:
          http://jeffreydf.github.io/diabetic-retinopathy-detection/
          https://www.kaggle.com/c/diabetic-retinopathy-detection#evaluation
    """
    def __init__(self, num_classes):
        super(QuadraticKappaLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, output, target, weights=None, y_pow=1, eps=1e-10):
        """

        :param output:
        :param target:
        :return:
        """
        # Converting output and target to one-hot encoded.
        M, N = output.size()
        t = torch.zeros(M, self.num_classes)
        target = t.scatter_(1, torch.LongTensor(target.reshape(-1, 1).cpu()), 1).cuda(1)

        if weights is None:
            ratings_mat = torch.Tensor.repeat(torch.arange(0, N)[:, None], (1, N)).cuda(1)
            ratings_squared = (ratings_mat - ratings_mat.t()) ** 2
            weights = ratings_squared.float() / (N - 1) ** 2
        output = output ** y_pow
        output_norm = output / (eps + output.sum(dim=1)[:, None])

        # The histograms of the raters.
        hist_rater_a = output_norm.sum(dim=0)
        hist_rater_b = target.sum(dim=0)
        # The confusion matrix.
        conf_mat = output_norm.t().mm(target)

        # The nominator.
        nom = torch.sum(weights * conf_mat)
        expected_probs = torch.mm(hist_rater_a[:, None],
                                  hist_rater_b[None, :])
        # The denominator.
        denom = torch.sum(weights * expected_probs / M)

        return -(1 - nom / denom)


if __name__ == '__main__':
    alpha = torch.rand(21, 1)
    print(alpha)
    FL = FocalLoss(num_classes=5, alpha=alpha, gamma=0)
    CE = nn.CrossEntropyLoss()
    N = 4
    C = 5
    inputs = torch.rand(N, C)
    targets = torch.LongTensor(N).random_(C)
    inputs_fl = inputs.clone()
    targets_fl = targets.clone()

    inputs_ce = inputs.clone()
    targets_ce = targets.clone()
    print('----inputs----')
    print(inputs)
    print('---target-----')
    print(targets)

    fl_loss = FL(inputs_fl, targets_fl)
    ce_loss = CE(inputs_ce, targets_ce)
    print('ce = {}, fl ={}'.format(ce_loss.data[0], fl_loss.data[0]))
    fl_loss.backward()
    ce_loss.backward()
    #print(inputs_fl.grad.data)
    print(inputs_ce.grad.data)



