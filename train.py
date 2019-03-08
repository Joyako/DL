# -*-coding:utf-8-*-

import gc
import time
import numpy as np
import torch
import shutil
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
import torchvision.models as models

from config_param import Options
from load_data import load_data
from loss import CenterLoss
from loss import FocalLoss
from loss import QuadraticKappaLoss

import torch.utils.model_zoo as model_zoo
import math


def vgg(args):
    # Load a pretrained model and reset final fully connected layer.
    # model = models.vgg11_bn(pretrained=True)
    model = models.vgg11_bn(pretrained=True)

    model.classifier = nn.Sequential(
        nn.Linear(512 * 2 * 2, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 2048),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(2048, args.num_classes),
    )
    if args.pretrained:
        model.load_state_dict(torch.load(args.load_model_path))

    return model


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)

        return x, y


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet(args):
    """"""
    model = resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.avgpool = nn.AvgPool2d(2, stride=1)
    model.fc = nn.Linear(num_ftrs, args.num_classes)
    if args.pretrained:
        model.load_state_dict(torch.load(args.load_model_path))

    return model


def train(args, model, device, train_loader, val_loader, optimizer, epoch, writer,
          iter_count, criterion, optimizer_centerloss=None):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # weights = torch.tensor([[0., 0.0625, 0.25, 0.5625, 1.],
    #                         [0.50, 0., 0.0625, 0.25, 0.5625],
    #                         [0.50, 0.0625, 0., 0.0625, 0.25],
    #                         [0.5625, 0.25, 0.0625, 0., 0.0625],
    #                         [1., 0.5625, 0.25, 0.0625, 0.]], dtype=torch.float32).cuda(1)
    weights = torch.tensor([[0., 0.50, 1.],
                            [0.55, 0., 0.25],
                            [1., 0.25, 0.]], dtype=torch.float32).to(device)

    acc = 0.0
    # criterion = nn.CrossEntropyLoss(weight=args.weight)
    file_label = ['validation/loss', 'validation/accuracy', 'validation/acc0',
                  'validation2/loss', 'validation2/accuracy', 'validation2/acc0']
    model_path = args.save_model_path + str(epoch) + '_'
    end = time.time()
    for batch_idx, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        optimizer.zero_grad()
        # optimizer_centerloss.zero_grad()

        img, target = data['image'].to(device), data['label'].to(device)
        x, output = model(img)
        # cont_kappa = quadratic_kappa(args, output, target, device, 2)
        loss = F.cross_entropy(output, target, weight=args.weight, size_average=True, ignore_index=-1)
        # loss += criterion(output, target, y_pow=2, weights=weights)
        # loss += criterion(x, target) * 0.0001
        # back propagation
        loss.backward()
        # update gradient
        if args.use_multi_gpu:
            optimizer.module.step()
        else:
            optimizer.step()
        # optimizer_centerloss.step()

        if args.is_print_grad:
            with open('grad.txt', 'a') as f:
                for param in model.parameters():
                    if param.requires_grad:
                        f.write(str(param.grad))
                f.write('\n')
                f.write('************************************************************************')
                f.write('\n')

        writer.add_scalar('train/loss', loss.item(), iter_count)
        # writer.add_graph(model, output)

        # measure accuracy and record loss
        losses.update(loss.item(), img.size(0))
        acc = accuracy(output, target)[0].item()
        top1.update(acc, img.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % (args.log_interval * 2) == 0:
            print('Train Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, batch_idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))

            writer.add_scalar('train/accuracy', acc, global_step=iter_count)

            # save model
            if batch_idx % (args.log_interval * 30) == 0 and batch_idx != 0:
                validate(args, model, device, val_loader, iter_count, writer, file_label[0:3], criterion)
                print(model_path + str(batch_idx) + '.pth')
                torch.save(model.state_dict(), model_path + str(batch_idx) + '.pth')

        iter_count += 1


def validate(args, model, device, val_loader, iter_num, writer, fname, criterion):
    # switch to evaluate mode
    model.eval()
    val_loss = AverageMeter()
    batch_time = AverageMeter()
    top1 = AverageMeter()

    weights = torch.tensor([[0., 0.50, 1.],
                            [0.55, 0., 0.25],
                            [1., 0.25, 0.]], dtype=torch.float32).to(device)
    with torch.no_grad():
        end = time.time()
        CM = torch.zeros(args.num_classes, args.num_classes, dtype=torch.float32)
        for batch_idx, data in enumerate(val_loader):
            img, target = data['image'].to(device), data['label'].to(device)
            x, output = model(img)
            # cont_kappa = quadratic_kappa(args, output, target, device, 2)
            loss = F.cross_entropy(output, target, reduction='sum',
                                   weight=args.weight, size_average=True, ignore_index=-1)
            # loss += criterion(output, target, y_pow=2, weights=weights)

            # measure accuracy and record loss
            acc = accuracy(output, target)
            val_loss.update(loss.item(), img.size(0))
            top1.update(acc[0].item(), img.size(0))

            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            cm = confusion_matrix(target, pred, args.num_classes)
            CM += cm
        print('\nConfusion matrix :')
        print(CM)
        print(CM / CM.sum(dim=0))
        acc_0 = CM[0][0] / CM.sum(dim=0)[0]

        print('Validate: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            batch_idx, len(val_loader), batch_time=batch_time, loss=val_loss, top1=top1))
        writer.add_scalar(fname[0], val_loss.val, iter_num)
        writer.add_scalar(fname[1], top1.val, iter_num)
        writer.add_scalar(fname[2], acc_0, iter_num)


def test(args, model, device, test_loader):
    model.load_state_dict(torch.load(args.test_model_path))
    model.eval()
    y_pred = []
    y_prab = []
    y_true = []
    with torch.no_grad():
        CM = torch.zeros(args.num_classes, args.num_classes, dtype=torch.float32)
        for batch_idx, data in enumerate(test_loader):
            # if args.use_multi_label:
            #     img, target1, target2 = data['image'].to(device), data['label'].to(device), data['ft'].to(device)
            # else:
            img, target = data['image'].to(device), data['label'].to(device)
            x, output = model(img)
            output = F.softmax(output)
            for op in np.array(output):
                y_prab.append(op.tolist())
            for t in np.array(target):
                y_true.append(t)
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            if 0:
                fname = data['fname']
                for i in range(pred.size()[0]):
                    if pred[i].item() == 0 and target[i].item() == 1:
                        shutil.copy('/data/chaisheng/dataset/url_image/' + fname[i], '/data/chaisheng/dataset/class1_1/')
            cm = confusion_matrix(target, pred, args.num_classes)
            CM += cm
            for p in pred:
                y_pred.append(p.item())
        print('\nConfusion matrix :')
        print(CM)
        print('Precision:')
        print(CM / CM.sum(dim=0))
        print('Recall:')
        recall = CM.t() / CM.sum(dim=1)
        print(recall.t())
    return y_pred, y_prab, y_true


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, targets, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    with torch.no_grad():
        batch_size = targets.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


def confusion_matrix(y_true, y_pred, num_classes):
    """

    :param y_true:
    :param y_pred:
    :return:
    """

    CM = torch.zeros(num_classes, num_classes, dtype=torch.float32)
    for i in range(len(y_true)):
        x = y_pred[i]
        y = y_true[i]
        if y >= num_classes or y < 0:
            continue
        CM[y][x] += 1

    return CM


def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    import  math

    mu = [0.5, 0.5, 0.55, 0.6, 0.99]
    m = epoch // 5
    k = 2
    # Exponential decay.has the mathematical l = l0 * eps(-k * t),
    # where k are hyperparameters and t is the epochs number
    lr = args.lr * (0.1 ** m)
    # lr = args.lr * math.exp(-k * epoch)
    # 1/t decay. has the mathematical form l = l0 / (1 + k* t),
    # where k are hyperparameters and t is the epochs number
    # lr = args.lr / (1 + k * epoch)
    if args.use_multi_gpu:
        for param_group in optimizer.module.param_groups:
            param_group['lr'] = lr
            # param_group['momentum'] = mu[m]
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            # param_group['momentum'] = mu[m]


def plot_precision_recall(args, y_scores, y_true):
    """

    :param y_test:
    :param y_score:
    :return:
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from itertools import cycle
    from sklearn.metrics import average_precision_score
    from sklearn.metrics import precision_recall_curve

    y_scores = np.array(y_scores, dtype=np.float32).reshape(-1, args.num_classes)

    y_true = np.array(y_true)
    y_scores = y_scores[y_true != 4]

    y_true = y_true[y_true != 4]
    y_test = np.eye(args.num_classes)[y_true]

    precision = dict()
    recall = dict()
    thresh = dict()
    average_precision = dict()
    for i in range(args.num_classes):
        precision[i], recall[i], thresh[i] = precision_recall_curve(y_test[:, i], y_scores[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_scores[:, i])

    for i in range(len(recall[0])):
        if recall[0][i] >= 0.2:
            print(thresh[0][i], precision[0][i], recall[0][i])

    plt.figure(figsize=(7, 7))

    lines = []
    labels = []
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    for i, color in zip(range(args.num_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('class {0} (area = {1:0.2f})'
                      ''.format(i, average_precision[i]))
        if i == 0:
            break

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(lines, labels, loc=(0.01, 0.01), prop=dict(size=14))
    plt.savefig('fig.jpg')
    plt.show()


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


def visualize_stn(test_loader, model, device):
    import matplotlib.pyplot as plt

    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(test_loader))['image'].to(device)

        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data).cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')

    plt.ioff()
    plt.show()


def visualize(args, y_pred, y_true, fpath):
    import cv2
    import os

    flist = None
    with open(fpath['image'] + 'flist', 'r') as f:
        flist = f.readlines()
        f.close()
    flist = flist[0:400]
    j = 0
    for i in range(len(flist)):
        l = flist[i].strip()
        print(l)
        head, tail = os.path.split(l)
        root, ext = os.path.splitext(tail)
        rgb = cv2.imread(fpath['image'] + l, cv2.IMREAD_COLOR)
        if rgb is None:
            raise Exception('image non-existence')

        if os.path.exists(fpath['txt'] + root + '.txt'):
            txt_list = None
            with open(fpath['txt'] + root + '.txt', 'r') as f:
                txt_list = f.readlines()
                f.close()

            for t in range(len(txt_list)):
                content = txt_list[t].split(';')
                vertex = content[4].split(',')
                x1 = int(vertex[0])
                y1 = int(vertex[1])
                x2 = int(vertex[2])
                y2 = int(vertex[3])
                if x1 > 0 and x2 > 0 and y1 > 0 and y2 > 0:
                    if args.use_multi_label:
                        cv2.putText(rgb, str(y_pred[j]) + '_' + str(y_true[j]), (x1 + 20, y1 + 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), lineType=cv2.LINE_AA)
                    else:
                        cv2.putText(rgb, str(y_pred[j]) + '_' + str(y_true[j]), (x1 + 20, y1 + 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), lineType=cv2.LINE_AA)
                    cv2.rectangle(rgb, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=2)
                    j += 1
            cv2.imwrite('./test/' + root + '.jpg', rgb)


def show_args(args):
    print('Mode : ', args.mode)
    print('Learning rate : ', args.lr)
    print('The number of classes : ',  args.num_classes)
    print('The saved path of log file : ', args.log_dir)
    print('Input data weight : ', args.weight)
    print('Batch size : ', args.batch_size)
    print('The path of save model : ', args.save_model_path)
    print('The path of train set : ', args.train_set_path)


def main():
    op = Options()
    args = op.parse()

    gc.collect()
    show_args(args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    # specify gpu id
    device = torch.device("cuda:1" if use_cuda else "cpu")
    # criterion = CenterLoss(num_classes=args.num_classes, device=device, feat_dim=512)
    # criterion = FocalLoss(args.num_classes, gamma=2)
    criterion = QuadraticKappaLoss(args.num_classes)
    if args.mode == 'train':
        torch.manual_seed(args.seed)

        # load data
        print('Loading train data...')
        train_loader, val_loader, nr_el = load_data(args)

        # model = vgg(args).to(device)
        model = resnet(args).to(device)
        if args.use_multi_gpu:
            model = nn.DataParallel(model, device_ids=[0, 1])

        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay,
                              nesterov=True)

        # optimizer_centerloss = optim.SGD(criterion.parameters(),
        #                                  lr=args.lr,
        #                                  momentum=args.momentum,
        #                                  weight_decay=args.weight_decay,
        #                                  nesterov=True)
        if args.use_multi_gpu:
            optimizer = nn.DataParallel(optimizer, device_ids=[0, 1])
        with SummaryWriter(args.log_dir) as writer:
            for epoch in range(1, args.epochs + 1):
                adjust_learning_rate(args, optimizer, epoch - 1)
                iter_count = int(np.ceil(nr_el / args.batch_size)) * (epoch - 1) + 14010
                train(args, model, device, train_loader, val_loader, optimizer, epoch, writer,
                      iter_count, criterion)

            # x = torch.autograd.Variable(torch.rand(1, 3, 224, 224))
            # writer.add_graph(model, x)
            # export scalar data to JSON for external processing
            writer.export_scalars_to_json('./all_scalars.json')
            writer.close()

    else:
        # model = vgg(args).to(device)
        model = resnet(args).to(device)
        print('Loading test data...')
        test_loader, _, _ = load_data(args)

        y_pred, y_socres, y_true = test(args, model, device, test_loader)
        # visualize(args, y_pred, y_true, fpath)
        plot_precision_recall(args, y_socres, y_true)


if __name__ == '__main__':
    main()