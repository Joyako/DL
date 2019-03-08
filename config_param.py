# -*-coding:utf-8-*-

import argparse
import torch


class Options():
    """Configure train model parameters."""
    def __init__(self):
        weight = {
            '0': None,
            '1': torch.tensor([[0.691, 0.158, 0.151]], device='cuda:1'),
            '2': torch.tensor([[0.013, 0.025*2, 0.089*2, 0.233, 0.640]], device='cuda:1'),
            '3': torch.tensor([[0.07, 0.187, 0.743]], device='cuda:0'),
            '4': torch.tensor([[0.405, 0.144, 0.451]], device='cuda:1'),
            '5': torch.tensor([[0.189, 0.067, 0.210, 0.427, 0.108]], device='cuda:1'),
            '6': torch.tensor([[0.271, 0.729]], device='cuda:0'),
            '7': torch.tensor([[0.062, 0.062*8, 0.876]], device='cuda:1')
        }

        self.parser = argparse.ArgumentParser(description='HandWriting Classifies')
        self.parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                                 help='input batch size for training (default: 64)')
        self.parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N',
                                 help='input batch size for testing (default: 100)')

        self.parser.add_argument('--train-set-path', default='/data/chaisheng/dataset/hdf5/train_list',
                                 help='path of data for training and test.')

        self.parser.add_argument('--val-set-path', default='/data/chaisheng/dataset/hdf5/val_list',
                                 help='path of data for training and test.')

        self.parser.add_argument('--save-model-path', default='/data/chaisheng/dataset/model0/',
                                 help='path of data for save model.')

        self.parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                                 help='learning rate (default: 1e-3)')

        self.parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                                 help='SGD momentum (default: 0.5)')
        self.parser.add_argument('--weight-decay', type=float, default=1e-5, metavar='W',
                                 help='SGD weight decay (default: 1e-5)')

        self.parser.add_argument('--no-cuda', action='store_true', default=False,
                                 help='disables CUDA training')
        self.parser.add_argument('--epochs', type=int, default=20, metavar='N',
                                 help='number of epochs to train (default: 20)')
        self.parser.add_argument('--seed', type=int, default=1, metavar='S',
                                 help='random seed (default: 1)')
        self.parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                                 help='how many batches to wait before logging training status')

        self.parser.add_argument('--pretrained', type=bool, default=False,
                                 help='how many batches to wait before logging training status')
        self.parser.add_argument('--load-model-path', default='/data/chaisheng/dataset/model2/11_200.pth',
                                 help='path of data for training and test.')

        self.parser.add_argument('--log-dir', default='runs/exp-3',
                                 help='path of data for save log.')
        self.parser.add_argument('--is-imshow', type=bool, default=False,
                                 help='Print the classify result  in source image.')
        self.parser.add_argument('--is-print-grad', type=bool, default=False,
                                 help='Save gradient value to text in training.')
        self.parser.add_argument('--test-model-path', default='/data/chaisheng/dataset/model2/20_200.pth',
                                 help='path of data for save model.')
        self.parser.add_argument('--test-set-path', default='/home/chaisheng/dataset/lmdb/test_data/',
                                 help='Path of data for training and test.')

        # When using multi-label, you must change parameters as follows:
        self.parser.add_argument('--use-multi-gpu', type=bool, default=False,
                                 help='When you use multi gpu, you need set True.')
        self.parser.add_argument('--weight', type=float, default=weight['1'], metavar='W',
                                 help='A manual rescaling weight given to each class')
        self.parser.add_argument('--num-classes', type=int, default=3, metavar='N',
                                 help='number of classify.')
        self.parser.add_argument('--mode', default='train',
                                 help='Network mode is "train" or "evaluation".')

    def parse(self):
        opt = self.parser.parse_args()

        # # mode
        # if opt.mode not in ["Train", "Test", "Validation", "train", "test", "validation"]:
        #     raise Exception("cannot recognize flag `mode`")

        return opt


def main():
    op = Options()
    args = op.parse()


if __name__ == "__main__":
    main()