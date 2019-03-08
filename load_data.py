# -*-coding:utf-8-*-

import io, os
import lmdb
import json
import torch
import numpy as np
import random
from PIL import ImageFilter, Image
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

from config_param import Options

import cv2
import h5py
import time
import warnings

warnings.filterwarnings("ignore")


class HDF5Dataset(Dataset):
    """
    Reading data from HDF5 file.

    """
    def __init__(self, path_hdf5, transform=None):
        """

        :param path_hdf5:
        :param transfrom:
        """
        self.data_num, self.path_hdf5 = get_data_info(path_hdf5)
        self.transform = transform

    def __len__(self):
        # Number of data items.
        return self.data_num.sum()

    def __getitem__(self, item):
        val_sum = self.data_num[0]
        pre_val = 0
        idx = np.array((-2, -2), dtype=np.int32)
        for i in range(1, len(self.data_num) + 1):
            if item < val_sum:
                idx[0] = i - 1
                idx[1] = item - pre_val
                break
            else:
                val_sum += self.data_num[i]
                pre_val += self.data_num[i - 1]

        with h5py.File(self.path_hdf5[idx[0]], 'r') as infile:
            dset = infile[str(idx[1])]
            image = dset['image'].value
            label = dset['label'].value[0][0] - 1
            classes = dset['class'].value - 1
            fname = dset['fname'].value
            if classes >= 3:
                label = -1

            if self.transform:
                image = self.transform(image)
            infile.close()

        return {'image': image, 'label': label, 'classes': classes, 'fname': fname}


class LMDBDataset(Dataset):
    """
    Reading data from HDF5 file.

    """
    def __init__(self, path_db, transform=None):
        """

        :param path_hdf5:
        :param transfrom:
        """
        self.data_num, self.path_db = get_data_info(path_db)
        self.transform = transform

    def __len__(self):
        # Number of data items.
        return self.data_num.sum()

    def __getitem__(self, item):
        val_sum = self.data_num[0]
        pre_val = 0
        idx = np.array((-2, -2), dtype=np.int32)
        for i in range(1, len(self.data_num) + 1):
            if item < val_sum:
                idx[0] = i - 1
                idx[1] = item - pre_val
                break
            else:
                val_sum += self.data_num[i]
                pre_val += self.data_num[i - 1]

        env = lmdb.open(idx[0], readonly=True, max_readers=126 * 2000)
        with env.begin() as txn:
            lmdb_data = txn.get(str(idx[1]).encode('latin-1'))
            dset = arr_deserialize2(lmdb_data.decode('latin-1'))
            image = dset['image'].value
            label = dset['label'].value[0][0] - 1
            classes = dset['class'].value - 1
            fname = dset['fname'].value
            if classes >= 3:
                label = -1

            if self.transform:
                image = self.transform(image)
            txn.close()

        return {'image': image, 'label': label, 'classes': classes, 'fname': fname}


def get_data_info(path_list, pre_read=True):
    """"""
    data_num = list()
    data_path = list()
    head, tail = os.path.split(path_list)
    with open(path_list, 'r') as f:
        content = f.readlines()
        for line in content:
            line = line.strip().split(',')
            path_hdf5 = os.path.join(head, line[0])
            if pre_read:
                data_num.append(int(line[1]))
                data_path.append(path_hdf5)
            else:
                with h5py.File(path_hdf5, 'r') as infile:
                    data_num.append(len(infile.keys()))
                    data_path.append(path_hdf5)

    return np.array(data_num), data_path


class DataArgumentDataset(Dataset):
    """Data Argument dataset."""
    def __init__(self, path_db, transform=None):
        """

        :param path_db: Path to the LMDB directory
        :param transform: Optional transform to be applied on a sample.
        """
        self.env = lmdb.open(path_db, readonly=True, max_readers=126 * 2000)
        if self.env is None:
            raise Exception('error: failed to open l')
        self.transform = transform

    def __len__(self):
        # Number of data items.
        return self.env.stat()['entries']

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            val = txn.get(str(idx).encode('latin-1'))
            info = arr_deserialize2(val.decode('latin-1'))

            image = info['arr']
            # label = info['label'][0][0] - 1
            label = info['label']
            classes = info['classes'] - 1
            if label == 0:
                label = 2
            elif label == 2:
                label = 0

            fname = info['fname']
            # try:
            #     if int(os.path.splitext(fname)[0]) > 10600:
            #         classes = -1
            # except ValueError:
            #     print('Invalid value!')
            #     print(fname)

            image = cv2.resize(image, dsize=(64, 64), interpolation=cv2.INTER_LINEAR)
            if self.transform:
                image = self.transform(image)

            return {'image': image, 'label': label, 'classes': classes, 'fname': fname}


class BlurredImage(object):
    """Blurring the image in a sample to a given size."""

    def __init__(self):
        """

        :param ksize: The window size of Gaussian filter.
        """
        self.ksize = np.random.uniform(0.1, 1.5, size=1)

    def __call__(self, image):
        image = image.filter(ImageFilter.GaussianBlur(radius=self.ksize))

        return image


def arr_deserialize(serialized):
    """

    :param serialized:
    :return:
    """
    s = json.loads(serialized)

    memfile = io.BytesIO()
    memfile.write(s['arr'].encode('latin-1'))
    memfile.seek(0)
    arr = np.load(memfile)

    return {'fname': s['fname'], 'id': s['id'], 'char': s['char'],
            'arr': arr, 'label': s['label'], 'classes': s['classes']}


def arr_deserialize2(serialized):
    """

    :param serialized:
    :return:
    """
    s = json.loads(serialized)

    memfile = io.BytesIO()
    memfile.write(s['arr'].encode('latin-1'))
    memfile.seek(0)
    arr = np.load(memfile)

    return {'fname': s['fname'], 'id': s['id'], 'arr': arr,
            'classes': s['classes'], 'label': s['label']}


def shuffle_data(dataset):
    n = len(dataset)
    idx = list(range(0, n))
    random.seed(0)
    random.shuffle(idx)
    shf_data = [dataset[i] for i in idx]

    return shf_data


def load_data(args):
    ks = np.random.randint(64, 80, size=1)
    data_transform = {
        'train': transforms.Compose([transforms.ToPILImage(),
                                     transforms.ColorJitter(brightness=0.4, contrast=0.3,
                                                            saturation=0.4, hue=0.4),
                                     BlurredImage(),
                                     # transforms.Resize((64, 64), Image.NEAREST),
                                     transforms.CenterCrop((64, 64)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])
                                     ]),
        'eval': transforms.Compose([transforms.ToPILImage(),
                                    # transforms.Resize((64, 64), Image.NEAREST),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])
                                   ])
    }

    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 12, 'pin_memory': True} if use_cuda else {}

    if args.mode == 'train':
        image_datasets = {'train': args.train_set_path,
                          'eval': args.val_set_path
                          }
        # dataset2 = DataArgumentDataset('/home/chaisheng/dataset/lmdb/test/', transform=data_transform['train'])
        # train_set = DataArgumentDataset(image_datasets['train'], transform=data_transform['train'])
        # val_set = DataArgumentDataset(image_datasets['eval'], transform=data_transform['eval'])
        train_set = HDF5Dataset(image_datasets['train'], transform=data_transform['train'])
        val_set = HDF5Dataset(image_datasets['eval'], transform=data_transform['eval'])

        # train_set = train_set + dataset2
        # train_set = shuffle_data(train_set)
        # val_set = shuffle_data(val_set)
        nr_el = len(train_set)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                                   shuffle=True, **kwargs)

        val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.test_batch_size,
                                                 shuffle=True, **kwargs)

        return train_loader, val_loader, nr_el

    elif args.mode == 'evaluation':
        dataset = DataArgumentDataset(args.test_set_path, transform=data_transform['eval'])
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.test_batch_size,
                                                  shuffle=False, **kwargs)

        return test_loader, None, None

    else:
        raise Exception('This mode is non-existence, please input "train" or "evaluation".')


def visualize(dataloader):
    """

    :param dataloader:
    :return:
    """
    import matplotlib.pyplot as plt

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(), sample_batched['label'], sample_batched['classes'])

        images_batch = sample_batched['image']
        if i_batch == 0:
            plt.figure()
            grid = utils.make_grid(images_batch)
            plt.imshow(grid.numpy().transpose((1, 2, 0)))
            plt.axis('off')
            plt.ioff()
            plt.show()

            break


def main():
    op = Options()
    args = op.parse()

    data_transform = transforms.Compose([transforms.ToPILImage(),
                                         # transforms.Resize((64, 64)),
                                         transforms.ToTensor()
                                         ])
    # dataset = DataArgumentDataset(args.train_set_path, transform=data_transform)
    dataset = HDF5Dataset('/data/chaisheng/dataset/hdf5/flist', transform=data_transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1, timeout=60)
    # data_queue = queue.Queue(maxsize=400)
    # preload_data = PreLoad(data_queue, 200, dataloader, 200)
    # preload_data.start()

    # data_iter = next(iter(dataloader))
    visualize(dataloader)


if __name__ == '__main__':
    main()