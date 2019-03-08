# -*-coding:utf-8-*-

import h5py
import os
import cv2
import random
from mpi4py import MPI


def write_data(path_hdf5, filelist, labelpath, imagepath):
    """"""
    key = 0
    for i, x in enumerate(filelist):
        if i % 800 == 0:
            filename = os.path.join(path_hdf5, 'train_' + str(int(i / 800)).zfill(3) + '.hdf5')
            key = 0
        with h5py.File(filename, 'a') as infile:
            # MPI atomic mode.
            # infile.atomic = True
            x = os.path.normpath(x).strip()
            img_name = os.path.join(imagepath, x)
            print(img_name)
            if not os.path.isfile(img_name) or not os.path.exists(img_name):
                continue
            root, _ = os.path.splitext(x)
            labelfile = os.path.join(labelpath, root + '.txt')
            if os.path.exists(labelfile):
                labeldata = None
                with open(labelfile, 'r') as f:
                    labeldata = f.readlines()
                    f.close()
                rgb = cv2.imread(img_name, cv2.IMREAD_COLOR)
                for y in range(len(labeldata)):
                    content = labeldata[y].strip().split(';')
                    if len(content) != 7:
                        continue
                    fname = content[0]
                    character = content[3]
                    c = content[5].split(',')
                    label = [(int(c[0]), int(c[1]))]
                    classes = int(content[-1])
                    if classes == 6:
                        continue

                    vertex = content[4].split(',')
                    xy = list()
                    for v in vertex:
                        if v == 'NaN' or int(v) < 0:
                            continue
                        xy.append(int(v))
                    if len(xy) == 4:
                        x1 = xy[0]
                        y1 = xy[1]
                        x2 = xy[2]
                        y2 = xy[3]
                        cell_img = rgb[y1:y2, x1:x2]
                        nr, nc = cell_img.shape[0:2]
                        if nr > 0 and nc > 0:
                            cell_img = cv2.resize(cell_img, (64, 64), cv2.INTER_LINEAR)
                            serialized = {'fname': fname, 'id': y, 'char': character,
                                          'label': label, 'image': cell_img, 'class': classes}
                            grp = infile.create_group(str(key))
                            for _, k in enumerate(serialized):
                                grp[k] = serialized[k]
                            key += 1


def read_data(path_hdf5):
    """"""
    with h5py.File(path_hdf5, 'r') as infile:
        print(len(infile.keys()))
        print(infile[str(0)]['fname'].value)
        # for k, dset in infile.items():
        #     print(dset['fname'].value, dset['class'].value)
            # img = dset['image'].value
            # cv2.imwrite('test.jpg', img)
            # if int(k) == 0:
            #     break


def main():
    """"""
    path_hdf5 = '/data/chaisheng/dataset/hdf5/'
    filelist = '/data/chaisheng/dataset/images/flist'
    imagepath = '/data/chaisheng/dataset/images/'
    labelpath = '/data/chaisheng/dataset/txt/'
    with open(filelist, 'r') as infile:
        flist = infile.readlines()
        infile.close()

    random.seed(0)
    random.shuffle(flist)
    # write_data(path_hdf5, flist, labelpath, imagepath)
    read_data(path_hdf5 + 'train_013.hdf5')


if __name__ == '__main__':
    main()
