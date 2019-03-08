# coding=utf-8

import cv2
import numpy as np
import json
import io
import lmdb
import os
from skimage.feature import local_binary_pattern
from skimage.filters import threshold_sauvola
from skimage.feature import greycomatrix, greycoprops
from sklearn.metrics.cluster import entropy
from skimage.feature import hog
import random
from scipy import ndimage as ndi
import adadoc


def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats


def extract_feature(arr):
    radius = 1
    n_points = radius * 8

    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

    distances = [1, 5]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = greycomatrix(arr, distances=distances, angles=angles, levels=256,
                        symmetric=False, normed=False)
    # properties = ['dissimilarity', 'homogeneity', 'contrast', 'ASM', 'energy', 'correlation']
    # glcm_feats = np.hstack([greycoprops(glcm, prop=prop).ravel() for prop in properties])
    glcm_feats = np.hstack([adadoc.greycoprops(glcm[:, :, i, :]) for i in range(0, 2)]).ravel()

    hog_feats = hog(arr, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm='L2-Hys',
                    feature_vector=True)

    ent = entropy(arr)

    # # prepare filter bank kernels
    # kernels = []
    # for theta in range(4):
    #     theta = theta / 4. * np.pi
    #     for sigma in (1, 3):
    #         for frequency in (0.05, 0.25):
    #             kernel = np.real(gabor_kernel(frequency, theta=theta,
    #                                           sigma_x=sigma, sigma_y=sigma))
    #             kernels.append(kernel)
    # gabor_feat = compute_feats(arr, kernels).ravel()

    thresh_sauvola = threshold_sauvola(arr, window_size=31, k=0.2)
    arr = arr > thresh_sauvola
    arr = (255 - arr*255).astype('uint8')

    # arr = adadoc.adath(arr, method=adadoc.ADATH_SAUVOLA | adadoc.ADATH_INVTHRESH,
    #                    xblock=21, yblock=21, k=0.2, dR=64, C=0)

    lbp_code = local_binary_pattern(arr, n_points, radius, 'uniform')

    # n_bins = int(lbp_code.max() + 1)
    n_bins = 16
    lbp_feats, _ = np.histogram(lbp_code, normed=True, bins=n_bins, range=(0, n_bins))

    data_feat = np.hstack([lbp_feats, ent, glcm_feats, hog_feats])

    return data_feat


def arr_serialize(file_name, id, arr, classes, label):
    """

    :param arr:
    :param label:
    :param id:
    :param img_file:
    :return:
    """
    memfile = io.BytesIO()
    np.save(memfile, arr)
    memfile.seek(0)

    serialized = json.dumps({'fname': file_name, 'id': id, 'arr': memfile.read().decode('latin-1'),
                             'classes': classes, 'label': label})

    return serialized


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

    return {'fname': s['fname'], 'id': id,'arr': arr,
            'classes': s['classes'], 'label': s['label']}


def write_lmdb(path_db, flist_names):
    """

    :param path_db:
    :param flist_names:
    :return:
    """
    env = lmdb.open(path_db, subdir=True, map_size=(1 << 42), create=True, map_async=True, writemap=True)

    for i, x in enumerate(flist_names):
        x = os.path.normpath(x)
        print(x)
        if os.path.isfile(x) and os.path.exists(x):
            head, tail = os.path.split(x)
            # fname = tail
            # ideograph = head.split(os.path.sep)[-1]
            root, ext = os.path.splitext(tail)
            info = root.split('_')
            label = int(info[1])
            id = int(info[-1])
            classes = 1

            # opencv cannot accept Chinese path string
            rgb = cv2.imdecode(np.fromfile(x, dtype=np.uint8), cv2.IMREAD_COLOR)
            if rgb is None:
                raise Exception('image non-existence')
            # nr, nc = np.shape(rgb)[0:2]
            # buf = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
            # rgb = cv2.imdecode(buf[1], cv2.IMREAD_COLOR).reshape(nr, nc, 3)
            rgb = cv2.resize(rgb, dsize=(64, 64), interpolation=cv2.INTER_LINEAR)

            serialized = arr_serialize(tail, id, rgb, classes, label)
            with env.begin(write=True, buffers=True) as txn:
                txn.put(bytes(str(i), encoding='latin-1'), bytes(serialized, encoding='latin-1'))

    env.close()


def read_lmdb(path_db):
    env = lmdb.open(path_db, readonly=True)

    stats = env.stat()
    nr_el = stats['entries']

    mdata = np.zeros((2, 5), dtype=np.int)
    for i in range(0, nr_el):
        with env.begin() as txn:
            s = txn.get(str(i).encode('latin-1'))
            s = s.decode('latin-1')
            v = arr_deserialize(s)
            # mdata[0][v['classes'] - 1] += 1
            mdata[1][v['label']] += 1
            print(v['fname'])
    print(mdata)


if __name__ == '__main__':
    with open('/data/chaisheng/dataset/url_image/flist') as infile:
        flist = infile.readlines()

    flist_names = [os.path.join('/data/chaisheng/dataset/url_image/', x).strip() for x in flist]

    nr_el = len(flist_names)
    random.seed(0)
    random.shuffle(flist_names)
    train_set = flist_names[0:int(0.7 * nr_el)]
    val_set = flist_names[int(0.7 * nr_el):int(0.9 * nr_el)]
    test_set = flist_names[int(0.9 * nr_el):]

    # the path of store lmdb file
    path_db_train = '/home/chaisheng/dataset/lmdb/train_data/'
    path_db_val = '/home/chaisheng/dataset/lmdb/validate_data/'
    path_db_test = '/home/chaisheng/dataset/lmdb/test_data/'

    write_lmdb(path_db_train, train_set)
    write_lmdb(path_db_val, val_set)
    write_lmdb(path_db_test, test_set)

    read_lmdb(path_db_train)
