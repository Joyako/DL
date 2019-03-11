import cv2
import numpy as np
import random
import os
import time
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage.feature import greycomatrix, greycoprops
import xgboost as xgb
from sklearn.metrics.cluster import entropy
import matplotlib.pyplot as plt
import shutil
import torch.utils.data
import lmdb
from write_lmdb import arr_deserialize
from skimage.filters import threshold_sauvola
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from PIL import Image, ImageEnhance
import adadoc


class CharacterDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, lmdb_file, transform=None):
        """
        Args:
            lmdb_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.env = lmdb.open(path_db, readonly=True)
        self.transform = transform

    def __len__(self):
        stats = self.env.stat()
        return stats['entries']

    def __getitem__(self, idx):
        sample = []
        with self.env.begin() as txn:
            s = txn.get(str(idx).encode('latin-1'))
            sample = arr_deserialize(s.decode('latin-1'))

        if self.transform:
            sample = self.transform(sample)

        return sample


def extract_feature(arr):
    radius = 1
    n_points = radius * 8

    # arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

    distances = [1, 5]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = greycomatrix(arr.copy(), distances=distances, angles=angles, levels=256,
                        symmetric=False, normed=False)
    # properties = ['dissimilarity', 'homogeneity', 'contrast', 'ASM', 'energy', 'correlation']
    # glcm_feats = np.hstack([greycoprops(glcm, prop=prop).ravel() for prop in properties])
    glcm_feats = np.hstack([adadoc.greycoprops(glcm[:, :, i, :]) for i in range(0, 2)]).ravel()

    hog_feats = hog(arr, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm='L2-Hys',
                    feature_vector=True)

    ent = entropy(arr)

    arr = adadoc.adath(arr, method=adadoc.ADATH_SAUVOLA | adadoc.ADATH_INVTHRESH,
                       xblock=21, yblock=21, k=0.2, dR=64, C=0)
    # thresh_sauvola = threshold_sauvola(arr, window_size=31, k=0.2)
    # arr = arr > thresh_sauvola
    # arr = (255 - arr * 255).astype('uint8')

    lbp_code = local_binary_pattern(arr, n_points, radius, 'uniform')

    # n_bins = int(lbp_code.max() + 1)
    n_bins = 16
    lbp_feats, _ = np.histogram(lbp_code, normed=True, bins=n_bins, range=(0, n_bins))

    data_feat = np.hstack([lbp_feats, ent, glcm_feats, hog_feats])

    return data_feat


def copy_files(flist_dir, flist, dst_dir):

    for it in flist:
        first, second = os.path.split(it)
        dst_fname = first+'_'+second
        dst_fname = os.path.join(dst_dir, dst_fname)
        src_fname = os.path.join(flist_dir, it)

        shutil.copyfile(src_fname, dst_fname)


def assembly_data(data_list):
    n = len(data_list)
    data_x = np.zeros((n, 617), dtype=np.float32)
    for i in range(n):
        rdm = np.random.uniform(0.0, 1.0)
        img = cv2.cvtColor(data_list[i], cv2.COLOR_BGR2GRAY)
        if rdm > 0.5:
            ks = np.random.choice(a=[3, 5, 7, 9, 11], size=1)
            img = cv2.GaussianBlur(img, ksize=(ks, ks), sigmaX=0)
            data_x[i, :] = extract_feature(img)
        else:
            img = Image.fromarray(img.astype('uint8'))
            enhancer = ImageEnhance.Brightness(img)
            brightness_factor = random.uniform(0.3, 1.0)
            img = enhancer.enhance(brightness_factor)
            data_x[i, :] = extract_feature(np.array(img, dtype='uint8').reshape(64, 64))

    return data_x


if __name__ == '__main__':
    path_db = '/home/chaisheng/dataset/lmdb/'

    print('Starting read file ...')
    dataset = CharacterDataset(path_db)
    nr_el = len(dataset)

    nr_train = int(nr_el*.7)
    nr_val = int(nr_el*.2)
    nr_test = nr_el - nr_train - nr_val

    idx_total = list(range(0, nr_el))
    random.seed(0)
    random.shuffle(idx_total)

    idx_train = idx_total[0:nr_train]
    idx_val = idx_total[nr_train:nr_train+nr_val]
    idx_test = idx_total[nr_train+nr_val:]

    train_set = [dataset[i] for i in idx_train]
    val_set = [dataset[i] for i in idx_val]
    test_set = [dataset[i] for i in idx_test]

    mode = 'val'
    if mode == 'val':
        # X = [x['features'] for x in train_set]
        # y = [x['label'] - 1 for x in train_set]
        # Xval = [x['features'] for x in val_set]
        # yval = [x['label'] - 1 for x in val_set]

        for ii in range(10):
            idx_trn = list(range(0, nr_train))
            train_part = int(nr_train * 0.5)
            random.seed(None)
            random.shuffle(idx_trn)
            idx_part = idx_trn[0:train_part]
            idx_eval = idx_trn[train_part:train_part+2000]

            X_img = [train_set[ii]['arr'] for ii in idx_part]
            Xval_img = [train_set[ii]['arr'] for ii in idx_eval]
            X = assembly_data(X_img)
            Xval = assembly_data(Xval_img)

            y = [train_set[ii]['label'] - 1 for ii in idx_part]
            yval = [train_set[ii]['label'] - 1 for ii in idx_eval]

            nr_len = len(y)
            w = [
            nr_len / (np.array(y) == 0).sum(),
            nr_len / (np.array(y) == 1).sum(),
            nr_len / (np.array(y) == 2).sum(),
            nr_len / (np.array(y) == 3).sum(),
            nr_len / (np.array(y) == 4).sum()]
            print('w =  ', w)
            s = w[0] + w[1] + w[2] + w[3] + w[4]
            w[0] /= s
            w[1] /= s
            w[2] /= s
            w[3] /= s
            w[4] /= s
            print('ww = ', w)

            dtrain = xgb.DMatrix(X, label=y)
            dval = xgb.DMatrix(Xval, label=yval)

            print('Start training...')

            n = len(val_set)
            t1 = time.time()
            param = {}
            param['max_depth'] = 8
            param['eta'] = 0.6
            param['silent'] = 0
            param['num_class'] = 5
            param['task'] = 'train'
            param['xgb_model'] = './xgboost_classifier.model'
            # param['objective'] = 'multi:softmax'
            param['objective'] = 'multi:softprob'
            param['nthread'] = 24
            param['verbose_eval'] = 4
            param['subsamples'] = 1
            param['weight'] = w
            # param['reg_alpha'] = 0.01
            param['reg_lambda'] = 0.01
            # param['eval_metric'] = 'map'
            # param['early_stopping_rounds'] = 50

            num_round = 100
            watchlist = [(dtrain, 'train'), (dval, 'test')]
            bst = xgb.train(param, dtrain, num_round, watchlist)
            # bst = xgb.cv(param, dtrain, num_round, metrics='map')

            bst.save_model('xgboost_classifier.model')

            pred = bst.predict(dval)

            pred = pred.reshape((pred.shape[0], -1))
            # # confusion matrix
            # cnf_matrix = confusion_matrix(yval, pred)
            # print('confusion matrix')
            # print(cnf_matrix)

            tab = [[1, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 1]]
            y_true = []
            for i in yval:
                y_true.append(tab[i])
            y_true = np.array(y_true)

            # for each class
            n_classes = 5
            precision = dict()
            recall = dict()
            average_precision = dict()
            for i in range(n_classes):
                precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], pred[:, i])
                average_precision[i] = average_precision_score(y_true[:, i], pred[:, i])
            precision["micro"], recall["micro"], _ = precision_recall_curve(y_true.ravel(), pred.ravel())

            average_precision["micro"] = average_precision_score(y_true, pred, average="samples")
            print('average precison ', average_precision)
     
    elif mode == 'test':
        Xtest = [x['features'] for x in test_set]
        ytest = [x['label'] - 1 for x in test_set]
        dtest = xgb.DMatrix(Xtest, label=ytest)

        param = {}
        param['max_depth'] = 8
        param['eta'] = 1
        param['silent'] = 0
        param['num_class'] = 5
        # param['objective'] = 'multi:softmax'
        param['objective'] = 'multi:softprob'
        param['nthread'] = 7
        param['verbose_eval'] = 4
        param['subsamples'] = 0.1
        param['weight'] = [0.18902786396558632, 0.066777947985807509, 0.20873566388424616, 0.4289161662748544, 0.10654235788950564]
        # param['reg_alpha'] = 0.01
        param['reg_lambda'] = 0.01
        model_file = 'xgboost_classifier.model'
        bst = xgb.Booster(params=param, model_file=model_file)

        pred = bst.predict(dtest)

        tab = [[1, 0, 0, 0, 0],
               [0, 1, 0, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 0, 1, 0],
               [0, 0, 0, 0, 1]]
        y_true = []
        for i in ytest:
            y_true.append(tab[i])
        y_true = np.array(y_true)

        # for each class
        n_classes = 5
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], pred[:, i])
            average_precision[i] = average_precision_score(y_true[:, i], pred[:, i])
        precision["micro"], recall["micro"], _ = precision_recall_curve(y_true.ravel(), pred.ravel())

        average_precision["micro"] = average_precision_score(y_true, pred, average="micro")
        print('average precison ', average_precision)
