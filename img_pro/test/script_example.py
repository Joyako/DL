import cv2
import scipy
import numpy as np

import os.path

import skimage.morphology as skmorph
import adadoc


# assumption for input image: bgr and fixed with of 1000
def sepline_binarize(src):
    nr, nc = src.shape[:2]
    lab = cv2.cvtColor(src, cv2.COLOR_BGR2Lab)
    lum = lab[:,:,0]

    # enhance the gray separation lines;
    lum = adadoc.contrst_enhance(lum, "linear_gamma", 2.2, 0.3)

    # binarization
    map_bin = adadoc.adath(lum, method=adadoc.ADATH_WOLFJOLION | adadoc.ADATH_INVTHRESH,
                           xblock=51, yblock=51, k=0.05, dR=128, C=0)

    bw = map_bin.copy()
    map_bin = cc_remove(map_bin)

    return map_bin, bw


def cc_remove(src):
    """
    src must be binary map (0 or 255);
    """
    info = cv2.connectedComponentsWithStats(src, connectivity=8)
    nr_labels = info[0]
    labeled = info[1]
    stats = info[2]
    centroids = info[3]
    for idx in range(1, nr_labels):
        area = stats[idx, cv2.CC_STAT_AREA]
        h = stats[idx, cv2.CC_STAT_HEIGHT]
        w = stats[idx, cv2.CC_STAT_WIDTH]
        l = stats[idx, cv2.CC_STAT_LEFT]
        t = stats[idx, cv2.CC_STAT_TOP]
        if area < 210:
            src[labeled == idx] = 0
            continue

        # remove Chinese characters
        roi = src[t:t+h, l:l+w]
        mus = cv2.moments(roi, binaryImage=True)
        density = mus['m00'] / (h*w)
        if density > 0.15 and h*w < 100*100:
            src[labeled == idx] = 0
            continue

    # remove dash-lines remains
    info = cv2.connectedComponentsWithStats(src, connectivity=4)
    nr_labels = info[0]
    labeled = info[1]
    stats = info[2]
    centroids = info[3]
    for idx in range(1, nr_labels):
        area = stats[idx, cv2.CC_STAT_AREA]
        h = stats[idx, cv2.CC_STAT_HEIGHT]
        w = stats[idx, cv2.CC_STAT_WIDTH]
        l = stats[idx, cv2.CC_STAT_LEFT]
        t = stats[idx, cv2.CC_STAT_TOP]
        if area < 36:
            src[labeled == idx] = 0
            continue

    return src


if __name__ == '__main__':
    ifname = './0001.jpg'
    bgr = cv2.imread(ifname)
    if bgr is None:
        raise Exception('file non-existence .')

    print(ifname)

    # normalize input image size
    nr, nc = bgr.shape[:2]
    r = 1000.0 / nc
    dsiz = (1000, int(nr * r))
    bgr = cv2.resize(bgr, dsiz, interpolation=cv2.INTER_LINEAR)

    # get file name id
    bname = os.path.basename(ifname)
    bname = os.path.splitext(bname)[0]
    bname = './' + bname

    # binarization of camera doc
    map_bin, bw = sepline_binarize(bgr)
    cv2.imwrite(bname + '_bw.png', bw)

    # detect separation lines of cells: vertical lines
    rho = 1
    theta = np.pi / 180
    threshold = 70
    min_line_len = 40
    max_line_gap = 5
    max_lines = np.iinfo(np.int32).max
    min_theta = -np.pi / 12
    max_theta = np.pi / 12
    lines = adadoc.ppht(map_bin, rho, theta, threshold,
                        min_line_len, max_line_gap,
                        max_lines, min_theta, max_theta)

    # draw detected separation lines
    plane = bgr.copy()
    for l in lines:
        cv2.line(plane, (l[0], l[1]), (l[2], l[3]), color=(0, 0, 0), thickness=2, lineType=8)
    cv2.imwrite(bname + '_line1.png', plane)

    # detect horizontal lines;
    min_theta = np.pi*5/12
    max_theta = np.pi*7/12
    lines = adadoc.ppht(map_bin, rho, theta, threshold,
                            min_line_len, max_line_gap,
                            max_lines, min_theta, max_theta)

    # draw detected horizontal lines;
    plane = bgr.copy()
    for l in lines:
        cv2.line(plane, (l[0], l[1]), (l[2], l[3]), color=(0, 0, 0), thickness=2, lineType=8)
    cv2.imwrite(bname + '_line2.png', plane)
            #
