import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops
import adadoc
import time

if __name__ == '__main__':
    img = cv2.imread('001000_0_1.png', cv2.IMREAD_COLOR)
    # if img is None:
    #     print('image non')
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grey = cv2.resize(grey, (128, 128), interpolation=cv2.INTER_LINEAR)
	
    # grey = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 2, 2, 2, 2, 2, 3, 3]).reshape(4, 4)
    distances = [1, 5]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = greycomatrix(grey, distances=distances, angles=angles, levels=256,
                        symmetric=False, normed=False)
	
    # for i in range(0, 256):
    #     print(glcm[i, :, 0, 0])
            
    properties = ['contrast', 'dissimilarity', 'homogeneity']
    t2 = time.time()
    feat = np.hstack([greycoprops(glcm, prop=prop).ravel() for prop in properties])
    t3 = time.time()
    print('time1 = ', t3 - t2)
    # print('glcm =')
    # print(glcm[:, :, 0, 0])
    print('feat = ', feat)
    t0 = time.time()
    # feature = np.hstack([adadoc.greycoprops(glcm[:, :, i, j]).ravel() for i in range(0, 2)])
    for i in range(0, 2):
        for j in range(0, 4):
            feature = adadoc.greycoprops(glcm[:, :, i, j]).ravel()
            print('feature = ', feature)
    t1 = time.time()
    print('time2 = ', t1 - t0)
    
