import cv2
import numpy as np
import scipy.misc
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
from matplotlib import pyplot as plt
from scipy import cluster


class BinaryImage:

    def __init__(self, img_raw, img_label):
        self.im_raw = img_raw
        self.im_label = img_label
        (m, n) = self.im_raw.shape
        for r in range(m):
            for c in range(n):
                if self.im_label[r][c] == 0:
                    self.im_label[r][c] = 1
                elif self.im_label[r][c] == 255:
                    self.im_label[r][c] = 0

    def fft_power_spectrum(self):
        """
        :param img: M x N uint8 graylevel image
        :return: T = M*N x 25 double texture parameters, each texture para is a column
        """
        img = self.im_raw
        (m, n) = img.shape
        T = np.array([[0.0 for _ in range(25)] for _ in range(m*n)])
        padding = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
        (M, N) = padding.shape # M rows, N cols
        #cv2.imshow('pad', padding)
        for i in range(2, M-2):
            for j in range(2, N-2):
                #print(padding[i-2:i+3, j-2:j+3])
                f = padding[i - 2:i + 3, j - 2:j + 3]
                F = np.fft.fft2(f)
                T[(i-2) * m + (j-2), :] = np.absolute(F.flatten())  # complex module
        return T

    def generate_ring(self, n):
        ring = np.zeros((2 * n - 1, 2 * n - 1, n))
        ring[n - 1, n - 1, 0] = 1
        for i in range(1, n):
            ring[n - i - 1: n + i, n - i - 1: n + i, i] = np.ones((2 * i + 1, 2 * i + 1))
            ring[n - i: n + i - 1, n - i: n + i - 1, i] = np.zeros((2 * (i - 1) + 1, 2 * (i - 1) + 1))
        return ring

    def fft_radial(self):
        img = self.im_raw
        # generate ring
        ring = self.generate_ring(10)
        (m, n) = img.shape
        T = np.array([[0.0 for _ in range(10)] for _ in range(m * n)])
        padding = cv2.copyMakeBorder(img, 9, 9, 9, 9, cv2.BORDER_CONSTANT, value=0)
        (M, N) = padding.shape  # M rows, N cols
        k = 9
        for i in range(k, M-k):
            for j in range(k, N-k):
                #print(padding[i-2:i+3, j-2:j+3])
                f = padding[i - k:i + k+1, j - k:j + k+1]
                F = np.absolute(np.fft.fft2(f))
                #print(F)
                t = np.zeros(10)
                for q in range(10):
                    t[q] = np.sum(np.multiply(F, ring[:, :, q]))
                T[(i - k) * m + (j - k), :] = t
        return T


    def isbi_to_df(self):
        # fft series
        T_texture = self.fft_power_spectrum()
        T_radial = self.fft_radial()
        laws = self.laws()

        # laplacian gradient
        laplacian = cv2.Laplacian(img_raw, cv2.CV_64F)
        # sobel-x
        sobelx = cv2.Sobel(img_raw, cv2.CV_64F, 1, 0, ksize=5)
        # sobel-y
        sobely = cv2.Sobel(img_raw, cv2.CV_64F, 0, 1, ksize=5)

        label = self.im_label

        (m, n) = self.im_raw.shape

        cols = ['label', 'graylevel', 'log', 'sobelx', 'sobely']
        for j in range(len(T_texture[0])):
            cols.append('fft_texture'+str(j))

        for j in range(len(T_radial[0])):
            cols.append('fft_radial'+str(j))

        for j in range(len(laws[0])):
            cols.append('laws'+str(j))

        print(cols)

        df = pd.DataFrame(index=range(m*n), columns=cols)
        list_of_data = []
        for r in range(m):
            for c in range(n):
                i = r * 512 + c
                #print(r, c)
                #print(r * 512 + c)

                data = [label[r][c], self.im_raw[r][c], laplacian[r][c], sobelx[r][c], sobely[r][c]] + \
                       list(T_texture[i]) + \
                       list(T_radial[i]) + \
                       list(laws[i])
                print(i, data)

                #list_of_data.append(pd.Series(data, index=cols))
                df.iloc[i] = data


        #df.append(list_of_data, ignore_index=True)
        return df

    def laws(self):
        (m, n) = self.im_raw.shape
        img = np.float32(self.im_raw)

        L7 = np.mat([1, 6, 15, 20, 15, 6, 1])
        E7 = np.mat([-1, -4, -5, 0, 5, 4, 1])
        S7 = np.mat([-1, -2, 1, 4, 1, -2, -1])
        W7 = np.mat([-1, 0, 3, 0, -3, 0, 1])
        R7 = np.mat([1, -2, -1, 4, -1, -2, 1])
        O7 = np.mat([-1, 6, -15, 20, -15, 6, -1])

        L7L7 = L7.T * L7
        L7E7 = L7.T * E7
        L7S7 = L7.T * S7
        L7W7 = L7.T * W7
        L7R7 = L7.T * R7
        L7O7 = L7.T * O7
        E7E7 = E7.T * E7
        W7R7 = W7.T * R7
        W7O7 = W7.T * O7
        mean = np.ones((5, 5), np.float32) / 25

        L7L7c = cv2.filter2D(img, -1, L7L7)
        L7E7c = cv2.filter2D(img, -1, L7E7)
        L7S7c = cv2.filter2D(img, -1, L7S7)
        L7W7c = cv2.filter2D(img, -1, L7W7)
        L7R7c = cv2.filter2D(img, -1, L7R7)
        L7O7c = cv2.filter2D(img, -1, L7O7)
        E7E7c = cv2.filter2D(img, -1, E7E7)
        W7R7c = cv2.filter2D(img, -1, W7R7)
        W7O7c = cv2.filter2D(img, -1, W7O7)
        MEANc = cv2.filter2D(img, -1, mean)


        res = []
        for r in range(m):
            for c in range(n):
                res.append([L7L7c[r][c],
                            L7E7c[r][c],
                            L7S7c[r][c],
                            L7W7c[r][c],
                            L7R7c[r][c],
                            L7O7c[r][c],
                            E7E7c[r][c],
                            W7R7c[r][c],
                            W7O7c[r][c],
                            MEANc[r][c]])

        return res



im_raw = cv2.imread('14.png')
im_label = cv2.imread('14_label.png')

img_raw = cv2.cvtColor(im_raw, cv2.COLOR_BGR2GRAY)
img_label = cv2.cvtColor(im_label, cv2.COLOR_BGR2GRAY)

bi = BinaryImage(img_raw, img_label)
df = bi.isbi_to_df()
print(df)
df.to_csv('data.csv')


"""
#T = fft_power_spectrum(img)
#[cidx,ctrs] = cluster.vq.kmeans2(T, 4)
#print(cidx, ctrs)
#T = fft_radial(img)
T = get_bgr_hist(im_raw)

# laplacian gradient
laplacian = cv2.Laplacian(img_raw,cv2.CV_64F)
# sobel-x
sobelx = cv2.Sobel(img_raw,cv2.CV_64F,1,0,ksize=5)

#sobel-y
sobely = cv2.Sobel(img_raw,cv2.CV_64F,0,1,ksize=5)

Laws = 0


print(laplacian)
print(sobelx)
print(sobely)

(height, width) = (len(img_raw), len(img_raw))

df = pd.DataFrame()

for n, i in enumerate(range(height)):
    for j in range(width):
        data = {}
        data['labal'] = img_label[i][j]

print(df)

print(img_label[0][0])
"""


