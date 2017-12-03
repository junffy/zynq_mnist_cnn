# This code is based on 
# https://github.com/oreilly-japan/deep-learning-from-scratch
# http://marsee101.blog19.fc2.com
# This modified code also takes over the MIT License.


# coding: utf-8
import numpy as np
from common.functions import *
from common.util import im2col, col2im

AF_OUT_MAG = 2 ** 5 # 出力の小数部
AF_OUT_INT = 2 ** 6 # 出力の整数部（+符号1ビット）
AF_WB_MAG = 2 ** 8 # 重みとバイアスの小数部
AF_WB_INT = 2 ** 1 # 重みとバイアスの整数部（+符号1ビット）

COV_OUT_MAG = 2 ** 7 # 出力の小数部
COV_OUT_INT = 2 ** 2 # 出力の整数部（+符号1ビット）
COV_WB_MAG = 2 ** 8 # 重みとバイアスの小数部
COV_WB_INT = 2 ** 1 # 重みとバイアスの整数部（+符号1ビット）

DEBUG = 1;

class Relu:
    def __init__(self):
        self.mask = None

    def forward_int(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def forward_msg(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward_int(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def forward_msg(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.W_int = W
        self.b_int = b

        self.x = None
        self.original_x_shape = None
        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None

    def forward_int(self, x):
        if (DEBUG == 1):
            print("x shape ={0}".format(x.shape))
            print("np.max(self.W) = {0}".format(np.max(self.W)))

        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        # x は量子化できているはず
        # wとbをINT8の範囲に修正 2017/05/22 by marsee
        self.W_int = np.array(self.W*AF_WB_MAG+0.5, dtype=int)
        self.b_int = np.array(self.b*AF_WB_MAG+0.5, dtype=int)

        for i in range(self.W_int.shape[0]):
            for j in range(self.W_int.shape[1]):
                if (self.W_int[i][j] > AF_WB_MAG*AF_WB_INT/2-1):
                    self.W_int[i][j] = AF_WB_MAG*AF_WB_INT/2-1
                elif (self.W_int[i][j] < -AF_WB_MAG*AF_WB_INT/2):
                    self.W_int[i][j] = -AF_WB_MAG*AF_WB_INT/2;

        for i in range(self.b_int.shape[0]):
            if (self.b_int[i] > AF_WB_MAG*AF_WB_INT/2-1):
                self.b_int[i] = AF_WB_MAG*AF_WB_INT/2-1
            elif (self.b_int[i] < -AF_WB_MAG*AF_WB_INT/2):
                self.b_int[i] = -AF_WB_MAG*AF_WB_INT/2
        
        self.W_int = np.array(self.W_int, dtype=float)
        self.b_int = np.array(self.b_int, dtype=float)
        
        self.W_int = self.W_int/AF_WB_MAG
        self.b_int = self.b_int/AF_WB_MAG

        out = np.dot(self.x, self.W_int) + self.b_int

        if (DEBUG == 1):
            print("np.max(self.W) = {0}".format(np.max(self.W)))
            print("np.max(self.b) = {0}".format(np.max(self.b)))

            print("x reshape ={0}".format(x.shape))
            print("np.max(x) = {0}".format(np.max(x)))
            print("np.min(x) = {0}".format(np.min(x)))        
            #print("x = {0}".format(self.x))
            print(self.W_int.shape)
            print("np.max(self.W_int) = {0}".format(np.max(self.W_int)))
            print("np.min(self.W_int) = {0}".format(np.min(self.W_int)))
            print(self.b_int.shape)
            print("np.max(self.b_int) = {0}".format(np.max(self.b_int)))
            print("np.min(self.b_int) = {0}".format(np.min(self.b_int)))
            print(out.shape)
            print("np.max(out) = {0}".format(np.max(out)))
            print("np.min(out) = {0}".format(np.min(out)))
            #print("out = {0}".format(out))

        out = np.array(out*AF_OUT_MAG+0.5, dtype=int)
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                if (out[i][j] > AF_OUT_MAG*AF_OUT_INT/2-1):
                    out[i][j] = AF_OUT_MAG*AF_OUT_INT/2-1
                elif (out[i][j] < -AF_OUT_MAG*AF_OUT_INT/2):
                    out[i][j] = -AF_OUT_MAG*AF_OUT_INT/2
        out = np.array(out, dtype=float)
        out = out/AF_OUT_MAG

        if (DEBUG == 1):
            print("np.max(out2) = {0}".format(np.max(out)))
            print("np.min(out2) = {0}".format(np.min(out)))

        return out

    def forward(self, x):
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def forward_msg(self, x):
        print("x shape ={0}".format(x.shape))
        
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        print("x reshape ={0}".format(x.shape))
        print("np.max(x) = {0}".format(np.max(x)))
        print("np.min(x) = {0}".format(np.min(x)))        
        #print("x = {0}".format(self.x))
        print(self.W.shape)
        print("np.max(self.W) = {0}".format(np.max(self.W)))
        print("np.min(self.W) = {0}".format(np.min(self.W)))
        print(self.b.shape)
        print("np.max(self.b) = {0}".format(np.max(self.b)))
        print("np.min(self.b) = {0}".format(np.min(self.b)))
        print(out.shape)
        print("np.max(out) = {0}".format(np.max(out)))
        print("np.min(out) = {0}".format(np.min(out)))
        #print("out = {0}".format(out))

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None # softmaxの出力
        self.t = None # 教師データ

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def forward_msg(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def forward_int(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 教師データがone-hot-vectorの場合
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx


class Dropout:
    """
    http://arxiv.org/abs/1207.0580
    """
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward_int(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def forward_msg(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None # Conv層の場合は4次元、全結合層の場合は2次元  

        # テスト時に使用する平均と分散
        self.running_mean = running_mean
        self.running_var = running_var  
        
        # backward時に使用する中間データ
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward_int(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)
        
        return out.reshape(*self.input_shape)
            
    def forward_msg(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)
        
        return out.reshape(*self.input_shape)
            
    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)
        
        return out.reshape(*self.input_shape)
            
    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
            
        out = self.gamma * xn + self.beta 
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        self.W_int = W
        self.b_int = b

        # 中間データ（backward時に使用）
        self.x = None   
        self.col = None
        self.col_W = None
        self.col_W_int = None
        
        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.db = None

    def forward_int(self, x):
        # wとbをINT8の範囲に修正 2017/06/06 by marsee
        self.W_int = np.array(self.W*COV_WB_MAG+0.5, dtype=int)
        self.b_int = np.array(self.b*COV_WB_MAG+0.5, dtype=int)
        for i in range(self.W_int.shape[0]):
            for j in range(self.W_int.shape[1]):
                for k in range(self.W_int.shape[2]):
                    for m in range(self.W_int.shape[3]):
                        if (self.W_int[i][j][k][m] > COV_WB_MAG*COV_WB_INT/2-1):
                            self.W_int[i][j][k][m] = COV_WB_MAG*COV_WB_INT/2-1
                        elif (self.W_int[i][j][k][m] < -COV_WB_MAG*COV_WB_INT/2):
                            self.W_int[i][j][k][m] = -COV_WB_MAG*COV_WB_INT/2;
        for i in range(self.b_int.shape[0]):
            if (self.b_int[i] > COV_WB_MAG*COV_WB_INT/2-1):
                self.b_int[i] = COV_WB_MAG*COV_WB_INT/2-1
            elif (self.b_int[i] < -COV_WB_MAG*COV_WB_INT/2):
                self.b_int[i] = -COV_WB_MAG*COV_WB_INT/2
        
        self.W_int = np.array(self.W_int, dtype=float)
        self.b_int = np.array(self.b_int, dtype=float)
        
        self.W_int = self.W_int/COV_WB_MAG
        self.b_int = self.b_int/COV_WB_MAG


        FN, C, FH, FW = self.W_int.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)
       
        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W_int = self.W_int.reshape(FN, -1).T

        out = np.dot(col, col_W_int) + self.b_int

        if (DEBUG == 1):
            print(x.shape)
            print("Conv col.shape = {0}".format(col.shape))
            print("Conv col_W.shape = {0}".format(col_W_int.shape))

            print("Conv np.max(x) = {0}".format(np.max(x)))
            print("Conv np.min(x) = {0}".format(np.min(x)))        
            #print("Conv x = {0}".format(self.x))
            print(self.W_int.shape)
            print("Conv np.max(self.W_int) = {0}".format(np.max(self.W_int)))
            print("Conv np.min(self.W_int) = {0}".format(np.min(self.W_int)))
            print(self.b_int.shape)
            print("Conv np.max(self.b_int) = {0}".format(np.max(self.b_int)))
            print("Conv np.min(self.b_int) = {0}".format(np.min(self.b_int)))
            print("Conv out.shape = {0}".format(out.shape))
            print("Conv np.max(out) = {0}".format(np.max(out)))
            print("Conv np.min(out) = {0}".format(np.min(out)))
            #print("Conv out = {0}".format(out))

        out = np.array(out*COV_OUT_MAG+0.5, dtype=int)
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                if (out[i][j] > COV_OUT_MAG*COV_OUT_INT/2-1):
                    out[i][j] = COV_OUT_MAG*COV_OUT_INT/2-1
                elif (out[i][j] < -COV_OUT_MAG*COV_OUT_INT/2):
                    out[i][j] = -COV_OUT_MAG*COV_OUT_INT/2
        out = np.array(out, dtype=float)
        out = out/COV_OUT_MAG

        if (DEBUG == 1):
            print("Conv np.max(out2) = {0}".format(np.max(out)))
            print("Conv np.min(out2) = {0}".format(np.min(out)))


        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        if (DEBUG == 1):
            print("Conv out.reshape = {0}".format(out.shape))

        self.x = x
        self.col = col
        self.col_W_int = col_W_int

        return out

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def forward_msg(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b

        print(x.shape)
        print("Conv col.shape = {0}".format(col.shape))
        print("Conv col_W.shape = {0}".format(col_W.shape))

        print("Conv np.max(x) = {0}".format(np.max(x)))
        print("Conv np.min(x) = {0}".format(np.min(x)))        
        #print("Conv x = {0}".format(self.x))
        print(self.W.shape)
        print("Conv np.max(self.W) = {0}".format(np.max(self.W)))
        print("Conv np.min(self.W) = {0}".format(np.min(self.W)))
        print(self.b.shape)
        print("Conv np.max(self.b) = {0}".format(np.max(self.b)))
        print("Conv np.min(self.b) = {0}".format(np.min(self.b)))
        print("Conv out.shape = {0}".format(out.shape))
        print("Conv np.max(out) = {0}".format(np.max(out)))
        print("Conv np.min(out) = {0}".format(np.min(out)))
        #print("Conv out = {0}".format(out))
        print("Conv np.max(out2) = {0}".format(np.max(out)))
        print("Conv np.min(out2) = {0}".format(np.min(out)))

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        print("Conv out.reshape = {0}".format(out.shape))

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def forward_int(self, x):
        if (DEBUG == 1):
            print("Pooling x.shape = {0}".format(x.shape))

        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        if (DEBUG == 1):
            print("Pooling out.shape = {0}".format(out.shape))

        return out

    def forward_msg(self, x):
        print("Pooling x.shape = {0}".format(x.shape))

        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        print("Pooling out.shape = {0}".format(out.shape))

        return out

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx

