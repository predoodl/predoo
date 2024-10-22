import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
import MNN
from tensorflow import keras
from tensorflow.keras import layers
import csv
import time
import math
F_mnn = MNN.expr

np.random.seed(0)


def input_withDiffDype(x, dtype):
    return tf.convert_to_tensor(x, dtype=dtype)

def torch_input_withDiffDype(x, dtype):
    return torch.tensor(x, dtype=dtype)

def tf_NormWithDiffDype(dtype):
    return tf.keras.layers.BatchNormalization(
        axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
        beta_initializer='zeros', gamma_initializer='ones',
        moving_mean_initializer='zeros', moving_variance_initializer='ones',dtype=dtype
)

def torch_NormWithDiffDype(dtype):
    torch_norm = torch.nn.BatchNorm2d(2)
    # torch_tanh.half()
    torch_norm.type(dtype)
    return torch_norm

def getDataForTensorflow(f):
    out = open(file=f, mode="a", newline='')
    csv_writer = csv.writer(out)
    for j in range(1000):
        print('j= ', j)
        print('------------------------------------------')
        x = np.random.randn(1, 2, 4, 4)
        csv_writer.writerow([x])
        csv_writer.writerow(["No.", "16_32", "16_64", "32_16", "32_64", "64_16", "64_32"])
        for i in range(100):
            print(i)
            res = []
            res.append(i)

            # TF Pooling
            x_32 = input_withDiffDype(x, tf.float32)
            x_16 = input_withDiffDype(x, tf.float16)
            x_64 = input_withDiffDype(x, tf.float64)

            s=time.time()
            tf_Norm_16 = tf_NormWithDiffDype('float16')
            tf_Norm_32 = tf_NormWithDiffDype('float32')
            tf_Norm_64 = tf_NormWithDiffDype('float64')

            out_16_16_1 = tf_Norm_16(x_16).numpy().astype(np.float32)
            out_16_16_2 = tf_Norm_16(x_16).numpy().astype(np.float64)
            out_16_32 = tf_Norm_32(x_16)
            out_16_64 = tf_Norm_64(x_16)

            diff1 = np.mean(out_16_32 - out_16_16_1)  # 低精度到高精度
            diff2 = np.mean(out_16_64 - out_16_16_2)  # 低精度到高精度


            out_32_32_1 = tf_Norm_32(x_32)
            out_32_32_2 = tf_Norm_32(x_32).numpy().astype(np.float64)
            out_32_16 = tf_Norm_16(x_32).numpy().astype(np.float32)
            out_32_64 = tf_Norm_64(x_32)

            diff3 = np.mean(out_32_16 - out_32_32_1)  # 高精度到低精度
            diff4 = np.mean(out_32_64 - out_32_32_2)  # 低精度到高精度


            out_64_16 = tf_Norm_16(x_64).numpy().astype(np.float64)
            out_64_32 = tf_Norm_32(x_64).numpy().astype(np.float64)
            out_64_64 = tf_Norm_64(x_64)

            diff5 = np.mean(out_64_16 - out_64_64)  # 高精度到低精度
            diff6 = np.mean(out_64_32 - out_64_64)  # 低精度到高精度

            e=time.time()

            res.append(diff1)
            res.append(diff2)
            res.append(diff3)
            res.append(diff4)
            res.append(diff5)
            res.append(diff6)


            csv_writer.writerow(res)

    out.close()


def getDataForTfWithG(f,g):
    out = open(file=f, mode="a", newline='')
    out1 = open(file=g, mode="a", newline='')

    csv_writer = csv.writer(out)
    csv_writer1 = csv.writer(out1)

    csv_writer.writerow(["No.", "16_32(16)", "16_64(16)", "32_16(32)", "32_64(32)", "64_16(64)", "64_32(64)",
                         "time1", "32_16(16)", "64_16(16)", "16_32(32)", "64_32(32)", "16_64(64)", "32_64(64)",
                         "time2","isNaN"])
    csv_writer1.writerow(
        ["No.", "当前最大误差(同输入)", "全局最大误差(同输入)", "引起最大误差的输入编号1", "当前最大误差(同算子)", "全局最大误差(同算子)", "引起最大误差的输入编号2"])
    h_error1 = 0
    h_error2 = 0
    for i in range(20):
        tmp1 = 0
        tmp2 = 0
        index1 = 0
        index2 = 0
        info = []
        info.append(i)
        for j in range(1000):
            print('j= ', j)
            print('------------------------------------------')
            x = np.random.randn(1, 2, 4, 4)
            res = []
            res.append(j)

            # TF Pooling
            x_32 = input_withDiffDype(x, tf.float32)
            x_16 = input_withDiffDype(x, tf.float16)
            x_64 = input_withDiffDype(x, tf.float64)

            s=time.time()
            tf_Norm_16 = tf_NormWithDiffDype('float16')
            tf_Norm_32 = tf_NormWithDiffDype('float32')
            tf_Norm_64 = tf_NormWithDiffDype('float64')

            out_16_16_1 = tf_Norm_16(x_16).numpy().astype(np.float32)
            out_16_16_2 = tf_Norm_16(x_16).numpy().astype(np.float64)
            out_16_32 = tf_Norm_32(x_16)
            out_16_64 = tf_Norm_64(x_16)

            diff1 = np.mean(np.abs(out_16_32 - out_16_16_1) ) # 低精度到高精度
            diff2 = np.mean(np.abs(out_16_64 - out_16_16_2))  # 低精度到高精度


            out_32_32_1 = tf_Norm_32(x_32)
            out_32_32_2 = tf_Norm_32(x_32).numpy().astype(np.float64)
            out_32_16 = tf_Norm_16(x_32).numpy().astype(np.float32)
            out_32_64 = tf_Norm_64(x_32)

            diff3 = np.mean(np.abs(out_32_16 - out_32_32_1))  # 高精度到低精度
            diff4 = np.mean(np.abs(out_32_64 - out_32_32_2))  # 低精度到高精度


            out_64_16 = tf_Norm_16(x_64).numpy().astype(np.float64)
            out_64_32 = tf_Norm_32(x_64).numpy().astype(np.float64)
            out_64_64 = tf_Norm_64(x_64)

            diff5 = np.mean(np.abs(out_64_16 - out_64_64)) # 高精度到低精度
            diff6 = np.mean(np.abs(out_64_32 - out_64_64))  # 低精度到高精度
            e=time.time()


            res.append(diff1)
            res.append(diff2)
            res.append(diff3)
            res.append(diff4)
            res.append(diff5)
            res.append(diff6)
            res.append(e-s)

            s = time.time()
            out_16_16 = tf_Norm_16(x_16)
            diff7 = np.mean(np.abs(tf_Norm_16(x_32) - out_16_16))
            diff8 = np.mean(np.abs(tf_Norm_16(x_64) - out_16_16))

            diff9 = np.mean(np.abs(tf_Norm_32(x_16) - out_32_32_1))
            diff10 = np.mean(np.abs(tf_Norm_32(x_64) - out_32_32_1))

            diff11 = np.mean(np.abs(tf_Norm_64(x_16) - out_64_64))
            diff12 = np.mean(np.abs(tf_Norm_64(x_32) - out_64_64))

            e = time.time()
            res.append(diff7)
            res.append(diff8)
            res.append(diff9)
            res.append(diff10)
            res.append(diff11)
            res.append(diff12)
            res.append(e - s)

            for n in out_32_32_1.numpy().ravel():
                if math.isnan(n):
                    res.append("NAN")
                    break

            csv_writer.writerow(res)

            if max(res[1:7]) > tmp1:
                index1 = j
                tmp1 = max(max(res[1:7]), tmp1)

            if max(res[8:14]) > tmp2:
                index2 = j
                tmp2 = max(max(res[8:14]), tmp2)

        h_error1 = max(h_error1, tmp1)
        h_error2 = max(h_error2, tmp2)
        info.append(tmp1)
        info.append(h_error1)
        info.append(index1)
        info.append(tmp2)
        info.append(h_error2)
        info.append(index2)

        csv_writer1.writerow(info)

    out.close()
    out1.close()


def tf_disturb(f):
    out = open(file=f, mode="a", newline='')
    csv_writer = csv.writer(out)
    for j in range(1000):
        print('j= ', j)
        print('------------------------------------------')
        x = np.random.randn(1, 2, 4, 4)
        a1 = 0.000001 * np.ones((1, 2, 4, 4), np.float64)
        a2 = 0.00000001 * np.ones((1, 2, 4, 4), np.float64)
        a3 = 0.0000000001 * np.ones((1, 2, 4, 4), np.float64)
        csv_writer.writerow([x])
        csv_writer.writerow(["No.", "16_32", "16_64", "32_16", "32_64", "64_16", "64_32"])

        getdata(x,csv_writer)
        csv_writer.writerow(["+10^-6", "16_32", "16_64", "32_16", "32_64", "64_16", "64_32"])
        getdata(x+a1,csv_writer)

        csv_writer.writerow(["+10^-8", "16_32", "16_64", "32_16", "32_64", "64_16", "64_32"])
        getdata(x+a2,csv_writer)

        csv_writer.writerow(["+10^-10", "16_32", "16_64", "32_16", "32_64", "64_16", "64_32"])
        getdata(x+a3,csv_writer)


    out.close()

def getdata(x,csv_writer):
    for i in range(20):
        res = []
        res.append(i)

        # TF Pooling
        x_32 = input_withDiffDype(x, tf.float32)
        x_16 = input_withDiffDype(x, tf.float16)
        x_64 = input_withDiffDype(x, tf.float64)

        tf_Norm_16 = tf_NormWithDiffDype('float16')
        tf_Norm_32 = tf_NormWithDiffDype('float32')
        tf_Norm_64 = tf_NormWithDiffDype('float64')

        out_16_16_1 = tf_Norm_16(x_16).numpy().astype(np.float32)
        out_16_16_2 = tf_Norm_16(x_16).numpy().astype(np.float64)
        out_16_32 = tf_Norm_32(x_16)
        out_16_64 = tf_Norm_64(x_16)

        diff1 = np.mean(out_16_32 - out_16_16_1)  # 低精度到高精度
        diff2 = np.mean(out_16_64 - out_16_16_2)  # 低精度到高精度
        res.append(diff1)
        res.append(diff2)

        out_32_32_1 = tf_Norm_32(x_32)
        out_32_32_2 = tf_Norm_32(x_32).numpy().astype(np.float64)
        out_32_16 = tf_Norm_16(x_32).numpy().astype(np.float32)
        out_32_64 = tf_Norm_64(x_32)

        diff3 = np.mean(out_32_16 - out_32_32_1)  # 高精度到低精度
        diff4 = np.mean(out_32_64 - out_32_32_2)  # 低精度到高精度
        res.append(diff3)
        res.append(diff4)

        out_64_16 = tf_Norm_16(x_64).numpy().astype(np.float64)
        out_64_32 = tf_Norm_32(x_64).numpy().astype(np.float64)
        out_64_64 = tf_Norm_64(x_64)

        diff5 = np.mean(out_64_16 - out_64_64)  # 高精度到低精度
        diff6 = np.mean(out_64_32 - out_64_64)  # 低精度到高精度
        res.append(diff5)
        res.append(diff6)

        csv_writer.writerow(res)

def tf_disturb_timeflow(f,g):
    out = open(file=f, mode="a", newline='')
    out1 = open(file=g, mode="a", newline='')
    csv_writer = csv.writer(out)
    csv_writer1 = csv.writer(out1)
    csv_writer.writerow(["No.", "16_32", "16_64", "32_16", "32_64", "64_16", "64_32","time","disturb"])
    csv_writer1.writerow(["No.", "当前最大误差", "全局最大误差", "引起最大误差的输入编号"])
    h_error = 0
    a1 = 0.000001 * np.ones((1, 2, 4, 4), np.float64)
    a2 = 0.00000001 * np.ones((1, 2, 4, 4), np.float64)
    a3 = 0.0000000001 * np.ones((1, 2, 4, 4), np.float64)
    for i in range(20):
        tmp=0
        index=0
        info=[]
        info.append(i)
        for j in range(1000):
            err=[]
            print('j= ', j)
            print('------------------------------------------')
            x = np.random.randn(1, 2, 4, 4)
            getdatafortimeflow(x,csv_writer,j,"0",err)
            getdatafortimeflow(x+a1,csv_writer,j,"e-6",err)
            getdatafortimeflow(x+a2,csv_writer,j,"e-8",err)
            getdatafortimeflow(x+a3,csv_writer,j,"e-10",err)
            if max(err)>tmp:
                tmp=max(err)
                index=j

        h_error=max(h_error,tmp)
        info.append(tmp)
        info.append(h_error)
        info.append(index)
        csv_writer1.writerow(info)

    out.close()
    out1.close()

def getdatafortimeflow(x,csv_writer,j,disturb,err):
    res = []
    res.append(j)

    x_32 = input_withDiffDype(x, tf.float32)
    x_16 = input_withDiffDype(x, tf.float16)
    x_64 = input_withDiffDype(x, tf.float64)

    s = time.time()
    tf_Norm_16 = tf_NormWithDiffDype('float16')
    tf_Norm_32 = tf_NormWithDiffDype('float32')
    tf_Norm_64 = tf_NormWithDiffDype('float64')

    out_16_16_1 = tf_Norm_16(x_16).numpy().astype(np.float32)
    out_16_16_2 = tf_Norm_16(x_16).numpy().astype(np.float64)
    out_16_32 = tf_Norm_32(x_16)
    out_16_64 = tf_Norm_64(x_16)

    diff1 = np.mean(out_16_32 - out_16_16_1)  # 低精度到高精度
    diff2 = np.mean(out_16_64 - out_16_16_2)  # 低精度到高精度

    out_32_32_1 = tf_Norm_32(x_32)
    out_32_32_2 = tf_Norm_32(x_32).numpy().astype(np.float64)
    out_32_16 = tf_Norm_16(x_32).numpy().astype(np.float32)
    out_32_64 = tf_Norm_64(x_32)

    diff3 = np.mean(out_32_16 - out_32_32_1)  # 高精度到低精度
    diff4 = np.mean(out_32_64 - out_32_32_2)  # 低精度到高精度

    out_64_16 = tf_Norm_16(x_64).numpy().astype(np.float64)
    out_64_32 = tf_Norm_32(x_64).numpy().astype(np.float64)
    out_64_64 = tf_Norm_64(x_64)

    diff5 = np.mean(out_64_16 - out_64_64)  # 高精度到低精度
    diff6 = np.mean(out_64_32 - out_64_64)  # 低精度到高精度
    e = time.time()

    res.append(diff1)
    res.append(diff2)
    res.append(diff3)
    res.append(diff4)
    res.append(diff5)
    res.append(diff6)
    res.append(e - s)
    res.append(disturb)
    err.append(max(res[1:7]))

    csv_writer.writerow(res)

if __name__ == '__main__':

    getDataForTfWithG("/home/ise/opTest/data/timeflow2/tf_gpu_2.3.1/norm.csv","/home/ise/opTest/data/timeflow2/tf_gpu_2.3.1/norm_count.csv")
