import numpy as np
import torch
import torch.nn.functional as F
import MNN
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import csv
import time
import math

F_mnn = MNN.expr

np.random.seed(0)


def input_withDiffDype(x, dtype):
    return tf.convert_to_tensor(x, dtype=dtype)


def tf_SigmoidWithDiffDype(dtype):
    return tf.keras.layers.Activation(
        'sigmoid', dtype=dtype
)

def torch_input_withDiffDype(x, dtype):
    return torch.tensor(x,dtype=dtype)

def torch_sigmoidWithDiffType(dtype):
    torch_sigmoid = torch.nn.Sigmoid()
    torch_sigmoid.type(dtype)
    return torch_sigmoid


def getdataforTorch(f):
    out = open(file=f, mode="a", newline='')
    csv_writer = csv.writer(out)
    for j in range(1000):
        print('j= ', j)
        print('------------------------------------------')
        x = np.random.randn(2, 2)
        csv_writer.writerow([x])
        csv_writer.writerow(["No.",  "32_16", "32_64", "64_16", "64_32"])

        for i in range(100):
            print(i)
            res = []
            res.append(i)

            x_32 = torch_input_withDiffDype(x, torch.float32)
            x_64 = torch_input_withDiffDype(x, torch.float64)

            torch_sigmoid_16 = torch_sigmoidWithDiffType(torch.float16)
            torch_sigmoid_32 = torch_sigmoidWithDiffType(torch.float32)
            torch_sigmoid_64 = torch_sigmoidWithDiffType(torch.float64)

            out_32_32_1 = torch_sigmoid_32(x_32).detach().numpy()
            out_32_16 = torch_sigmoid_16(x_32).detach().numpy()
            out_32_64 = torch_sigmoid_64(x_32).detach().numpy()

            diff3 = np.mean(out_32_16 - out_32_32_1)  # 高精度到低精度
            diff4 = np.mean(out_32_64 - out_32_32_1)  # 低精度到高精度
            res.append(diff3)
            res.append(diff4)

            out_64_16 = torch_sigmoid_16(x_64).detach().numpy()
            out_64_32 = torch_sigmoid_32(x_64).detach().numpy()
            out_64_64 = torch_sigmoid_64(x_64).detach().numpy()

            diff5 = np.mean(out_64_16 - out_64_64)  # 高精度到低精度
            diff6 = np.mean(out_64_32 - out_64_64)  # 低精度到高精度
            res.append(diff5)
            res.append(diff6)

            csv_writer.writerow(res)
    out.close()

def torch_disturb(f):
    out = open(file=f, mode="a", newline='')
    csv_writer = csv.writer(out)
    for j in range(1000):
        print('j= ', j)
        print('------------------------------------------')
        x = np.random.randn(2, 2)
        a1 = 0.000001 * np.ones((2, 2), np.float64)
        a2 = 0.00000001 * np.ones((2, 2), np.float64)
        a3 = 0.0000000001 * np.ones((2, 2), np.float64)

        csv_writer.writerow([x])
        csv_writer.writerow(["No.", "32_16", "32_64", "64_16", "64_32"])
        getTorchData(x, csv_writer)

        csv_writer.writerow(["+10^-6", "32_16", "32_64", "64_16", "64_32"])
        getTorchData(x + a1, csv_writer)

        csv_writer.writerow(["+10^-8", "32_16", "32_64", "64_16", "64_32"])
        getTorchData(x + a2, csv_writer)

        csv_writer.writerow(["+10^-10", "32_16", "32_64", "64_16", "64_32"])
        getTorchData(x + a3, csv_writer)

    out.close()

def getTorchData(x,csv_writer):
    for i in range(20):
        res = []
        res.append(i)

        x_32 = torch_input_withDiffDype(x, torch.float32)
        x_64 = torch_input_withDiffDype(x, torch.float64)

        torch_sigmoid_16 = torch_sigmoidWithDiffType(torch.float16)
        torch_sigmoid_32 = torch_sigmoidWithDiffType(torch.float32)
        torch_sigmoid_64 = torch_sigmoidWithDiffType(torch.float64)

        out_32_32_1 = torch_sigmoid_32(x_32).detach().numpy()
        out_32_16 = torch_sigmoid_16(x_32).detach().numpy()
        out_32_64 = torch_sigmoid_64(x_32).detach().numpy()

        diff3 = np.mean(out_32_16 - out_32_32_1)  # 高精度到低精度
        diff4 = np.mean(out_32_64 - out_32_32_1)  # 低精度到高精度
        res.append(diff3)
        res.append(diff4)

        out_64_16 = torch_sigmoid_16(x_64).detach().numpy()
        out_64_32 = torch_sigmoid_32(x_64).detach().numpy()
        out_64_64 = torch_sigmoid_64(x_64).detach().numpy()

        diff5 = np.mean(out_64_16 - out_64_64)  # 高精度到低精度
        diff6 = np.mean(out_64_32 - out_64_64)  # 低精度到高精度
        res.append(diff5)
        res.append(diff6)

        csv_writer.writerow(res)


def getDataforTensorflow(f):
    out = open(file=f, mode="a", newline='')
    csv_writer = csv.writer(out)
    for j in range(1000):
        print('j= ', j)
        print('------------------------------------------')
        x = np.random.randn(2, 2)
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

            tf_Sigmoid_16 = tf_SigmoidWithDiffDype('float16')
            tf_Sigmoid_32 = tf_SigmoidWithDiffDype('float32')
            tf_Sigmoid_64 = tf_SigmoidWithDiffDype('float64')

            out_16_16_1 = tf_Sigmoid_16(x_16).numpy().astype(np.float32)
            out_16_16_2 = tf_Sigmoid_16(x_16).numpy().astype(np.float64)
            out_16_32 = tf_Sigmoid_32(x_16)
            out_16_64 = tf_Sigmoid_64(x_16)

            diff1 = np.mean(out_16_32 - out_16_16_1)  # 低精度到高精度
            diff2 = np.mean(out_16_64 - out_16_16_2)  # 低精度到高精度
            res.append(diff1)
            res.append(diff2)

            out_32_32_1 = tf_Sigmoid_32(x_32)
            out_32_32_2 = tf_Sigmoid_32(x_32).numpy().astype(np.float64)
            out_32_16 = tf_Sigmoid_16(x_32).numpy().astype(np.float32)
            out_32_64 = tf_Sigmoid_64(x_32)

            diff3 = np.mean(out_32_16 - out_32_32_1)  # 高精度到低精度
            diff4 = np.mean(out_32_64 - out_32_32_2)  # 低精度到高精度
            res.append(diff3)
            res.append(diff4)

            out_64_16 = tf_Sigmoid_16(x_64).numpy().astype(np.float64)
            out_64_32 = tf_Sigmoid_32(x_64).numpy().astype(np.float64)
            out_64_64 = tf_Sigmoid_64(x_64)

            diff5 = np.mean(out_64_16 - out_64_64)  # 高精度到低精度
            diff6 = np.mean(out_64_32 - out_64_64)  # 低精度到高精度
            res.append(diff5)
            res.append(diff6)

            csv_writer.writerow(res)
    out.close()


def getDataforTfWihthG(f,g):
    out = open(file=f, mode="a", newline='')
    out1 = open(file=g, mode="a", newline='')

    csv_writer = csv.writer(out)
    csv_writer1 = csv.writer(out1)

    csv_writer.writerow(["No.", "16_32(16)", "16_64(16)", "32_16(32)", "32_64(32)", "64_16(64)", "64_32(64)",
                         "time1", "32_16(16)", "64_16(16)", "16_32(32)", "64_32(32)", "16_64(64)", "32_64(64)",
                         "time2", "isNaN"])
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
            x = np.random.randn(2, 2)
            res = []
            res.append(j)

            x_32 = input_withDiffDype(x, tf.float32)
            x_16 = input_withDiffDype(x, tf.float16)
            x_64 = input_withDiffDype(x, tf.float64)

            s=time.time()
            tf_Sigmoid_16 = tf_SigmoidWithDiffDype('float16')
            tf_Sigmoid_32 = tf_SigmoidWithDiffDype('float32')
            tf_Sigmoid_64 = tf_SigmoidWithDiffDype('float64')

            out_16_16_1 = tf_Sigmoid_16(x_16).numpy().astype(np.float32)
            out_16_16_2 = tf_Sigmoid_16(x_16).numpy().astype(np.float64)
            out_16_32 = tf_Sigmoid_32(x_16)
            out_16_64 = tf_Sigmoid_64(x_16)

            diff1 = np.mean(np.abs(out_16_32 - out_16_16_1))  # 低精度到高精度
            diff2 = np.mean(np.abs(out_16_64 - out_16_16_2))  # 低精度到高精度


            out_32_32_1 = tf_Sigmoid_32(x_32)
            out_32_32_2 = tf_Sigmoid_32(x_32).numpy().astype(np.float64)
            out_32_16 = tf_Sigmoid_16(x_32).numpy().astype(np.float32)
            out_32_64 = tf_Sigmoid_64(x_32)

            diff3 = np.mean(np.abs(out_32_16 - out_32_32_1))  # 高精度到低精度
            diff4 = np.mean(np.abs(out_32_64 - out_32_32_2))  # 低精度到高精度


            out_64_16 = tf_Sigmoid_16(x_64).numpy().astype(np.float64)
            out_64_32 = tf_Sigmoid_32(x_64).numpy().astype(np.float64)
            out_64_64 = tf_Sigmoid_64(x_64)

            diff5 = np.mean(np.abs(out_64_16 - out_64_64))  # 高精度到低精度
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
            out_16_16 = tf_Sigmoid_16(x_16)
            diff7 = np.mean(np.abs(tf_Sigmoid_16(x_32) - out_16_16))
            diff8 = np.mean(np.abs(tf_Sigmoid_16(x_64) - out_16_16))

            diff9 = np.mean(np.abs(tf_Sigmoid_32(x_16) - out_32_32_1))
            diff10 = np.mean(np.abs(tf_Sigmoid_32(x_64) - out_32_32_1))

            diff11 = np.mean(np.abs(tf_Sigmoid_64(x_16) - out_64_64))
            diff12 = np.mean(np.abs(tf_Sigmoid_64(x_32) - out_64_64))

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
        x = np.random.randn(2, 2)
        a1 = 0.000001 * np.ones((2, 2), np.float64)
        a2 = 0.00000001 * np.ones((2, 2), np.float64)
        a3 = 0.0000000001 * np.ones((2, 2), np.float64)
        csv_writer.writerow([x])
        csv_writer.writerow(["No.", "16_32", "16_64", "32_16", "32_64", "64_16", "64_32"])
        getdata(x, csv_writer)

        csv_writer.writerow(["+10^-6", "16_32", "16_64", "32_16", "32_64", "64_16", "64_32"])
        getdata(x + a1, csv_writer)

        csv_writer.writerow(["+10^-8", "16_32", "16_64", "32_16", "32_64", "64_16", "64_32"])
        getdata(x + a2, csv_writer)

        csv_writer.writerow(["+10^-10", "16_32", "16_64", "32_16", "32_64", "64_16", "64_32"])
        getdata(x + a3, csv_writer)

    out.close()


def getdata(x,csv_writer):
    for i in range(20):
        res = []
        res.append(i)

        # TF Pooling
        x_32 = input_withDiffDype(x, tf.float32)
        x_16 = input_withDiffDype(x, tf.float16)
        x_64 = input_withDiffDype(x, tf.float64)

        tf_Sigmoid_16 = tf_SigmoidWithDiffDype('float16')
        tf_Sigmoid_32 = tf_SigmoidWithDiffDype('float32')
        tf_Sigmoid_64 = tf_SigmoidWithDiffDype('float64')

        out_16_16_1 = tf_Sigmoid_16(x_16).numpy().astype(np.float32)
        out_16_16_2 = tf_Sigmoid_16(x_16).numpy().astype(np.float64)
        out_16_32 = tf_Sigmoid_32(x_16)
        out_16_64 = tf_Sigmoid_64(x_16)

        diff1 = np.mean(out_16_32 - out_16_16_1)  # 低精度到高精度
        diff2 = np.mean(out_16_64 - out_16_16_2)  # 低精度到高精度
        res.append(diff1)
        res.append(diff2)

        out_32_32_1 = tf_Sigmoid_32(x_32)
        out_32_32_2 = tf_Sigmoid_32(x_32).numpy().astype(np.float64)
        out_32_16 = tf_Sigmoid_16(x_32).numpy().astype(np.float32)
        out_32_64 = tf_Sigmoid_64(x_32)

        diff3 = np.mean(out_32_16 - out_32_32_1)  # 高精度到低精度
        diff4 = np.mean(out_32_64 - out_32_32_2)  # 低精度到高精度
        res.append(diff3)
        res.append(diff4)

        out_64_16 = tf_Sigmoid_16(x_64).numpy().astype(np.float64)
        out_64_32 = tf_Sigmoid_32(x_64).numpy().astype(np.float64)
        out_64_64 = tf_Sigmoid_64(x_64)

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
    a1 = 0.000001 * np.ones((2, 2), np.float64)
    a2 = 0.00000001 * np.ones((2, 2), np.float64)
    a3 = 0.0000000001 * np.ones((2, 2), np.float64)
    for i in range(20):
        tmp=0
        index=0
        info=[]
        info.append(i)
        for j in range(1000):
            err=[]
            print('j= ', j)
            print('------------------------------------------')
            x = np.random.randn(2, 2)
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
    tf_Sigmoid_16 = tf_SigmoidWithDiffDype('float16')
    tf_Sigmoid_32 = tf_SigmoidWithDiffDype('float32')
    tf_Sigmoid_64 = tf_SigmoidWithDiffDype('float64')

    out_16_16_1 = tf_Sigmoid_16(x_16).numpy().astype(np.float32)
    out_16_16_2 = tf_Sigmoid_16(x_16).numpy().astype(np.float64)
    out_16_32 = tf_Sigmoid_32(x_16)
    out_16_64 = tf_Sigmoid_64(x_16)

    diff1 = np.mean(out_16_32 - out_16_16_1)  # 低精度到高精度
    diff2 = np.mean(out_16_64 - out_16_16_2)  # 低精度到高精度

    out_32_32_1 = tf_Sigmoid_32(x_32)
    out_32_32_2 = tf_Sigmoid_32(x_32).numpy().astype(np.float64)
    out_32_16 = tf_Sigmoid_16(x_32).numpy().astype(np.float32)
    out_32_64 = tf_Sigmoid_64(x_32)

    diff3 = np.mean(out_32_16 - out_32_32_1)  # 高精度到低精度
    diff4 = np.mean(out_32_64 - out_32_32_2)  # 低精度到高精度

    out_64_16 = tf_Sigmoid_16(x_64).numpy().astype(np.float64)
    out_64_32 = tf_Sigmoid_32(x_64).numpy().astype(np.float64)
    out_64_64 = tf_Sigmoid_64(x_64)

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

def tf_random(f,g):
    out = open(file=f, mode="a", newline='')
    out1 = open(file=g, mode="a", newline='')

    csv_writer = csv.writer(out)
    csv_writer1 = csv.writer(out1)

    csv_writer.writerow(["No.", "16_64","32_64","time",
                         "isNaN"])
    csv_writer1.writerow(
        ["No.", "当前最大误差", "全局最大误差", "引起最大误差的输入编号"])
    h_error1 = 0

    for i in range(20):
        tmp1 = 0
        index1 = 0
        info = []
        info.append(i)
        for j in range(1000):
            print('j= ', j)
            x = np.random.randn(2,2)
            res = []
            res.append(j)

            x_32 = input_withDiffDype(x, tf.float32)
            x_16 = input_withDiffDype(x, tf.float16)
            x_64 = input_withDiffDype(x, tf.float64)

            # TF Conv2D
            s = time.time()
            tf_Sigmoid_16 = tf_SigmoidWithDiffDype('float16')
            tf_Sigmoid_32 = tf_SigmoidWithDiffDype('float32')
            tf_Sigmoid_64 = tf_SigmoidWithDiffDype('float64')


            out_16_16_2 = tf_Sigmoid_16(x_16).numpy().astype(np.float64)
            out_32_32_2 = tf_Sigmoid_32(x_32).numpy().astype(np.float64)
            out_64_64 = tf_Sigmoid_64(x_64)

            diff1 = np.mean(np.abs(out_16_16_2-out_64_64))  # 低精度到高精度
            diff2 = np.mean(np.abs(out_32_32_2-out_64_64))  # 低精度到高精度

            e = time.time()

            res.append(diff1)
            res.append(diff2)
            res.append(e - s)


            for n in out_64_64.numpy().ravel():
                if math.isnan(n):
                    res.append("NAN")
                    break
            csv_writer.writerow(res)

            if max(res[1:3]) > tmp1:
                index1 = j
                tmp1 = max(res[1:3])

        h_error1 = max(h_error1, tmp1)
        info.append(tmp1)
        info.append(h_error1)
        info.append(index1)

        csv_writer1.writerow(info)

    out.close()
    out1.close()
    
if __name__ == '__main__':

    getDataforTfWihthG("/home/ise/opTest/data/timeflow2/tf_gpu_2.3.1/sigmoid.csv",
                       "/home/ise/opTest/data/timeflow2/tf_gpu_2.3.1/sigmoid_count.csv")
