import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import csv
import time
from queue import Queue
import math

a=[]
a1 = 0.0001 * np.ones((2,2), np.float64)
a2 = 0.000001 * np.ones((2,2), np.float64)
a3 = 0.00000001 * np.ones((2,2), np.float64)


def input_withDiffDype(x,dtype):
    return tf.convert_to_tensor(x, dtype=dtype)

def tf_SigmoidWithDiffDype(dtype):
    return tf.keras.layers.Activation(
        'sigmoid', dtype=dtype
)

def createCorpus(n):
    q=Queue()
    for i in range(n):
        x = np.random.randn(2,2)
        q.put(x)
    return q


def Max_guided(corpus, f, g):
    out = open(file=f, mode='a', newline='')
    csv_writer = csv.writer(out)
    out1 = open(file=g, mode="a", newline='')
    csv_writer1 = csv.writer(out1)
    csv_writer.writerow(["No.", "16_32(16)", "16_64(16)", "32_16(32)", "32_64(32)", "64_16(64)", "64_32(64)",
                         "time1", "32_16(16)", "64_16(16)", "16_32(32)", "64_32(32)", "16_64(64)", "32_64(64)", "time2",
                         "isNaN"])
    csv_writer1.writerow(
        ["No.", "当前最大误差(同输入)", "全局最大误差(同输入)", "引起最大误差的输入编号1", "当前最大误差(同算子)", "全局最大误差(同算子)", "引起最大误差的输入编号2"])
    h_error1 = 0
    h_error2 = 0
    maxine1 = 0
    maxine2 = 0
    j = 0
    index1 = 0
    index2 = 0
    while not corpus.empty() and j < 20000:
        x = corpus.get()
        maxse, maxe1, maxe2 = getMaxdiff(x, csv_writer, j)
        # if max(err)>0.0022 and index>0:
        if maxe1 > maxine1:
            index1 = j
            maxine1 = maxe1  # 最大误差

        if maxe2 > maxine2:
            index2 = j
            maxine2 = maxe2  # 最大误差

        if maxse > 0.0003:
            corpus.put(x + a1)
            corpus.put(x + a2)
            corpus.put(x + a3)

        if j % 999 == 0:
            r = []
            h_error1 = max(h_error1, maxine1)
            h_error2 = max(h_error2, maxine2)
            r.append(j // 999)
            r.append(maxine1)
            r.append(h_error1)
            r.append(index1)
            r.append(maxine2)
            r.append(h_error2)
            r.append(index2)
            csv_writer1.writerow(r)

            maxine1 = 0
            maxine2 = 0
            index1 = 0
            index2 = 0
        j += 1
        print(j)

    out.close()
    out1.close()


def getMaxdiff(x, csv_writer, j):
    res = []
    maxe = []
    res.append(j)
    # weights = torch.empty(3, 3, 3, 8)
    # torch.nn.init.constant_(weights, 5e-2)
    # Tensorflow padding behavior. Assuming that kH == kW to keep this simple.

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

    diff1 = np.mean(np.abs(out_16_32 - out_16_16_1))  # 低精度到高精度
    diff2 = np.mean(np.abs(out_16_64 - out_16_16_2))  # 低精度到高精度

    dif1 = np.max(np.abs(out_16_32 - out_16_16_1))  # 低精度到高精度
    dif2 = np.max(np.abs(out_16_64 - out_16_16_2))  # 低精度到高精度

    out_32_32_1 = tf_Sigmoid_32(x_32)
    out_32_32_2 = tf_Sigmoid_32(x_32).numpy().astype(np.float64)
    out_32_16 = tf_Sigmoid_16(x_32).numpy().astype(np.float32)
    out_32_64 = tf_Sigmoid_64(x_32)

    diff3 = np.mean(np.abs(out_32_16 - out_32_32_1))  # 高精度到低精度
    diff4 = np.mean(np.abs(out_32_64 - out_32_32_2))  # 低精度到高精度

    dif3 = np.max(np.abs(out_32_16 - out_32_32_1))  # 高精度到低精度
    dif4 = np.max(np.abs(out_32_64 - out_32_32_2))  # 低精度到高精度

    out_64_16 = tf_Sigmoid_16(x_64).numpy().astype(np.float64)
    out_64_32 = tf_Sigmoid_32(x_64).numpy().astype(np.float64)
    out_64_64 = tf_Sigmoid_64(x_64)

    diff5 = np.mean(np.abs(out_64_16 - out_64_64))  # 高精度到低精度
    diff6 = np.mean(np.abs(out_64_32 - out_64_64))  # 低精度到高精度

    dif5 = np.max(np.abs(out_64_16 - out_64_64))  # 高精度到低精度
    dif6 = np.max(np.abs(out_64_32 - out_64_64))  # 低精度到高精度

    e = time.time()
    res.append(diff1)
    res.append(diff2)
    res.append(diff3)
    res.append(diff4)
    res.append(diff5)
    res.append(diff6)
    res.append(e - s)

    s = time.time()
    out_16_16 = tf_Sigmoid_16(x_16)
    out_32_16_1 = tf_Sigmoid_16(x_32)
    out_64_16_1 = tf_Sigmoid_16(x_64)
    diff7 = np.mean(np.abs(out_32_16_1 - out_16_16))
    diff8 = np.mean(np.abs(out_64_16_1 - out_16_16))

    dif7 = np.max(np.abs(out_32_16_1 - out_16_16))
    dif8 = np.max(np.abs(out_64_16_1 - out_16_16))

    out_64_32_1 = tf_Sigmoid_32(x_64)
    diff9 = np.mean(np.abs(out_16_32 - out_32_32_1))
    diff10 = np.mean(np.abs(out_64_32_1 - out_32_32_1))

    dif9 = np.max(np.abs(out_16_32 - out_32_32_1))
    dif10 = np.max(np.abs(out_64_32_1 - out_32_32_1))

    diff11 = np.mean(np.abs(out_16_64 - out_64_64))
    diff12 = np.mean(np.abs(out_32_64 - out_64_64))

    dif11 = np.max(np.abs(out_16_64 - out_64_64))
    dif12 = np.max(np.abs(out_32_64 - out_64_64))

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

    maxe.append(dif1)
    maxe.append(dif2)
    maxe.append(dif3)
    maxe.append(dif4)
    maxe.append(dif5)
    maxe.append(dif6)
    maxe.append(dif7)
    maxe.append(dif8)
    maxe.append(dif9)
    maxe.append(dif10)
    maxe.append(dif11)
    maxe.append(dif12)

    csv_writer.writerow(res)
    return max(maxe[:]), max(res[1:7]), max(res[8:14])


def Mean_guided(corpus, f, g):
    out = open(file=f, mode='a', newline='')
    csv_writer = csv.writer(out)
    out1 = open(file=g, mode="a", newline='')
    csv_writer1 = csv.writer(out1)
    csv_writer.writerow(["No.", "16_32(16)", "16_64(16)", "32_16(32)", "32_64(32)", "64_16(64)", "64_32(64)",
                         "time1", "32_16(16)", "64_16(16)", "16_32(32)", "64_32(32)", "16_64(64)", "32_64(64)", "time2",
                         "isNaN"])
    csv_writer1.writerow(
        ["No.", "当前最大误差(同输入)", "全局最大误差(同输入)", "引起最大误差的输入编号1", "当前最大误差(同算子)", "全局最大误差(同算子)", "引起最大误差的输入编号2"])
    h_error1 = 0
    h_error2 = 0
    maxine1 = 0
    maxine2 = 0
    j = 0
    index1 = 0
    while not corpus.empty() and j < 20000:
        x = corpus.get()
        maxe1, maxe2 = getMeandiff(x, csv_writer, j)
        if max(maxe1, maxe2) > 1e-4:
            corpus.put(x + a1)
            corpus.put(x + a2)
            corpus.put(x + a3)

        if maxe1 > maxine1:
            index1 = j
            maxine1 = maxe1  # 最大误差

        if maxe2 > maxine2:
            index2 = j
            maxine2 = maxe2  # 最大误差

        if j % 999 == 0:
            r = []
            h_error1 = max(h_error1, maxine1)
            h_error2 = max(h_error2, maxine2)
            r.append(j // 999)
            r.append(maxine1)
            r.append(h_error1)
            r.append(index1)
            r.append(maxine2)
            r.append(h_error2)
            r.append(index2)
            csv_writer1.writerow(r)
            maxine1 = 0
            maxine2 = 0
            index1 = 0
            index2 = 0
        j += 1
        print(j)

    out.close()
    out1.close()


def getMeandiff(x, csv_writer, j):
    res = []
    res.append(j)
    # weights = torch.empty(3, 3, 3, 8)
    # torch.nn.init.constant_(weights, 5e-2)
    # Tensorflow padding behavior. Assuming that kH == kW to keep this simple.

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
    e = time.time()

    res.append(diff1)
    res.append(diff2)
    res.append(diff3)
    res.append(diff4)
    res.append(diff5)
    res.append(diff6)
    res.append(e - s)

    s = time.time()
    out_16_16 = tf_Sigmoid_16(x_16)
    out_32_16_1 = tf_Sigmoid_16(x_32)
    out_64_16_1 = tf_Sigmoid_16(x_64)
    diff7 = np.mean(np.abs(out_32_16_1 - out_16_16))
    diff8 = np.mean(np.abs(out_64_16_1 - out_16_16))

    dif7 = np.max(np.abs(out_32_16_1 - out_16_16))
    dif8 = np.max(np.abs(out_64_16_1 - out_16_16))

    out_64_32_1 = tf_Sigmoid_32(x_64)
    diff9 = np.mean(np.abs(out_16_32 - out_32_32_1))
    diff10 = np.mean(np.abs(out_64_32_1 - out_32_32_1))

    dif9 = np.max(np.abs(out_16_32 - out_32_32_1))
    dif10 = np.max(np.abs(out_64_32_1 - out_32_32_1))

    diff11 = np.mean(np.abs(out_16_64 - out_64_64))
    diff12 = np.mean(np.abs(out_32_64 - out_64_64))

    dif11 = np.max(np.abs(out_16_64 - out_64_64))
    dif12 = np.max(np.abs(out_32_64 - out_64_64))

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
    return max(res[1:7]), max(res[8:14])

if __name__=='__main__':
    corpus = createCorpus(1000)
    # Max_guided(corpus, "E:\Dtype_test\Max_guided2\\tf_cpu_2.4.0\\tf_Sigmoid.csv","E:\Dtype_test\Max_guided2\\tf_cpu_2.4.0\\tf_Sigmoid_count.csv")
    # Mean_guided(corpus,"E:\Dtype_test\Mean_guided2\\tf_cpu_2.4.0\\tf_Sigmoid.csv","E:\Dtype_test\Mean_guided2\\tf_cpu_2.4.0\\tf_Sigmoid_count.csv")

    Max_guided(corpus, "/home/ise/opTest/data/Max_guided2/tf_gpu_2.4.0/sigmoid.csv",
               "/home/ise/opTest/data/Max_guided2/tf_gpu_2.4.0/sigmoid_count.csv")
    Mean_guided(corpus,"/home/ise/opTest/data/Mean_guided2/tf_gpu_2.4.0/sigmoid.csv","/home/ise/opTest/data/Mean_guided2/tf_gpu_2.4.0/sigmoid_count.csv")