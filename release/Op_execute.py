import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import csv
import time
import math
import numpy as np
from seed import seed_mutation

def input_withDiffDype(x,dtype,opname):
    if opname == 'conv2d' or opname == 'MaxPooling2D':
        return tf.convert_to_tensor(x.transpose((0, 2, 3, 1)), dtype=dtype)
    else:
        return tf.convert_to_tensor(x, dtype=dtype)

def tf_opWithDiffDype(dtype, opInfo):
    opname = opInfo["OpName"]
    if opname == 'conv2d':
        filter = opInfo["filters"]
        kernel_size = opInfo['kernel_size']
        strides = tuple(opInfo['strides'])
        padding = opInfo['padding']
        return layers.Conv2D(
            filter, [kernel_size,kernel_size], strides=strides, padding=padding,
            dtype=dtype,kernel_initializer = keras.initializers.Constant(value=0.5)
        )
    elif opname == 'BatchNormalization':
        momentum = opInfo['momentum']
        epsilon = opInfo['epsilon']
        return tf.keras.layers.BatchNormalization(
            axis=-1, momentum=momentum, epsilon=epsilon, center=True, scale=True,
            beta_initializer='zeros', gamma_initializer='ones',
            moving_mean_initializer='zeros', moving_variance_initializer='ones', dtype=dtype
        )
    elif opname == 'MaxPooling2D':
        pool_size = opInfo['pool_size']
        strides = opInfo['strides']
        padding = opInfo['padding']
        return layers.MaxPooling2D(
            pool_size, strides, padding=padding, dtype=dtype
        )
    elif opname == 'Relu':
        return layers.ReLU(
            dtype=dtype
        )
    elif opname == 'Sigmoid':
        return tf.keras.layers.Activation(
            'sigmoid', dtype=dtype
        )
    elif opname == 'Tanh':
        return tf.keras.layers.Activation(
            'tanh', dtype=dtype
        )
    elif opname == 'Softmax':
        return layers.Softmax(
            dtype=dtype
        )



def executeOp_Random(corpus, opinfo, rounds):
    opname = opinfo['OpName']
    filename1 = opname+'_diffdata.csv'
    filename2 = opname+'_count.csv'
    out = open(file=filename1, mode="a", newline='')
    out1 = open(file=filename2,mode="a", newline='')

    csv_writer = csv.writer(out)
    csv_writer1 = csv.writer(out1)

    csv_writer.writerow(["No.",  "16_32(16)", "16_64(16)", "32_16(32)", "32_64(32)", "64_16(64)", "64_32(64)",
                         "time1","32_16(16)", "64_16(16)", "16_32(32)", "64_32(32)", "16_64(64)", "32_64(64)","time2","isNaN"])
    csv_writer1.writerow(["No.","当前最大误差(同输入)","全局最大误差(同输入)","引起最大误差的输入编号1","当前最大误差(同算子)","全局最大误差(同算子)","引起最大误差的输入编号2"])
    h_error1=0
    h_error2=0
    for i in range(rounds):
        tmp1=0
        tmp2=0
        index1=0
        index2=0
        info=[]
        info.append(i)
        j=0
        while not corpus.empty():
            x = corpus.get()
            max1, max2 = getMeandiff(x, csv_writer, j, opname, opinfo, "mean")
            if max1>tmp1:
                index1=j
                tmp1=max(max1,tmp1)

            if max2>tmp2:
                index2=j
                tmp2=max(max2,tmp2)

            j+=1
        h_error1=max(h_error1,tmp1)
        h_error2=max(h_error2,tmp2)
        info.append(tmp1)
        info.append(h_error1)
        info.append(index1)
        info.append(tmp2)
        info.append(h_error2)
        info.append(index2)

        csv_writer1.writerow(info)

    out.close()
    out1.close()


def execute_guided(corpus, opinfo, tn, strategy):
    opname = opinfo['OpName']
    if strategy == "meanguided":
        err_threshold = opinfo["MeanErr_threshold"]
    else:
        err_threshold = opinfo["MaxErr_threshold"]

    filename1 = opname + '_diffdata.csv'
    filename2 = opname + '_count.csv'
    out=open(file=filename1,mode='a',newline='')
    csv_writer=csv.writer(out)
    out1 = open(file=filename2, mode="a", newline='')
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
    while not corpus.empty() and j<tn:
        x=corpus.get()
        err=[]
        if strategy == 'meanguided':
            maxe1,maxe2 = getMeandiff(x, csv_writer, j, opname, opinfo, "mean")
            if max(maxe1, maxe2) > err_threshold:
                seed_mutation(x, corpus)
        else :
            maxse, maxe1,maxe2 = getMeandiff(x, csv_writer, j, opname, opinfo, "max")
            if maxse > err_threshold:
                seed_mutation(x, corpus)

        if maxe1>maxine1:
            index1=j
            maxine1=maxe1 #最大误差

        if maxe2>maxine2:
            index2=j
            maxine2=maxe2 #最大误差

        if j%999==0:
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



def getMeandiff(x, csv_writer, j, opname, opinfo, strategy):
    res = []
    res.append(j)
    x_32 = input_withDiffDype(x, tf.float32,opname)
    x_16 = input_withDiffDype(x, tf.float16,opname)
    x_64 = input_withDiffDype(x, tf.float64,opname)

    s = time.time()
    tf_16 = tf_opWithDiffDype('float16', opinfo)
    tf_32 = tf_opWithDiffDype('float32', opinfo)
    tf_64 = tf_opWithDiffDype('float64', opinfo)

    out_16_16_1 = tf_16(x_16).numpy().astype(np.float32)
    out_16_16_2 = tf_16(x_16).numpy().astype(np.float64)
    out_16_32 = tf_32(x_16)
    out_16_64 = tf_64(x_16)

    diff1 = np.mean(np.abs(out_16_32 - out_16_16_1))  # 低精度到高精度
    diff2 = np.mean(np.abs(out_16_64 - out_16_16_2))  # 低精度到高精度

    out_32_32_1 = tf_32(x_32)
    out_32_32_2 = tf_32(x_32).numpy().astype(np.float64)
    out_32_16 = tf_16(x_32).numpy().astype(np.float32)
    out_32_64 = tf_64(x_32)

    diff3 = np.mean(np.abs(out_32_16 - out_32_32_1))  # 高精度到低精度
    diff4 = np.mean(np.abs(out_32_64 - out_32_32_2))  # 低精度到高精度

    out_64_16 = tf_16(x_64).numpy().astype(np.float64)
    out_64_32 = tf_32(x_64).numpy().astype(np.float64)
    out_64_64 = tf_64(x_64)

    diff5 = np.mean(np.abs(out_64_16 - out_64_64)) # 高精度到低精度
    diff6 = np.mean(np.abs(out_64_32 - out_64_64))  # 低精度到高精度
    e = time.time()

    res.append(diff1)
    res.append(diff2)
    res.append(diff3)
    res.append(diff4)
    res.append(diff5)
    res.append(diff6)
    res.append(e-s)

    s = time.time()
    out_16_16 = tf_16(x_16)
    out_32_16_1 = tf_16(x_32)
    out_64_16_1 = tf_16(x_64)
    diff7 = np.mean(np.abs(out_32_16_1 - out_16_16))
    diff8 = np.mean(np.abs(out_64_16_1 - out_16_16))

    out_64_32_1 = tf_32(x_64)
    diff9 = np.mean(np.abs(out_16_32 - out_32_32_1))
    diff10 = np.mean(np.abs(out_64_32_1 - out_32_32_1))

    diff11 = np.mean(np.abs(out_16_64 - out_64_64))
    diff12 = np.mean(np.abs(out_32_64 - out_64_64))

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
        else:
            res.append("NO")

    csv_writer.writerow(res)
    if strategy == "mean":
        return max(res[1:7]), max(res[8:14])
    else:
        maxe=[]
        dif1 = np.max(np.abs(out_16_32 - out_16_16_1))  # 低精度到高精度
        dif2 = np.max(np.abs(out_16_64 - out_16_16_2))  # 低精度到高精度
        dif3 = np.max(np.abs(out_32_16 - out_32_32_1))  # 高精度到低精度
        dif4 = np.max(np.abs(out_32_64 - out_32_32_2))  # 低精度到高精度
        dif5 = np.max(np.abs(out_64_16 - out_64_64))  # 高精度到低精度
        dif6 = np.max(np.abs(out_64_32 - out_64_64))  # 低精度到高精度
        dif7 = np.max(np.abs(out_32_16_1 - out_16_16))
        dif8 = np.max(np.abs(out_64_16_1 - out_16_16))
        dif9 = np.max(np.abs(out_16_32 - out_32_32_1))
        dif10 = np.max(np.abs(out_64_32_1 - out_32_32_1))
        dif11 = np.max(np.abs(out_16_64 - out_64_64))
        dif12 = np.max(np.abs(out_32_64 - out_64_64))
        maxe = max(dif1, dif2, dif3, dif4, dif5, dif6,
                   dif7, dif8, dif9, dif10, dif11, dif12)
        return maxe, max(res[1:7]), max(res[8:14])




