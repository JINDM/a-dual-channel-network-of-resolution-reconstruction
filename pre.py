import imageio
from tensorflow.contrib import slim
import os
import tensorflow as tf
import h5py
import numpy as np
from mypannet import canshu as config1
from osgeo import gdal
from mypannet import processing


# def input_setup():
#     print('creating dataset ...')
#     sub_label = []
#     sub_res = []
#     sub_input = []
#     lis = os.listdir(config1.pan_dir)
#     for file1 in lis:
#         print('dealing:', file1)
#         input_data, res_data, label_data = processing.
#         prepare_input(config1.pan_dir + '/' + file1, config1.ms_dir + '/' + file1)
#         print('input_data', input_data.shape)
#         print('label_data', label_data.shape)
#         print('res_data', res_data.shape)
#         sub_input.append(input_data)
#         sub_label.append(label_data)
#         sub_res.append(res_data)
#     arrdata = np.asarray(sub_input)
#     arrres = np.asarray(sub_res)
#     arrlabel = np.asarray(sub_label)
#     make_data(arrdata, arrres, arrlabel)
#     print("dataset is ok")   # 存成h5格式

def input_setup():
    print('creating dataset ...')
    sub_label = []
    sub_res = []
    sub_input = []
    lis = os.listdir(config1.ms_dir)
    for file1 in lis:
        print('dealing:', file1)
        input_data, res_data, label_data = processing.prepare_ms_pan(config1.pan_dir + '/' + file1, config1.ms_dir + '/' + file1, config1.scale)
        # input_data, res_data, label_data = processing.prepare_ms(config1.ms_dir + '/' + file1, config1.scale)
        print('input_data', input_data.shape)
        print('label_data', label_data.shape)
        print('res_data', res_data.shape)
        sub_input.append(input_data)
        sub_label.append(label_data)
        sub_res.append(res_data)
    arrdata = np.asarray(sub_input)
    arrres = np.asarray(sub_res)
    arrlabel = np.asarray(sub_label)
    make_data(arrdata, arrres, arrlabel)
    print("dataset is ok")   # 存成h5格式


def imsave(path, image):
    return imageio.imsave(path, image)


def make_data(data, res, label):
    savepath = config1.savepath
    with h5py.File(savepath, 'w')as hf:
        hf.create_dataset('label', data=label)
        hf.create_dataset('res', data=res)
        hf.create_dataset('data', data=data)


def read_data(path):
    with h5py.File(path, 'r') as hf:    # 读取h5格式数据文件(用于训练或测试)
        data = np.array(hf.get('data'))
        res_data = np.array(hf.get('res'))
        label = np.array(hf.get('label'))
    return data, res_data, label


def resBlock(x, channels=6, kernel_size=[3, 3], scale=1):
    tmp = slim.conv2d(x, channels,kernel_size,activation_fn=None)
    tmp = tf.nn.relu(tmp)
    tmp = slim.conv2d(tmp, channels,kernel_size,activation_fn=None)
    tmp *= scale
    return x + tmp


def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def prepare_data(dataset):
    # 训练
    filenames = os.listdir(dataset)  # 输出dataset下的所有文件名
    print(filenames)
    data_dir = os.path.join(os.getcwd(), dataset)
    print(data_dir)
    # data = glob.glob(os.path.join(data_dir,"*.bmp"))
    # return data


def ComputPSNR(image_target, output):
    mse = tf.reduce_mean(tf.squared_difference(image_target, output))
    PSNR = tf.constant(255 ** 2, dtype=tf.float32) / mse
    PSNR = tf.constant(10, dtype=tf.float32) * log10(PSNR)
    return PSNR


def read_img(filename):
    dataset = gdal.Open(filename)  # 打开文件
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数
    im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
    im_proj = dataset.GetProjection()  # 地图投影信息
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 将数据写成数组，对应栅格矩阵
    if len(im_data.shape) == 3:
        im_bands = im_data.shape[0]
    else:
        im_bands = 1
    del dataset
    return im_proj, im_geotrans, im_data, im_bands


def write_img(filename, im_proj, im_geotrans, im_data):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)
    dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
    dataset.SetProjection(im_proj)  # 写入投影

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
    else:
        for m in range(im_bands):
            dataset.GetRasterBand(m + 1).WriteArray(im_data[m])
    del dataset


def upsample(x, scale=config1.scale, features=64, activation=tf.nn.relu):
    assert scale in [2, 3, 4]
    x = slim.conv2d(x, features, [3, 3], activation_fn=activation)
    if scale == 2:
        ps_features = 3*(scale**2)
        x = slim.conv2d(x, ps_features, [3, 3], activation_fn=activation)
        x = PS(x, 2, color=True)
    elif scale == 3:
        ps_features = 3*(scale**2)
        x = slim.conv2d(x, ps_features, [3, 3], activation_fn=activation)
        x = PS(x, 3, color=True)
    elif scale == 4:
        ps_features = 3*(2**2)
        for i in range(2):
            x = slim.conv2d(x, ps_features, [3, 3], activation_fn=activation)
            x = PS(x, 2, color=True)
    return x


def _phase_shift(I, r):
    bsize, a, b, c = I.get_shape().as_list()
    bsize = tf.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
    X = tf.reshape(I, (bsize, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X],2)  # bsize, b, a*r, r
    X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X],2)  # bsize, a*r, b*r
    return tf.reshape(X, (bsize, a*r, b*r, 1))


def PS(X, r, color=False):
    if color:
        Xc = tf.split(X, 3, 3)
        X = tf.concat([_phase_shift(x, r) for x in Xc], 3)
    else:
        X = _phase_shift(X, r)
    return X
