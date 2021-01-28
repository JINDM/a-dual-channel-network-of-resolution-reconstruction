import processing
import gdal
import tensorflow as tf
import numpy as np
import os

checkpoint_dir = 'E:/dataset/7-12/checkpoint/old/ms+pan/'
result_path = 'C:/Users/SurveyComputer/Desktop/dataset/test/new/7-12/cropland/result/'
pan_test = 'C:/Users/SurveyComputer/Desktop/dataset/test/new/7-12/cropland/pan/'
ms_test = 'C:/Users/SurveyComputer/Desktop/dataset/test/new/7-12/cropland/ms/'
image_size = 64


# pan+ms
def input_up(pan_dir, ms_dir):
    # 打开pan
    ds1 = gdal.Open(pan_dir)
    rows1 = ds1.RasterYSize
    cols1 = ds1.RasterXSize  # 行列
    pan = ds1.ReadAsArray(0, 0, rows1, cols1)  # 将数据写成数组，对应栅格矩阵
    # 打开ms
    ds2 = gdal.Open(ms_dir)
    rows2 = ds2.RasterYSize
    cols2 = ds2.RasterXSize  # 行列
    im_geotrans = ds2.GetGeoTransform()  # 仿射矩阵
    im_proj = ds2.GetProjection()  # 地图投影信息
    ms = ds2.ReadAsArray(0, 0, rows2, cols2)  # 将数据写成数组，对应栅格矩阵
    _pan_size = pan.shape  # 128
    _ms_size = ms[0].shape   # 32
    pan_down_shape = tuple(np.array(_pan_size) // 2)   # 32
    ms_down_shape = tuple(np.array(_ms_size) // 2)     # 8
    down_pan = processing._downsample(pan, pan_down_shape)
    down_ms = [processing._downsample(arr, ms_down_shape) for arr in ms]
    h_d_pan = processing._high_pass_filter(down_pan)
    h_d_ms = [processing._high_pass_filter(arr) for arr in down_ms]
    up_down_ms = np.stack([processing._upsample(arr, _ms_size) for arr in h_d_ms])
    # up_down_pan = processing._upsample(h_d_pan, _pan_size)
    input_data = np.vstack([h_d_pan[np.newaxis, :, :], up_down_ms])
    input_data = input_data.T
    res_data = np.stack([processing._upsample(arr, _ms_size) for arr in down_ms])
    res_data = res_data.T
    return input_data, res_data, im_geotrans, im_proj


# ms
def input_up2(dir1):
    ds1 = gdal.Open(dir1)
    rows1 = ds1.RasterYSize
    cols1 = ds1.RasterXSize  # 行列
    im_geotrans = ds1.GetGeoTransform()  # 仿射矩阵
    im_proj = ds1.GetProjection()  # 地图投影信息
    ms = ds1.ReadAsArray(0, 0, rows1, cols1)  # 将数据写成数组，对应栅格矩阵
    down_size = ms[0].shape   # 128
    down_shape = tuple(np.array(down_size) // 2)  # 64
    down_ms = [processing._downsample(arr, down_shape) for arr in ms]
    h_d_ms = [processing._high_pass_filter(arr) for arr in down_ms]
    input_data = np.stack([processing._upsample(arr, (image_size, image_size)) for arr in h_d_ms])
    input_data = input_data.T
    res_data = np.stack([processing._upsample(arr, (image_size, image_size)) for arr in down_ms])
    res_data = res_data.T
    return input_data, res_data, im_geotrans, im_proj


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


with tf.Session() as sess:
    files = os.listdir(ms_test)
    for file in files:
        # test_data, test_res, geo, proj = input_up2(ms_test + file)
        test_data, test_res, geo, proj = input_up(pan_test + file, ms_test + file)
        print(test_data.shape)
        test_data = test_data.reshape([1, image_size, image_size, 7])
        test_res = test_res.reshape([1, image_size, image_size, 6])
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        saver = tf.train.import_meta_graph(checkpoint_dir + 'model.ckpt.meta')
        saver.restore(sess, checkpoint_dir + 'model.ckpt')   # .data文件
        graph = tf.get_default_graph()
        x = graph.get_operation_by_name('image').outputs[0]
        y_ = graph.get_operation_by_name('res').outputs[0]
        print('finish loading model!')
        pred = tf.get_collection('network-output')[0]
        result = sess.run(pred, {x: test_data, y_: test_res})
        result = result.squeeze()   # 除去size为1的维度
        result = result.T
        print('dealing:', file, result.shape)
        write_img(result_path + file, proj, geo, result)
        print('-----------')
