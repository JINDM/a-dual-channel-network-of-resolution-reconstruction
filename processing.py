from osgeo import gdal
import cv2
import numpy as np


def _downsample(img, dsize, interpolation=cv2.INTER_CUBIC, ksize=(7, 7)):    #
    blur = cv2.GaussianBlur(img, ksize, 0)
    downsampled = cv2.resize(blur, dsize, interpolation=interpolation)
    return downsampled


def _upsample(img, dsize, interpolation=cv2.INTER_CUBIC):
    upsampled = cv2.resize(img, dsize, interpolation=interpolation)
    return upsampled


def _high_pass_filter(img, ksize=(5, 5)):
    blur = cv2.blur(img, ksize)
    high_pass_filtered = img - blur
    return high_pass_filtered


def prepare_ms(ms_dir, scale):
    # 打开ms
    global input_data, res_data
    ds2 = gdal.Open(ms_dir)
    rows1 = ds2.RasterYSize
    cols1 = ds2.RasterXSize  # 行列
    ms = ds2.ReadAsArray(0, 0, rows1, cols1)  # 将数据写成数组，对应栅格矩阵
    ms_size = ms[0].shape
    if scale == 1:
        down_shape = tuple(np.array(ms_size) // 2)
        down_ms = [_downsample(arr, down_shape) for arr in ms]
        h_d_ms = [_high_pass_filter(arr) for arr in down_ms]
        input_data = np.stack([_upsample(arr, ms_size) for arr in h_d_ms])
        input_data = input_data.T
        res_data = np.stack([_upsample(arr, ms_size) for arr in down_ms])
        res_data = res_data.T
    elif scale == 2:
        down_shape = tuple(np.array(ms_size) // 2)
        down_ms = [_downsample(arr, down_shape) for arr in ms]
        h_d_ms = [_high_pass_filter(arr) for arr in down_ms]
        input_data = np.stack([_upsample(arr, ms_size) for arr in h_d_ms])
        input_data = input_data.T
        res_data = np.stack([_upsample(arr, ms_size) for arr in down_ms])
        res_data = res_data.T
    elif scale == 4:
        down_shape = tuple(np.array(ms_size) // 4)
        down_ms = [_downsample(arr, down_shape) for arr in ms]
        h_d_ms = [_high_pass_filter(arr) for arr in down_ms]
        input_data = np.stack([_upsample(arr, ms_size) for arr in h_d_ms])
        input_data = input_data.T
        res_data = np.stack([_upsample(arr, ms_size) for arr in down_ms])
        res_data = res_data.T
    label_data = ms.T
    return input_data, res_data, label_data


def prepare_ms_pan(pan_dir, ms_dir, scale):
    # 打开pan
    global input_data, res_data, label_data
    ds1 = gdal.Open(pan_dir)
    rows1 = ds1.RasterYSize
    cols1 = ds1.RasterXSize  # 行列
    pan = ds1.ReadAsArray(0, 0, rows1, cols1)  # 将数据写成数组，对应栅格矩阵
    # 打开ms
    ds2 = gdal.Open(ms_dir)
    rows2 = ds2.RasterYSize
    cols2 = ds2.RasterXSize  # 行列
    ms = ds2.ReadAsArray(0, 0, rows2, cols2)     # 将数据写成数组，对应栅格矩阵
    if scale == 1:
        _pan_size = pan.shape   # 100
        _ms_size = ms[0].shape  # 100
        pan_down_shape = tuple(np.array(_pan_size) // 2)  # 50
        ms_down_shape = tuple(np.array(_ms_size) // 2)
        down_pan = _downsample(pan, pan_down_shape)                           # 50
        down_ms = [_downsample(arr, ms_down_shape) for arr in ms]             # 50
        h_d_pan = _high_pass_filter(down_pan)
        h_d_ms = [_high_pass_filter(arr) for arr in down_ms]
        up_down_ms = np.stack([_upsample(arr, _ms_size) for arr in h_d_ms])
        up_down_pan = _upsample(h_d_pan, _pan_size)
        input_data = np.vstack([up_down_pan[np.newaxis, :, :], up_down_ms])
        input_data = input_data.T
        res_data = np.stack([_upsample(arr, _ms_size) for arr in down_ms])
        res_data = res_data.T
    elif scale == 2:
        _pan_size = pan.shape  # 100
        _ms_size = ms[0].shape  # 50
        pan_down_shape = tuple(np.array(_pan_size) // 2)  # 50
        ms_down_shape = tuple(np.array(_ms_size) // 2)    # 25
        down_pan = _downsample(pan, pan_down_shape)
        down_ms = [_downsample(arr, ms_down_shape) for arr in ms]
        h_d_pan = _high_pass_filter(down_pan)
        h_d_ms = [_high_pass_filter(arr) for arr in down_ms]
        up_down_ms = np.stack([_upsample(arr, pan_down_shape) for arr in h_d_ms])
        input_data = np.vstack([h_d_pan[np.newaxis, :, :], up_down_ms])
        input_data = input_data.T
        res_data = np.stack([_upsample(arr, _ms_size) for arr in down_ms])
        res_data = res_data.T
    elif scale == 4:
        _pan_size = pan.shape  # 256
        _ms_size = ms[0].shape  # 64
        pan_down_shape = tuple(np.array(_pan_size) // 4)  # 64
        ms_down_shape = tuple(np.array(_ms_size) // 4)    # 16
        down_pan = _downsample(pan, pan_down_shape)
        down_ms = [_downsample(arr, ms_down_shape) for arr in ms]
        h_d_pan = _high_pass_filter(down_pan)
        h_d_ms = [_high_pass_filter(arr) for arr in down_ms]
        up_down_ms = np.stack([_upsample(arr, pan_down_shape) for arr in h_d_ms])
        input_data = np.vstack([h_d_pan[np.newaxis, :, :], up_down_ms])
        input_data = input_data.T
        res_data = np.stack([_upsample(arr, _ms_size) for arr in down_ms])
        res_data = res_data.T
    label_data = ms.T
    return input_data, res_data, label_data


def prepare_ms_pan_1(pan_dir, ms_dir, scale):
    # 打开pan
    global input_data, res_data, label_data
    ds1 = gdal.Open(pan_dir)
    rows1 = ds1.RasterYSize
    cols1 = ds1.RasterXSize  # 行列
    pan = ds1.ReadAsArray(0, 0, rows1, cols1)  # 将数据写成数组，对应栅格矩阵
    # 打开ms
    ds2 = gdal.Open(ms_dir)
    rows2 = ds2.RasterYSize
    cols2 = ds2.RasterXSize  # 行列
    ms = ds2.ReadAsArray(0, 0, rows2, cols2)     # 将数据写成数组，对应栅格矩阵
    _pan_size = pan.shape  # 128
    _ms_size = ms[0].shape  # 64
    pan_down_shape = tuple(np.array(_pan_size) // 4)  # 64
    ms_down_shape = tuple(np.array(_ms_size) // 4)    # 32
    down_pan = _downsample(pan, pan_down_shape)
    down_ms = [_downsample(arr, ms_down_shape) for arr in ms]
    input_data = np.reshape(down_pan, [cols2, cols2, 1])
    input_data = input_data.T
    input_data = np.reshape(down_pan, [cols2, cols2, 1])
    res_data = np.stack([_upsample(arr, _ms_size) for arr in down_ms])
    res_data = res_data.T
    label_data = ms.T
    return input_data, res_data, label_data
