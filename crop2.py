# -*- coding: utf-8 -*-
import argparse
import glob
import os
import gdal
import numpy as np


def crop_image(image_file, sample_image_file,
               xoff, yoff, xsize, ysize,
               compress_image, scale, resample_mode,
               outsize_x, outsize_y, ot="Byte"):
    """
    裁剪单个影像至特定目录，可以将(xsize,ysize)裁切至(outsize_x,outsize_y)大小的切片：
    image_file, 需要被裁剪的影像路径
    sample_image_file, 裁剪输出结果的影像路径
    xoff, 裁切起点的x坐标
    yoff, 裁切起点的y坐标
    xsize, 裁切大小x
    ysize, 裁切大小y
    compress_image, 是否压缩 0 or 1
    ot, 输出影像的类型 Byte UInt16 Int16 UInt32 Int32 Float32 Float64 CInt16 CInt32 CFloat32 CFloat64
    scale, 是否将裁切影像重采样至另外的大小
    resample_mode, 重采样的模式：{nearest (default),bilinear,cubic,cubicspline,lanczos,average,rms,mode}
    outsize_x, 最终输出的切片大小x
    outsize_y, 最终输出的切片大小y
    """
    cmd = ["gdal_translate", image_file, sample_image_file, "-srcwin",
           str(xoff), str(yoff), str(xsize), str(ysize),
           "-ot", ot,
           # "-co", "COMPRESS=DEFLATE", #"-co", "PHOTOMETRIC=YCBCR",
           "-co", "TILED=YES",
           "--config", "GDAL_TIFF_INTERNAL_MASK YES"
           ]
    if compress_image:
        cmd.append("-co")
        cmd.append("COMPRESS=DEFLATE")
    if scale != 1:
        cmd.append("-r")
        cmd.append(resample_mode)
        cmd.append("-outsize")
        cmd.append(str(outsize_x))
        cmd.append(str(outsize_y))
    strcmd = " ".join(cmd)
    os.system(strcmd)


def make_cmd(image_file, sample_image_file,
             xoff, yoff, xsize, ysize,
             compress_image, scale, resample_mode,
             outsize_x, outsize_y, ot="Byte"):
    cmd = ["gdal_translate", image_file, sample_image_file, "-srcwin",
           str(xoff), str(yoff), str(xsize), str(ysize),
           "-ot", ot,
           # "-co", "COMPRESS=DEFLATE", #"-co", "PHOTOMETRIC=YCBCR",
           # "-co", "TILED=YES",
           # "--config", "GDAL_TIFF_INTERNAL_MASK YES"
           ]
    if compress_image:
        cmd.append("-co")
        cmd.append("COMPRESS=DEFLATE")
    if scale != 1:
        cmd.append("-r")
        cmd.append(resample_mode)
        cmd.append("-outsize")
        cmd.append(str(outsize_x))
        cmd.append(str(outsize_y))
    return cmd


def make_sample(input_dir, output_dir,
                sample_width, sample_height, sample_stride, scale,
                crop_all, none_value, make_csv, csv_mode, compress_image, data_type,
                name1, name2, name3, name4):
    """
    将一个目录内，以name1, name2, name3, name4为名的文件夹内的以data_type为类型的文件按照一定大小裁剪至output_dir
    input_dir, 输入路径
    output_dir, 输出路径
    sample_width, 裁切大小x
    sample_height, 裁切大小y
    sample_stride, 裁剪的步长
    scale, 将即将要裁剪的切片按照一定比例重采样至新的大小
    crop_all, 是否裁剪所有影像，因为有的部分会全是无效值
    none_value, 没有数据的地方的值
    make_csv, 是否制作csv记录影像信息
    csv_mode, w	打开一个文件只用于写入。a 打开一个文件用于追加。a+	打开一个文件用于读写。
    compress_image, 压缩文件
    data_type, 文件的后缀 tiff tif
    name1, name2, name3, name4
    """
    outsize_x = sample_width
    outsize_y = sample_height
    sample_width = int(scale * sample_width)
    sample_height = int(scale * sample_height)
    sample_stride = int(scale * sample_stride)

    if name1 != '':
        input_name1_dir = os.path.join(input_dir, name1)
        output_name1_dir = os.path.join(output_dir, name1)
        if not os.path.exists(output_name1_dir):
            os.makedirs(output_name1_dir)
    else:
        print('name1 is None')
        return -1
    if name2 != '':
        input_name2_dir = os.path.join(input_dir, name2)
        output_name2_dir = os.path.join(output_dir, name2)
        if not os.path.exists(output_name2_dir):
            os.makedirs(output_name2_dir)
    if name3 != '':
        input_name3_dir = os.path.join(input_dir, name3)
        output_name3_dir = os.path.join(output_dir, name3)
        if not os.path.exists(output_name3_dir):
            os.makedirs(output_name3_dir)
    if name4 != '':
        input_name4_dir = os.path.join(input_dir, name4)
        output_name4_dir = os.path.join(output_dir, name4)
        if not os.path.exists(output_name4_dir):
            os.makedirs(output_name4_dir)

    if make_csv:
        info_file = open(output_dir + '/indices.csv', csv_mode)
        info_file.write('name,height,width,positive_ratio,mean_value\n')

    for image_file in glob.glob(input_name1_dir + '/*' + data_type):
        image_name = os.path.basename(image_file)
        if name2 != '':
            name2_file = input_name2_dir + "/" + image_name
        if name3 != '':
            name3_file = input_name3_dir + "/" + image_name
        if name4 != '':
            name4_file = input_name4_dir + "/" + image_name

        ds = gdal.Open(image_file)
        xoffs = list(range(0, ds.RasterXSize - sample_width, sample_stride * 2))
        xoffs.append(ds.RasterXSize - sample_width)
        yoffs = list(range(0, ds.RasterYSize - sample_height, sample_stride))
        yoffs.append(ds.RasterYSize - sample_height)
        # print(len(xoffs))
        # print(len(yoffs))
        # sys.exit()

        if ds.RasterXSize < sample_width or ds.RasterYSize < sample_height:
            continue

        for i, xoff in enumerate(xoffs):
            for j, yoff in enumerate(yoffs):
                xsize = min(sample_width, ds.RasterXSize - xoff)
                ysize = min(sample_height, ds.RasterYSize - yoff)
                print('xoff=%d, yoff=%d', xoff, yoff)
                hz = (str(i).zfill(2) + str(j).zfill(2)).zfill(6)
                sample_image_name = os.path.splitext(image_name)[0] + '_' + hz + data_type  # 用于立体匹配
                sample_image_file = output_name1_dir + "/" + sample_image_name

                image_data = ds.ReadAsArray(xoff, yoff, xsize=xsize, ysize=ysize)
                empty_num = np.sum(image_data == none_value)

                # image
                print('making sample: ' + sample_image_name)
                if crop_all or empty_num < image_data.size * 0.8:
                    # name1
                    crop_image(image_file, sample_image_file,
                               xoff, yoff, xsize, ysize,
                               compress_image, scale, 'nearest',
                               outsize_x, outsize_y)
                    if make_csv:
                        mean_value = np.mean(np.reshape(image_data, (image_data.shape[0], -1)))
                    # name2
                    if name2 != '':
                        sample_name2_file = output_name2_dir + "/" + sample_image_name
                        crop_image(name2_file, sample_name2_file,
                                   xoff, yoff, xsize, ysize,
                                   compress_image, scale, 'nearest',
                                   outsize_x, outsize_y)
                        if make_csv:
                            label_ds = gdal.Open(sample_name2_file)
                            labeldata = label_ds.ReadAsArray()
                            positive_ratio = np.sum(labeldata[labeldata > 0]) / float(outsize_x * outsize_y)
                            info_file.write(
                                '%s,%d,%d,%.3f,%d\n' % (sample_image_name, ysize, xsize, positive_ratio, mean_value))
                    # name3
                    if name3 != '':
                        sample_name3_file = output_name3_dir + "/" + sample_image_name
                        crop_image(name3_file, sample_name3_file,
                                   xoff, yoff, xsize, ysize,
                                   False, scale, 'nearest',
                                   outsize_x, outsize_y)
                    # name4
                    if name4 != '':
                        sample_name4_file = output_name4_dir + "/" + sample_image_name
                        crop_image(name4_file, sample_name4_file,
                                   xoff, yoff, xsize, ysize,
                                   compress_image, scale, 'nearest',
                                   outsize_x, outsize_y)
    if make_csv:
        info_file.close()


def make_multiscale_sample(input_dir, output_dir,
                           sample_width, sample_height, sample_stride, scale_list,
                           crop_all, none_value, make_csv, compress_image, data_type,
                           name1, name2, name3, name4):
    for i, scale in enumerate(scale_list):
        make_sample(input_dir, output_dir,
                    sample_width, sample_height, sample_stride, scale,
                    crop_all, none_value, make_csv, 'w' if i == 0 else 'a', compress_image, data_type,
                    name1, name2, name3, name4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample Generator')
    parser.add_argument('-input_dir', type=str, default='E:\\BMDownload\\永定区_卫图',
                        help='working directory with data prepared')
    parser.add_argument('-output_dir', type=str, default='E:\\BMDownload\\永定区_卫图\\crop',
                        help='sample checkpoint path')
    parser.add_argument('-sample_width', type=int, default=512,
                        help='sample_width of checkpoint data')
    parser.add_argument('-sample_height', type=int, default=512,
                        help='sample_height of checkpoint data')
    parser.add_argument('-sample_stride', type=int, default=256,
                        help='sample_stride of checkpoint data')
    parser.add_argument('-scale_list', type=int, default=[1],
                        help='')
    parser.add_argument('-crop_all', type=int, default=False,
                        help='')
    parser.add_argument('-none_value', type=int, default=0,
                        help='none data value')
    parser.add_argument('-make_csv', type=int, default=True,
                        help='')
    parser.add_argument('-compress_image', type=int, default=True,
                        help='')
    parser.add_argument('-data_type', type=str, default='.tif',
                        help='')
    parser.add_argument('-name1', type=str, default='image',
                        help='file name1')
    parser.add_argument('-name2', type=str, default='',
                        help='file name2')
    parser.add_argument('-name3', type=str, default='',
                        help='file name3')
    parser.add_argument('-name4', type=str, default='',
                        help='file name4')
    args = parser.parse_args()

    input_dir = r"C:\Users\zpl\Desktop\a"
    output_dir = r"C:\Users\zpl\Desktop\a\crop"

    make_sample(input_dir, output_dir,
                768, 384, 96, 1,
                False, 0, True, 'a+', True, '.tif',
                'img', '', '', '')

    # input_dir2 = r"C:\Users\zpl\Desktop\MVS\dataset\xuzhou3\004\clip\1"
    # output_dir2 = r"C:\Users\zpl\Desktop\MVS\dataset\xuzhou3\004\crop\1"
    #
    # input_dir3 = r"C:\Users\zpl\Desktop\MVS\dataset\xuzhou3\004\clip\2"
    # output_dir3 = r"C:\Users\zpl\Desktop\MVS\dataset\xuzhou3\004\crop\2"

    # make_sample(input_dir, output_dir,
    #             768, 384, 96, 1,
    #             False, 0, True, 'a+', False, '.tif',
    #             'DOM', '', '', '')
    # make_sample(input_dir2, output_dir2,
    #             768, 384, 96, 1,
    #             False, 0, True, 'a+', False, '.png',
    #             'image', '', '', '')
    # make_sample(input_dir3, output_dir3,
    #             768, 384, 96, 1,
    #             False, 0, True, 'a+', False, '.png',
    #             'image', '', '', '')
