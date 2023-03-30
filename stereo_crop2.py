import os
import glob
from osgeo import gdal
import sys
import ColinearCondition as CC
from cal_para import *


def crop_image(image_file, sample_image_file,
               xoff, yoff, xsize, ysize,
               compress_image, scale, resample_mode,
               outsize_x, outsize_y, ot="Float32"):
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


def aa(in_dir, img_file1, img_file2, img_file3, label_file, out_dir):
    files = glob.glob(in_dir + "/*.tif")
    dem = r"C:\Users\zpl\Desktop\test\DEM\Xuzhou_DEM_03m.tif"
    cam1 = r"C:\Users\zpl\Desktop\test\image\029519.cam"
    cam2 = r"C:\Users\zpl\Desktop\test\image\030518.cam"
    cam3 = r"C:\Users\zpl\Desktop\test\image\030519.cam"
    cal = CC.ConverXYZ2Imge()
    cal.set_elevation_path(dem)
    for file in files:
        fn = os.path.basename(file)
        ds = gdal.Open(file)

        r = 0.3
        x_size = ds.RasterXSize  # 行数
        y_size = ds.RasterYSize  # 列数
        tran = ds.GetGeoTransform()
        x_min = tran[0]
        y_max = tran[3]
        x_max = x_min + x_size * r
        y_min = y_max - y_size * r
        # 投影
        cal.set_camera_path(cam1)
        out1 = cal.convert_xyzcorner_2_img([x_min, x_max, x_min, x_max], [y_max, y_max, y_min, y_min])
        x1_min = min([out1[0].x, out1[1].x, out1[2].x, out1[3].x])
        x1_max = max([out1[0].x, out1[1].x, out1[2].x, out1[3].x])
        y1_min = min([out1[0].y, out1[1].y, out1[2].y, out1[3].y])
        y1_max = max([out1[0].y, out1[1].y, out1[2].y, out1[3].y])
        x1off = x1_min
        y1off = y1_min
        w1 = x1_max - x1_min
        h1 = y1_max - y1_min
        ref_txt = r"C:\Users\zpl\Desktop\test\cam\29519.txt"
        out_txt = r"C:\Users\zpl\Desktop\test\out\cams\29519/" + fn.split(".")[0] + ".txt"
        get_gx(ref_txt, x1off, y1off, out_txt)

        cal.set_camera_path(cam2)
        out2 = cal.convert_xyzcorner_2_img([x_min, x_max, x_min, x_max], [y_max, y_max, y_min, y_max])
        x2_min = min([out2[0].x, out2[1].x, out2[2].x, out2[3].x])
        x2_max = max([out2[0].x, out2[1].x, out2[2].x, out2[3].x])
        y2_min = min([out2[0].y, out2[1].y, out2[2].y, out2[3].y])
        y2_max = max([out2[0].y, out2[1].y, out2[2].y, out2[3].y])
        x2off = x2_min
        y2off = y2_min
        w2 = x2_max - x2_min
        h2 = y2_max - y2_min
        ref_txt = r"C:\Users\zpl\Desktop\test\cam\30518.txt"
        out_txt = r"C:\Users\zpl\Desktop\test\out\cams\30518/" + fn.split(".")[0] + ".txt"
        get_gx(ref_txt, x2off, y2off, out_txt)

        cal.set_camera_path(cam3)
        out3 = cal.convert_xyzcorner_2_img([x_min, x_max, x_min, x_max], [y_max, y_max, y_min, y_max])
        x3_min = min([out3[0].x, out3[1].x, out3[2].x, out3[3].x])
        x3_max = max([out3[0].x, out3[1].x, out3[2].x, out3[3].x])
        y3_min = min([out3[0].y, out3[1].y, out3[2].y, out3[3].y])
        y3_max = max([out3[0].y, out3[1].y, out3[2].y, out3[3].y])

        x3off = x3_min
        y3off = y3_min
        w3 = x3_max - x3_min
        h3 = y3_max - y3_min

        x3_l = x3_min - 863
        y3_l = y3_min - 477
        ref_txt = r"C:\Users\zpl\Desktop\test\cam\30519.txt"
        out_txt = r"C:\Users\zpl\Desktop\test\out\cams\30519/" + fn.split(".")[0] + ".txt"
        get_gx(ref_txt, x3off, y3off, out_txt)

        f1 = os.path.basename(img_file1).split(".")[0]
        f2 = os.path.basename(img_file2).split(".")[0]
        f3 = os.path.basename(img_file3).split(".")[0]
        out_dir1 = os.path.join(out_dir, "image", f1)
        out_dir2 = os.path.join(out_dir, "image", f2)
        out_dir3 = os.path.join(out_dir, "image", f3)
        out_label = os.path.join(out_dir, "label", f3)
        if not os.path.exists(out_dir1):
            os.makedirs(out_dir1)
        if not os.path.exists(out_dir2):
            os.makedirs(out_dir2)
        if not os.path.exists(out_dir3):
            os.makedirs(out_dir3)
        if not os.path.exists(out_label):
            os.makedirs(out_label)
        out_file1 = os.path.join(out_dir1, fn)
        out_file2 = os.path.join(out_dir2, fn)
        out_file3 = os.path.join(out_dir3, fn)
        out_label = os.path.join(out_label, fn)
        crop_image(img_file1, out_file1, x1off, y1off, w1, h1, True, 0, 'nearest', 768, 384, ot="Byte")
        crop_image(img_file2, out_file2, x2off, y2off, w2, h2, True, 0, 'nearest', 768, 384, ot="Byte")
        crop_image(img_file3, out_file3, x3off, y3off, w3, h3, True, 0, 'nearest', 768, 384, ot="Byte")
        # crop_image(label_file, out_label, x3_l, y3_l, w3, h3, True, 0, 'nearest', 768, 384, ot="Float32")
        # sys.exit()


if __name__ == '__main__':
    in_dir = r"C:\Users\zpl\Desktop\xuzhou\test02\data\building_label\test02.shp"
    img_file1 = r"C:\Users\zpl\Desktop\test\image\029519.tif"
    img_file2 = r"C:\Users\zpl\Desktop\test\image\030518.tif"
    img_file3 = r"C:\Users\zpl\Desktop\test\image\030519.tif"
    label_file = r"C:\Users\zpl\Desktop\test\label\30519.tif"
    out_dir = r"C:\Users\zpl\Desktop\test\out"
    aa(in_dir, img_file1, img_file2, img_file3, label_file, out_dir)
