import ColinearCondition as CC
from osgeo import gdal
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
    dsm = r"E:\ChenLab\Xuzhou-Aerial\DSM-CC\DSM_merge.tif"
    cam1 = r"C:\Users\zpl\Desktop\xuzhou3_n\test03\data\image\029518.cam"
    cam2 = r"C:\Users\zpl\Desktop\xuzhou3_n\test03\data\image\030517.cam"
    cam3 = r"C:\Users\zpl\Desktop\xuzhou3_n\test03\data\image\030518.cam"
    cal = CC.ConverXYZ2Imge()
    cal.set_elevation_path(dsm)
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
        out1 = cal.convert_xyzcorner_2_img([x_min], [y_max])
        x1off = out1[0].x
        y1off = out1[0].y
        w = 768
        h = 384
        ref_txt = r"C:\Users\zpl\Desktop\xuzhou3_n\test03\data\cam\29518.txt"
        out_txt = r"C:\Users\zpl\Desktop\aa\test03\data\out\cams\29518/" + fn.split(".")[0] + ".txt"
        # get_gx(ref_txt, x1off, y1off, out_txt)

        cal.set_camera_path(cam2)
        out2 = cal.convert_xyzcorner_2_img([x_min], [y_max])
        x2off = out2[0].x
        y2off = out2[0].y
        ref_txt = r"C:\Users\zpl\Desktop\xuzhou3_n\test04\data\cam\30516.txt"
        out_txt = r"C:\Users\zpl\Desktop\aa\test04\data\out\cams\30516/" + fn.split(".")[0] + ".txt"
        # get_gx(ref_txt, x2off, y2off, out_txt)

        cal.set_camera_path(cam3)
        out3 = cal.convert_xyzcorner_2_img([x_min], [y_max])
        x3off = out3[0].x
        y3off = out3[0].y
        x3_l = out3[0].x - 4502
        y3_l = out3[0].y - 1
        ref_txt = r"C:\Users\zpl\Desktop\xuzhou3_n\test04\data\cam\30517.txt"
        out_txt = r"C:\Users\zpl\Desktop\aa\test04\data\out\cams\30517/" + fn.split(".")[0] + ".txt"
        # get_gx(ref_txt, x3off, y3off, out_txt)

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
        crop_image(img_file1, out_file1, x1off, y1off, w, h, True, 1, 'nearest', 768, 384, ot="Byte")
        crop_image(img_file2, out_file2, x2off, y2off, w, h, True, 1, 'nearest', 768, 384, ot="Byte")
        crop_image(img_file3, out_file3, x3off, y3off, w, h, True, 1, 'nearest', 768, 384, ot="Byte")
        # crop_image(label_file, out_label, x3_l, y3_l, w, h, True, 1, 'nearest', 768, 384, ot="Float32")
        # sys.exit()


if __name__ == '__main__':
    in_dir = r"C:\Users\zpl\Desktop\xuzhou3_n\test03\data\crop\DOM_shp"
    img_file1 = r"C:\Users\zpl\Desktop\xuzhou3_n\test03\data\mask_all\029518.tif"
    img_file2 = r"C:\Users\zpl\Desktop\xuzhou3_n\test03\data\mask_all\030517.tif"
    img_file3 = r"C:\Users\zpl\Desktop\xuzhou3_n\test03\data\mask_all\030518.tif"
    label_file = r"C:\Users\zpl\Desktop\xuzhou3_n\test04\data\label\30517.tif"
    out_dir = r"C:\Users\zpl\Desktop\aa\test04\data\out\mask_all"
    aa(in_dir, img_file1, img_file2, img_file3, label_file, out_dir)
