import glob
import os
import gdal


def get_xy(image_file, label_file):
    ds = gdal.Open(image_file)
    tran = ds.GetGeoTransform()
    x0 = tran[0]
    y0 = tran[3]
    ds2 = gdal.Open(label_file)
    tran2 = ds2.GetGeoTransform()
    x1 = tran2[0]
    y1 = tran2[3]
    print(abs(x1 - x0), abs(y1 - y0))


def get_gx(ref_file, xoff, yoff, out_txt):
    f = open(ref_file, "r")
    ref_txt = f.readlines()
    x0 = ref_txt[6].split(" ")[1]
    y0 = ref_txt[6].split(" ")[2]
    with open(out_txt, "w") as f:
        for i in range(6):
            f.write(ref_txt[i])
        f.write("11748.353500 %.2f %.2f\n\n" % (float(x0) - xoff, float(y0) - yoff))
        f.write(ref_txt[8])


def get_gx2(in_file, ref_txt, out_txt):
    fn = os.path.basename(in_file).split(".")[0].split("_")[-1]
    x = fn[2:4]
    y = fn[4:6]
    x0 = ref_txt[6].split(" ")[1]
    y0 = ref_txt[6].split(" ")[2]
    with open(out_txt, "w") as f:
        for i in range(6):
            f.write(ref_txt[i])
        f.write("11748.353500 %s %s\n\n" % ((float(x0) - int(x) * 192), (float(y0) - int(y) * 96)))
        f.write(ref_txt[8])


def get_gx_dir(in_dir, ref_file, out_dir):
    f = open(ref_file, "r")
    ref_txt = f.readlines()
    f.close()
    # print(ref_txt)
    in_files = glob.glob(in_dir + "/*.png")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for in_file in in_files:
        fn = os.path.basename(in_file).split(".")[0]
        out_txt = os.path.join(out_dir, fn + ".txt")
        get_gx2(in_file, ref_txt, out_txt)


if __name__ == '__main__':
    in_file = r"C:\Users\zpl\Desktop\xuzhou3_n\test04\004\crop\2\image"
    ref_file = r"C:\Users\zpl\Desktop\MVS\dataset\xuzhou3\004\cams\30517.txt"
    out_dir = r"C:\Users\zpl\Desktop\MVS\dataset\xuzhou3\004\crop\2\cams"
    img_file = r"E:\ChenLab\Xuzhou-Aerial\Images\030517.tif"
    label_file = r"C:\Users\zpl\Desktop\xxx\30517\30517.tif"
    get_xy(img_file, label_file)
