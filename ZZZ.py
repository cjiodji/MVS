import glob
import os
from cal_para import get_gx, get_gx_dir
import gdal

ref_txt = r"C:\Users\zpl\Desktop\xuzhou3_n\test03\data\cam\30518.txt"
a = r"C:\Users\zpl\Desktop\demo\crop\030518"
out_dir = r"C:\Users\zpl\Desktop\demo\crop\cams\030518"
files = glob.glob(a + "/*.tif")
for file in files:
    fn = os.path.basename(file).replace(".tif", ".txt")
    out_txt = os.path.join(out_dir, fn)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    ds = gdal.Open(file)
    tran = ds.GetGeoTransform()
    xoff = tran[0]
    yoff = tran[3]
    dx = abs(xoff)
    dy = abs(17310 - yoff)
    get_gx(ref_txt, dx, dy, out_txt)
