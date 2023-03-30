import ColinearCondition as CC
import gdal
from stereo_crop import crop_image

dsm = r"E:\ChenLab\Xuzhou-Aerial\DSM-CC\DSM_merge.tif"

cam = r"C:\Users\zpl\Desktop\MVS\dataset/xuzhou3\test03\image\030518.cam"
cal = CC.ConverXYZ2Imge()
cal.set_elevation_path(dsm)
cal.set_camera_path(cam)

dom = r"C:\Users\zpl\Desktop\a\test03a.tif"
ds = gdal.Open(dom)
r = 0.3
x_size = ds.RasterXSize  # 行数
y_size = ds.RasterYSize  # 列数
tran = ds.GetGeoTransform()
x_min = tran[0]
y_max = tran[3]
x_max = x_min + x_size * r
y_min = y_max - y_size * r

out3 = cal.convert_xyzcorner_2_img([x_min, x_max, x_min, x_max], [y_max, y_max, y_min, y_max])
x3_min = min([out3[0].x, out3[1].x, out3[2].x, out3[3].x])
x3_max = max([out3[0].x, out3[1].x, out3[2].x, out3[3].x])
y3_min = min([out3[0].y, out3[1].y, out3[2].y, out3[3].y])
y3_max = max([out3[0].y, out3[1].y, out3[2].y, out3[3].y])
w3 = x3_max - x3_min
h3 = y3_max - y3_min

img = r"C:\Users\zpl\Desktop\MVS\dataset/xuzhou3\test03\image\030518.tif"
out = r"C:\Users\zpl\Desktop\a\03a.tif"
crop_image(img, out, x3_min, y3_min, w3, h3, True, 1, None, 0, 0, ot="Byte")
