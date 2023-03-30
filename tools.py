import glob
import gdal
import laspy
from ParamTransform import *
from utils.raster_io import fastly_save_image


def clip_by_pixel(in_file, out_file):
    """
    以最小的范围裁切,左上角对齐
    """
    ds = gdal.Open(in_file)
    img = ds.ReadAsArray()
    if img.ndim == 2:
        img = img[0:4030, 0:6206]
        fastly_save_image(img, out_file, gdal_type=gdal.GDT_Float32)
    if img.ndim == 3:
        img = img[:, 0:4030, 0:6206]
        fastly_save_image(img, out_file, gdal_type=gdal.GDT_Byte)


def run_dir(in_dir, out_dir):
    """
    批量操作
    """
    in_files = glob.glob(in_dir + "/*.tif")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for in_file in in_files:
        fn = os.path.basename(in_file).replace(".tif", ".png")
        out_file = os.path.join(out_dir, fn)
        clip_by_pixel(in_file, out_file)


def _rename(in_dir):
    file_list = os.listdir(in_dir)
    for file in file_list:
        fn = os.path.join(in_dir, file)
        nn = os.path.join(in_dir, fn.split("_")[-1])
        os.rename(fn, nn)


def label_transform(in_file, out_file):
    """
    输入 tif
    输出 经过处理的png
    """
    ds = gdal.Open(in_file)
    img = ds.ReadAsArray()
    img[img == 0] = -9999
    img = -img
    # img = (-img - 3000) * 64
    fastly_save_image(img, out_file, ref_file=in_file, gdal_type=gdal.GDT_Float32)


def las_transform(in_las, out_las):
    # 三维变换
    cams_pd = pd.read_csv(r"C:\Users\zpl\Desktop\MVS\dataset\xuzhou3\XZTotalCam.txt", sep=" ", index_col=0)
    las = laspy.read(in_las)
    cams = cams_pd.to_numpy()
    # 030520
    cam = cams[17][1:]
    rotation = RotationMatrix(cam[3], cam[4], cam[5])
    t = np.array([cam[0], cam[1], cam[2]]).reshape(-1, 1)
    pointcloud = las
    pointcloud = np.vstack((pointcloud.x, pointcloud.y, pointcloud.z))
    pointcloud -= t
    pointcloud = np.linalg.inv(rotation) @ pointcloud
    pointcloud = pointcloud[:3, :].transpose()
    header = laspy.LasHeader(point_format=3, version="1.2")
    las1 = laspy.LasData(header)
    # las1 = laspy.LasData(laspy.header())
    las1.x = pointcloud[:, 0]
    las1.y = pointcloud[:, 1]
    las1.z = pointcloud[:, 2]
    las1.write(out_las)


if __name__ == '__main__':
    a = r"C:\Users\zpl\Desktop\whu"
    i = r"C:\Users\zpl\Desktop\MVS\dataset\xuzhou3\test02\label\029519.tif"
    o = r"C:\Users\zpl\Desktop\MVS\dataset\xuzhou3\test02\label\29519.tif"
    il = r"E:\ChenLab\Xuzhou-Aerial\DSM-LiDAR\Xuzhou_lidar_AOI.las"
    ol = r"C:\Users\zpl\Desktop\MVS\dataset\xuzhou3\test04\LAS\029517\aoi.las"
    # clip_by_pixel(i, o)
    # label_transform(i, o)
    _rename(a)
    # run_dir(i, o)
    # las_transform(il, ol)
