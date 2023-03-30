import os
import shutil


def _rank(in_dir, out_dir):
    files = os.listdir(in_dir)
    for file in files:
        fn = os.path.basename(file).split(".")[0]
        dir_name1 = fn.split("_")[-1]
        dir_name2 = fn.split("_")[-2]
        dir_name = dir_name2 + "_" + dir_name1
        print(dir_name)
        full_file = os.path.join(in_dir, file)
        out_full_dir = os.path.join(out_dir, dir_name)
        if not os.path.exists(out_full_dir):
            os.makedirs(out_full_dir)
        shutil.copy(full_file, out_full_dir)


if __name__ == '__main__':
    in_dir = r"E:\ChenLab\MVS\demo\image"
    out_dir = r"E:\ChenLab\MVS\demo\stereo"
    _rank(in_dir, out_dir)
