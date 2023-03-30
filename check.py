import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import numpy as np
import cv2
from kornia.utils import create_meshgrid
import matplotlib.pyplot as plt


class ArialDataset(Dataset):
    def __init__(self, root_dir,
                 split="train",
                 n_views=3,
                 n_depths=100,
                 interval_scale=0.5,
                 depth_interval=1):

        assert split in ['train', 'val', 'test'], 'split must be "train", "val", "test"!'
        self.image_scale = 1
        self.depth_interval = depth_interval
        self.root_dir = root_dir
        self.split = split
        self.n_views = n_views
        self.n_depths = n_depths
        self.interval_scale = interval_scale
        self.transform = T.Compose([T.ToTensor(), ])
        self.build_metas()

    def build_metas(self):
        split_path = os.path.join(self.root_dir, self.split)
        with open(os.path.join(split_path, "index.txt")) as f:
            scans = [line.rstrip() for line in f.readlines()]
        with open(os.path.join(split_path, "pair.txt")) as f:
            pair = [int(i) for i in f.readline().rstrip().split()]

        view_comb = []
        # select different views as ref_img
        for ref in range(0, pair[0]):
            # the num of reference image
            ref_num = int(pair[(pair[0] + 1) * ref + 1])
            tmp = [ref_num]
            for src in range(self.n_views - 1):
                # selected view image
                src_num = pair[(pair[0] + 1) * ref + 3 + src]
                tmp.append(src_num)
            view_comb.append(tmp)

        self.metas = []
        for scan in scans:
            for comb in view_comb:
                for id in os.listdir(os.path.join(os.path.join(self.root_dir, self.split), f"Cams/{scan}/0")):
                    # comb [ref, srcs]
                    self.metas.append([scan, comb, id.split(".")[0]])

    def read_cam_file(self, camfile_path, scale_x, scale_y):
        with open(camfile_path) as f:
            lines = [line.rstrip() for line in f.readlines()]

        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        f, x0, y0 = np.fromstring(lines[6], dtype=np.float32, sep=' ')
        depth_min, depth_max, depth_interval = np.fromstring(lines[8], dtype=np.float32, sep=' ')

        extrinsics = extrinsics.reshape((4, 4))

        intrinsics = np.zeros((3, 3))
        intrinsics[0][0] = f * self.image_scale * scale_x
        intrinsics[1][1] = f * self.image_scale * scale_y
        intrinsics[0][2] = x0 * self.image_scale * scale_x
        intrinsics[1][2] = y0 * self.image_scale * scale_y
        intrinsics[2][2] = 1

        return intrinsics, extrinsics, depth_min, depth_max, depth_interval

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        scan, comb, sample = self.metas[idx]
        imgs = []
        proj_mats = []
        image_path = os.path.join(self.root_dir, self.split, "Images", scan)
        cam_path = os.path.join(self.root_dir, self.split, "Cams", scan)

        for i, vid in enumerate(comb):
            img_filename = os.path.join(image_path, str(vid), f'{sample}.tif')
            cam_filename = os.path.join(cam_path, str(vid), f'{sample}.txt')
            # img = np.array(cv2.cvtColor(cv2.imread(img_filename), cv2.COLOR_BGR2RGB), dtype=np.float32)
            img = cv2.imread(img_filename).astype("float32")

            # scale_x = img.shape[1] / 200
            # scale_y = img.shape[0] / 60
            #
            # img = cv2.resize(img, (200, 60),
            #                  interpolation=cv2.INTER_NEAREST)

            imgs += [self.transform(img)]

            # reference view
            intrinsics, extrinsics, depth_min, depth_max, depth_interval = self.read_cam_file(cam_filename, scale_x=1,
                                                                                              scale_y=1)
            proj_mat = extrinsics
            proj_mat[:3, :4] = torch.FloatTensor(intrinsics @ proj_mat[:3, :4])
            proj_mat = torch.FloatTensor(proj_mat)
            depth_min = 3360
            if i == 0:
                depth_values = torch.arange(depth_min,
                                            self.depth_interval * self.n_depths + depth_min,
                                            self.depth_interval,
                                            dtype=torch.float32)

                proj_mats += [torch.inverse(proj_mat)]
            else:
                proj_mats += [proj_mat]
        imgs = torch.stack(imgs)
        proj_mats = torch.stack(proj_mats)
        return imgs, proj_mats, depth_values


def homo_warp(src_feat, src_proj, ref_proj_inv, depth_values):
    # src_feat: (B, F, H, W)
    # src_proj: (B, 4, 4)
    # ref_proj_inv: (B, 4, 4)
    # depth_values: (B, D)
    # out: (B, C, D, H, W)
    B, C, H, W = src_feat.shape
    # NOTE 所有的深度插值都是一样的
    D = depth_values.shape[1]
    device = src_feat.device
    dtype = src_feat.dtype

    transform = src_proj @ ref_proj_inv  # 矩阵相乘=a.dot(b)
    R = transform[:, :3, :3]  # (B, 3, 3)
    T = transform[:, :3, 3:]  # (B, 3, 1)
    # create grid from the ref frame
    ref_grid = create_meshgrid(H, W, normalized_coordinates=False)  # (1, H, W, 2)
    ref_grid = ref_grid.to(device).to(dtype)
    ref_grid = ref_grid.permute(0, 3, 1, 2)  # (1, 2, H, W)

    ref_grid = ref_grid.reshape(1, 2, H * W)  # (1, 2, H*W)
    ref_grid = ref_grid.expand(B, -1, -1)  # (B, 2, H*W)
    # ref, XYZ 3D点 (x,y,1) 有 H*W个
    ref_grid = torch.cat((ref_grid, torch.ones_like(ref_grid[:, :1])), 1)  # (B, 3, H*W)

    # 变成一个立体锥
    ref_grid_d = ref_grid.unsqueeze(2) * depth_values.view(B, 1, D, 1)  # (B, 3, D, H*W)
    # 拉成一条 点
    ref_grid_d = ref_grid_d.view(B, 3, D * H * W)
    # NOTE 投影
    src_grid_d = R @ ref_grid_d + T  # (B, 3, D*H*W)
    del ref_grid_d, ref_grid, transform, R, T  # release (GPU) memory
    # NOTE 齐次性
    src_grid = src_grid_d[:, :2] / src_grid_d[:, -1:]  # divide by depth (B, 2, D*H*W)
    del src_grid_d
    src_grid[:, 0] = src_grid[:, 0] / ((W - 1) / 2) - 1  # scale to -1~1
    src_grid[:, 1] = src_grid[:, 1] / ((H - 1) / 2) - 1  # scale to -1~1
    src_grid = src_grid.permute(0, 2, 1)  # (B, D*H*W, 2)
    src_grid = src_grid.view(B, D, H * W, 2)

    warped_src_feat = F.grid_sample(src_feat, src_grid,
                                    mode='bilinear', padding_mode='zeros',
                                    align_corners=True)  # (B, C, D, H*W) 有d个批次（平面）的采样
    warped_src_feat = warped_src_feat.view(B, C, D, H, W)

    return warped_src_feat


if __name__ == '__main__':
    root = r"C:\Users\zpl\Desktop\template"
    out_dir = r"C:\Users\zpl\Desktop\template\out"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    dataset = ArialDataset(root)
    # a = dataset[0]

    plt.imshow(dataset[0][0][0].int().permute(1, 2, 0))

    # %%
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    tmp = iter(dataloader)
    item = next(tmp)
    imgs = item[0].cuda()
    proj_matrices = item[1].cuda()
    depth_values = item[2].cuda()
    imgs = torch.unbind(imgs, 1)
    proj_matrices = torch.unbind(proj_matrices, 1)
    ref_img, src_imgs = imgs[0], imgs[1:]
    ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

    warped_imgs = homo_warp(src_imgs[0], src_projs[0], ref_proj, depth_values)

    cv2.imwrite(out_dir + "/ref.png",
                ref_img.permute([0, 2, 3, 1])[0].detach().cpu().numpy()[:, :, ::-1].astype(np.int32))
    cv2.imwrite(out_dir + "/src.png",
                src_imgs[0].permute([0, 2, 3, 1])[0].detach().cpu().numpy()[:, :, ::-1].astype(np.int32))
    for i in range(warped_imgs.shape[2]):
        warped_img = warped_imgs[:, :, i, :, :].permute([0, 2, 3, 1]).contiguous()
        img_np = warped_img[0].detach().cpu().numpy()
        cv2.imwrite(out_dir + "/tmp" + str(i) + ".png", img_np[:, :, :].astype(np.int32))
