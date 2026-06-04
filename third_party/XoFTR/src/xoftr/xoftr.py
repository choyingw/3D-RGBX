import torch
import torch.nn as nn
from einops.einops import rearrange
from .backbone import ResNet_8_2
from .utils.position_encoding import PositionEncodingSine
from .xoftr_module import LocalFeatureTransformer, FineProcess, CoarseMatching, FineSubMatching
import cv2
import numpy as np


class XoFTR(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config

        # Modules
        self.backbone = ResNet_8_2(config["resnet"])
        self.pos_encoding = PositionEncodingSine(config["coarse"]["d_model"])
        self.loftr_coarse = LocalFeatureTransformer(config["coarse"])
        self.coarse_matching = CoarseMatching(config["match_coarse"])
        self.fine_process = FineProcess(config)
        self.fine_matching = FineSubMatching(config)

    def forward(self, data):
        """
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        # 1. Local Feature CNN
        data.update(
            {"bs": data["image0"].size(0), "hw0_i": data["image0"].shape[2:], "hw1_i": data["image1"].shape[2:]}
        )

        eps = 1e-6

        image0_mean = data["image0"].mean(dim=[2, 3], keepdim=True)
        image0_std = data["image0"].std(dim=[2, 3], keepdim=True)
        image0 = (data["image0"] - image0_mean) / (image0_std + eps)

        image1_mean = data["image1"].mean(dim=[2, 3], keepdim=True)
        image1_std = data["image1"].std(dim=[2, 3], keepdim=True)
        image1 = (data["image1"] - image1_mean) / (image1_std + eps)

        if data["hw0_i"] == data["hw1_i"]:  # faster & better BN convergence
            feats_c, feats_m, feats_f = self.backbone(torch.cat([image0, image1], dim=0))
            (feat_c0, feat_c1) = feats_c.split(data["bs"])
            (feat_m0, feat_m1) = feats_m.split(data["bs"])
            (feat_f0, feat_f1) = feats_f.split(data["bs"])
        else:  # handle different input shapes
            feat_c0, feat_m0, feat_f0 = self.backbone(image0)
            feat_c1, feat_m1, feat_f1 = self.backbone(image1)

        data.update(
            {
                "hw0_c": feat_c0.shape[2:],
                "hw1_c": feat_c1.shape[2:],
                "hw0_m": feat_m0.shape[2:],
                "hw1_m": feat_m1.shape[2:],
                "hw0_f": feat_f0.shape[2:],
                "hw1_f": feat_f1.shape[2:],
            }
        )

        # save coarse features for fine matching
        feat_c0_pre, feat_c1_pre = feat_c0.clone(), feat_c1.clone()

        # 2. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        feat_c0 = rearrange(self.pos_encoding(feat_c0), "n c h w -> n (h w) c")
        feat_c1 = rearrange(self.pos_encoding(feat_c1), "n c h w -> n (h w) c")

        mask_c0 = mask_c1 = None  # mask is useful in training
        if "mask0" in data:
            mask_c0, mask_c1 = data["mask0"].flatten(-2), data["mask1"].flatten(-2)
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

        # 4. fine-level matching module
        feat_f0_unfold, feat_f1_unfold = self.fine_process(
            feat_f0, feat_f1, feat_m0, feat_m1, feat_c0, feat_c1, feat_c0_pre, feat_c1_pre, data
        )

        # 5. match fine-level and sub-pixel refinement
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

    def filtering(self, data):
        """
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        # 1. Local Feature CNN
        data.update(
            {"bs": data["image0"].size(0), "hw0_i": data["image0"].shape[2:], "hw1_i": data["image1"].shape[2:]}
        )
        # print(
        #     "AdwwdwAA", data["image0"].shape[2:], data["image0"].max(), data["image0"].min(), data["image1"].shape[2:]
        # )
        # st0 = data["image0"].detach().cpu().numpy()[0, 0]
        # cv2.imwrite("/home/choyingw/Documents/XoFTR/RGBT-Scenes/Building/tmp0.png", (st0 * 255.0).astype(np.uint8))
        # st1 = data["image1"].detach().cpu().numpy()[0, 0]
        # cv2.imwrite("/home/choyingw/Documents/XoFTR/RGBT-Scenes/Building/tmp1.png", (st1 * 255.0).astype(np.uint8))
        # exit()

        eps = 1e-6

        image0_mean = data["image0"].mean(dim=[2, 3], keepdim=True)
        image0_std = data["image0"].std(dim=[2, 3], keepdim=True)
        image0 = (data["image0"] - image0_mean) / (image0_std + eps)

        image1_mean = data["image1"].mean(dim=[2, 3], keepdim=True)
        image1_std = data["image1"].std(dim=[2, 3], keepdim=True)
        image1 = (data["image1"] - image1_mean) / (image1_std + eps)

        if data["hw0_i"] == data["hw1_i"]:  # faster & better BN convergence
            feats_c, feats_m, feats_f = self.backbone(torch.cat([image0, image1], dim=0))
            (feat_c0, feat_c1) = feats_c.split(data["bs"])
            (feat_m0, feat_m1) = feats_m.split(data["bs"])
            (feat_f0, feat_f1) = feats_f.split(data["bs"])
        else:  # handle different input shapes
            feat_c0, feat_m0, feat_f0 = self.backbone(image0)
            feat_c1, feat_m1, feat_f1 = self.backbone(image1)

        data.update(
            {
                "hw0_c": feat_c0.shape[2:],
                "hw1_c": feat_c1.shape[2:],
                "hw0_m": feat_m0.shape[2:],
                "hw1_m": feat_m1.shape[2:],
                "hw0_f": feat_f0.shape[2:],
                "hw1_f": feat_f1.shape[2:],
            }
        )

        # save coarse features for fine matching
        feat_c0_pre, feat_c1_pre = feat_c0.clone(), feat_c1.clone()

        # 2. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        feat_c0 = rearrange(self.pos_encoding(feat_c0), "n c h w -> n (h w) c")
        feat_c1 = rearrange(self.pos_encoding(feat_c1), "n c h w -> n (h w) c")

        mask_c0 = mask_c1 = None  # mask is useful in training
        if "mask0" in data:
            mask_c0, mask_c1 = data["mask0"].flatten(-2), data["mask1"].flatten(-2)
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # 3. match coarse-level
        sim_matrix = self.coarse_matching.forward_sim(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)
        print("Sim matrix shape:", sim_matrix.shape)

        diag = sim_matrix[0, :, :].diagonal().cpu()  # .float()

        # write_turbo_depth_metric(
        #     "sim_ori.png", sim_matrix[0, :, :].cpu().numpy(), vmin=sim_matrix.min(), vmax=sim_matrix.max()
        # )
        fifty_per = torch.quantile(diag, 0.5)
        ninenine_per = torch.quantile(diag, 0.99)
        r = fifty_per / ninenine_per
        thr = min(max(0.4 * (1 + 4 * (max(0.8 - r, 0.0))), 0.0), 1.0)

        cutoff = torch.quantile(diag, thr)

        # Build mask of positions to zero
        mask = diag < cutoff
        idx = torch.arange(len(diag))[mask]

        H, W = data["hw0_i"]
        ps = 8
        num_rows = H // ps
        num_cols = W // ps

        # idx are the selected *patch* indices in [0, num_rows*num_cols)
        idx = torch.arange(len(diag), device=diag.device)[mask]

        # map flat → (row, col) in patch grid
        row_patch = idx // num_cols  # 0..num_rows-1  (y)
        col_patch = idx % num_cols  # 0..num_cols-1  (x)

        # canvas (H, W)
        canvas = torch.ones((H, W), device=idx.device, dtype=torch.float32)

        # pixel coordinates for each selected patch
        rows = (ps * row_patch)[:, None] + torch.arange(ps, device=idx.device)[None, :]  # (N, ps)
        cols = (ps * col_patch)[:, None] + torch.arange(ps, device=idx.device)[None, :]  # (N, ps)

        # make an (N, ps, ps) grid, then flatten to 1D index lists
        rows_pix = rows[:, :, None].expand(-1, ps, ps).reshape(-1)
        cols_pix = cols[:, None, :].expand(-1, ps, ps).reshape(-1)

        # write zeros
        canvas.index_put_((rows_pix, cols_pix), torch.zeros_like(rows_pix, dtype=canvas.dtype))

        # # Compute patch coordinates
        # row_patch = idx % (data["hw0_i"][0] // 8)  # width in patches
        # col_patch = idx // (data["hw0_i"][1] // 8)  # height in patches

        # # Start with all ones
        # canvas = torch.ones(*data["hw0_i"])

        # # Build index ranges for each patch
        # rows = torch.arange(8)[None, :] + (8 * col_patch)[:, None]
        # cols = torch.arange(8)[None, :] + (8 * row_patch)[:, None]

        # # Use broadcasting to assign zeros
        # canvas.index_put_(
        #     (rows[:, :, None].expand(-1, -1, 8).reshape(-1), cols[:, None, :].expand(-1, 8, -1).reshape(-1)),
        #     torch.zeros(rows.numel() * 8),
        # )

        # print("DDW", canvas.shape)
        canvas_draw = canvas.detach().cpu().numpy()  # [0, 0]
        cv2.imwrite(
            "/home/choyingw/Documents/XoFTR/RGBT-Scenes/Building/canvas.png", (canvas_draw * 255.0).astype(np.uint8)
        )

        data.update({"conf_mask": canvas, "sim_matrix": sim_matrix})

        # np.save('', masked_img1)
        # cv2.imwrite("sim_mask2.png", (canvas.numpy() * 255.0).astype(np.uint8))

        # cutoff = torch.quantile(diag, 0.4)
        # canvas = torch.ones(640, 640)
        # for i in range(len(diag)):
        #     if diag[i] < cutoff:
        #         row_patch = i % 80
        #         col_patch = i // 80
        #         row_patch = row_patch
        #         col_patch = col_patch
        #         canvas[8 * col_patch : 8 * (col_patch + 1), 8 * row_patch : 8 * (row_patch + 1)] = 0
        # cv2.imwrite("sim_mask2.png", (canvas.numpy() * 255.0).astype(np.uint8))

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith("matcher."):
                state_dict[k.replace("matcher.", "", 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
