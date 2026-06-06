#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.graphics_utils import getWorld2View2
import copy
import numpy as np


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_video")
    # gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    # makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx < len(views) - 1:
            view_next = views[idx + 1]
        else:
            view_next = views[idx]

        ###
        view05 = copy.deepcopy(view)
        view05.R = 0.0 * view_next.R + 1.0 * view.R
        view05.T = 0.0 * view_next.T + 1.0 * view.T
        trans = np.array([0.0, 0.0, 0.0])
        scale = 1.0
        view05.world_view_transform = (
            torch.tensor(getWorld2View2(view05.R, view05.T, trans, scale)).transpose(0, 1).cuda()
        )
        view05.full_proj_transform = (
            view05.world_view_transform.unsqueeze(0).bmm(view05.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        view05.camera_center = view05.world_view_transform.inverse()[3, :3]

        # METU-VISTIR
        # rendering = render(view05, gaussians, pipeline, background, use_trained_exp=train_test_exp)["render"][
        #     :, :, 350:-350
        # ]
        # rendering = torch.nn.functional.interpolate(rendering[None, ...], (518, 518), mode="area")[0]

        # temperature = render(view05, gaussians, pipeline, background, use_trained_exp=train_test_exp)["temperature"][
        #     :, :, 350:-350
        # ]
        # temperature = torch.nn.functional.interpolate(temperature[None, ...], (518, 518), mode="area")[0]

        rendering = render(view05, gaussians, pipeline, background, use_trained_exp=train_test_exp)["render"]
        temperature = render(view05, gaussians, pipeline, background, use_trained_exp=train_test_exp)["temperature"]

        torchvision.utils.save_image(rendering, os.path.join(render_path, "{0:05d}".format(idx * 10) + "_rgb.png"))
        torchvision.utils.save_image(temperature, os.path.join(render_path, "{0:05d}".format(idx * 10) + "_temp.png"))

        ###
        view05 = copy.deepcopy(view)
        view05.R = 0.25 * view_next.R + 0.75 * view.R
        view05.T = 0.25 * view_next.T + 0.75 * view.T
        trans = np.array([0.0, 0.0, 0.0])
        scale = 1.0
        view05.world_view_transform = (
            torch.tensor(getWorld2View2(view05.R, view05.T, trans, scale)).transpose(0, 1).cuda()
        )
        view05.full_proj_transform = (
            view05.world_view_transform.unsqueeze(0).bmm(view05.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        view05.camera_center = view05.world_view_transform.inverse()[3, :3]

        # METU-VISTIR
        # rendering = render(view05, gaussians, pipeline, background, use_trained_exp=train_test_exp)["render"][
        #     :, :, 350:-350
        # ]
        # rendering = torch.nn.functional.interpolate(rendering[None, ...], (518, 518), mode="area")[0]

        # temperature = render(view05, gaussians, pipeline, background, use_trained_exp=train_test_exp)["temperature"][
        #     :, :, 350:-350
        # ]
        # temperature = torch.nn.functional.interpolate(temperature[None, ...], (518, 518), mode="area")[0]

        rendering = render(view05, gaussians, pipeline, background, use_trained_exp=train_test_exp)["render"]
        temperature = render(view05, gaussians, pipeline, background, use_trained_exp=train_test_exp)["temperature"]

        torchvision.utils.save_image(rendering, os.path.join(render_path, "{0:05d}".format(idx * 10 + 1) + "_rgb.png"))
        torchvision.utils.save_image(
            temperature, os.path.join(render_path, "{0:05d}".format(idx * 10 + 1) + "_temp.png")
        )

        ###
        view05 = copy.deepcopy(view)
        view05.R = 0.5 * view_next.R + 0.5 * view.R
        view05.T = 0.5 * view_next.T + 0.5 * view.T
        trans = np.array([0.0, 0.0, 0.0])
        scale = 1.0
        view05.world_view_transform = (
            torch.tensor(getWorld2View2(view05.R, view05.T, trans, scale)).transpose(0, 1).cuda()
        )
        view05.full_proj_transform = (
            view05.world_view_transform.unsqueeze(0).bmm(view05.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        view05.camera_center = view05.world_view_transform.inverse()[3, :3]

        # # METU-VISTIR
        # rendering = render(view05, gaussians, pipeline, background, use_trained_exp=train_test_exp)["render"][
        #     :, :, 350:-350
        # ]
        # rendering = torch.nn.functional.interpolate(rendering[None, ...], (518, 518), mode="area")[0]

        # temperature = render(view05, gaussians, pipeline, background, use_trained_exp=train_test_exp)["temperature"][
        #     :, :, 350:-350
        # ]
        # temperature = torch.nn.functional.interpolate(temperature[None, ...], (518, 518), mode="area")[0]

        rendering = render(view05, gaussians, pipeline, background, use_trained_exp=train_test_exp)["render"]
        temperature = render(view05, gaussians, pipeline, background, use_trained_exp=train_test_exp)["temperature"]

        torchvision.utils.save_image(rendering, os.path.join(render_path, "{0:05d}".format(idx * 10 + 2) + "_rgb.png"))
        torchvision.utils.save_image(
            temperature, os.path.join(render_path, "{0:05d}".format(idx * 10 + 2) + "_temp.png")
        )

        ###
        view05 = copy.deepcopy(view)
        view05.R = 0.75 * view_next.R + 0.25 * view.R
        view05.T = 0.75 * view_next.T + 0.25 * view.T
        trans = np.array([0.0, 0.0, 0.0])
        scale = 1.0
        view05.world_view_transform = (
            torch.tensor(getWorld2View2(view05.R, view05.T, trans, scale)).transpose(0, 1).cuda()
        )
        view05.full_proj_transform = (
            view05.world_view_transform.unsqueeze(0).bmm(view05.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        view05.camera_center = view05.world_view_transform.inverse()[3, :3]

        # METU-VISTIR
        # rendering = render(view05, gaussians, pipeline, background, use_trained_exp=train_test_exp)["render"][
        #     :, :, 350:-350
        # ]
        # rendering = torch.nn.functional.interpolate(rendering[None, ...], (518, 518), mode="area")[0]

        # temperature = render(view05, gaussians, pipeline, background, use_trained_exp=train_test_exp)["temperature"][
        #     :, :, 350:-350
        # ]
        # temperature = torch.nn.functional.interpolate(temperature[None, ...], (518, 518), mode="area")[0]

        rendering = render(view05, gaussians, pipeline, background, use_trained_exp=train_test_exp)["render"]
        temperature = render(view05, gaussians, pipeline, background, use_trained_exp=train_test_exp)["temperature"]

        torchvision.utils.save_image(rendering, os.path.join(render_path, "{0:05d}".format(idx * 10 + 3) + "_rgb.png"))
        torchvision.utils.save_image(
            temperature, os.path.join(render_path, "{0:05d}".format(idx * 10 + 3) + "_temp.png")
        )


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
                dataset.train_test_exp,
            )

        if not skip_test:
            render_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
                dataset.train_test_exp,
            )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
