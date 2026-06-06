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


def is_metu_dataset(source_path):
    return "METU_VisTIR" in source_path or "/METU" in source_path


def resolve_render_transform(source_path, crop_margin, render_size):
    if crop_margin >= 0 and render_size >= 0:
        return crop_margin, render_size

    if is_metu_dataset(source_path):
        return (
            350 if crop_margin < 0 else crop_margin,
            518 if render_size < 0 else render_size,
        )

    return (
        0 if crop_margin < 0 else crop_margin,
        0 if render_size < 0 else render_size,
    )


def maybe_crop_and_resize(image, crop_margin=0, render_size=0):
    if crop_margin > 0 and image.shape[-1] > 2 * crop_margin:
        image = image[:, :, crop_margin:-crop_margin]

    if render_size > 0:
        image = torch.nn.functional.interpolate(image[None, ...], (render_size, render_size), mode="area")[0]

    return image


def render_set(
    model_path,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    train_test_exp,
    render_output="",
    crop_margin=0,
    render_size=0,
):
    if render_output:
        render_path = os.path.join(render_output, name)
        gts_path = os.path.join(render_output, name, "gt")
    else:
        render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
        gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        render_pkg = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp)
        rendering = maybe_crop_and_resize(render_pkg["render"], crop_margin, render_size)
        temperature = maybe_crop_and_resize(render_pkg["temperature"], crop_margin, render_size)

        # NIR + others
        # rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp)["render"]
        # temperature = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp)["temperature"]

        torchvision.utils.save_image(rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png"))
        torchvision.utils.save_image(temperature, os.path.join(render_path, "{0:05d}".format(idx) + "_temp.png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png"))


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    render_output: str = "",
    crop_margin: int = -1,
    render_size: int = -1,
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        crop_margin, render_size = resolve_render_transform(dataset.source_path, crop_margin, render_size)

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
                render_output,
                crop_margin,
                render_size,
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
                render_output,
                crop_margin,
                render_size,
            )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--render_output", default="", type=str)
    parser.add_argument("--crop_margin", default=-1, type=int)
    parser.add_argument("--render_size", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args.render_output,
        args.crop_margin,
        args.render_size,
    )
