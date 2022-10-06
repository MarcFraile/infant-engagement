#!/bin/env -S python3 -u


# ================ IMPORTS ================ #


from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import imageio.v3 as iio

import torch
from torch import Tensor, nn
from torchvision import transforms

from local import heatmap, util
from local.cli import PrettyCli, CliHelper
from local.transforms import AssertDims, AdjustLength, Contiguous
from local.network_models import Net
from local.datasets import VideoDataset


# ================ SETTINGS ================ #


# ---- Constants ---- #

TASKS = [ "people", "eggs", "drums" ]
VARIABLES = [ "attending", "participating" ]

# ---- Net Dirs ---- #

NET_DIRS : Dict[str, Dict[str, Path]] = {
    "people": {
        "attending"     : Path("artifacts/final/20220405T212235"),
        # "participating" : Path("artifacts/final/20220405T211630"),
    },
    # "eggs"  : {
    #     "attending"     : Path("artifacts/final/20220405T222620"),
    #     "participating" : Path("artifacts/final/20220407T110032"),
    # },
    # "drums" : {
    #     "attending"     : Path("artifacts/final/20220406T155349"),
    #     "participating" : Path("artifacts/final/20220407T105856"),
    # },
}

FLATTENED_NET_DIRS = { f"NET_ROOT_{task.upper()}_{variable.upper()}": path for (task, inner) in NET_DIRS.items() for (variable, path) in inner.items() }

# ---- Filesystem ---- #

CWD             : Path = Path(os.getcwd()).resolve() # Book-keeping.
CURRENT_SCRIPT  : Path # Book-keeping.

VIDEO_ROOT      : Path # Load vids from this directory.
FEATURE_ROOT    : Path # Load pre-computed encoded samples from this directory.
ANNOTATION_FILE : Path # Load annotations from this file.
STATS_FILE      : Path # Load fold sample statistics from this file.
SNIPPETS_FILE   : Path # Load the choice of snippets to process from here.

INPUT_DIRS = {
    "CWD"          : CWD,
    "VIDEO_ROOT"   : Path("data/processed/video/"),
    "FEATURE_ROOT" : Path("data/processed/baked_samples/"),
    **FLATTENED_NET_DIRS,
}

INPUT_FILES = {
    "CURRENT_SCRIPT"  : Path(__file__).resolve().relative_to(CWD),
    "ANNOTATION_FILE" : Path("data/processed/engagement/stratified_annotation_spans.csv"),
    "STATS_FILE"      : Path("data/processed/fold_statistics.json"),
    "SNIPPETS_FILE"   : Path("data/processed/human_attention/candidate_snippets.csv"),
}

locals().update(INPUT_DIRS)
locals().update(INPUT_FILES)

OUTPUT_ROOT = Path("artifacts/machine_attention")

# ---- Helpers ---- #

cli = PrettyCli()
helper = CliHelper(cli)


# ================ FUNCTIONS ================ #


def main() -> None:
    cli.main_title("Calculate Machine Attention")

    env = helper.report_environment()
    helper.report_input_sources(INPUT_DIRS, INPUT_FILES)
    gpu = helper.report_gpu()
    start_time, out_dir = helper.setup_output_dir(OUTPUT_ROOT, NET_DIRS)
    snippets = helper.report_snippet_info(SNIPPETS_FILE)

    for task in TASKS:
        if not (task in NET_DIRS):
            continue
        for variable in VARIABLES:
            if not variable in NET_DIRS[task]:
                continue
            net_dir = NET_DIRS[task][variable]
            relabeled_heatmap(net_dir, out_dir, task, variable, gpu, snippets)


def relabeled_heatmap(net_dir: Path, out_dir: Path, task: str, variable: str, gpu: torch.device, snippets: pd.DataFrame) -> None:
    cli.chapter(f"Task: {task}; Variable: {variable}")

    net = helper.load_pickled_net(net_dir / "net.pt", gpu)
    assert type(net) == Net

    test_transform = transforms.Compose([
        AssertDims(height=160),
        AdjustLength(keep=(60, 60)),
        transforms.CenterCrop(160),
        Contiguous(),
    ])

    net_task, net_variable, snippet_duration, snippet_subdivision = report_training_params(net_dir)
    assert net_task     == task
    assert net_variable == variable

    manager = helper.report_video_manager(
        VIDEO_ROOT, ANNOTATION_FILE, STATS_FILE,
        task, variable, snippet_duration, snippet_subdivision,
        test_transform=test_transform,
    )
    dataset = manager.test_set()

    generate_heatmaps(dataset, net, gpu, out_dir, task, variable, snippets)


def report_training_params(root: Path) -> Tuple[str, str, float, int]:
    cli.section("Training Script Parameters")

    params = helper.json_read(root / "script_params.json")

    task                = params["GENERAL" ]["TASK"]
    variable            = params["GENERAL" ].get("VARIABLE", "participating")
    snippet_duration    = params["FINETUNE"]["SNIPPET_DURATION"]
    snippet_subdivision = params["FINETUNE"]["SNIPPET_SUBDIVISION"]

    cli.print({
        "Task" : task,
        "Variable" : variable,
        "Snippet Duration" : snippet_duration,
        "Snippet Subdivision" : snippet_subdivision,
    })

    return task, variable, snippet_duration, snippet_subdivision


@dataclass
class SnippetData:
    session           : str
    sample_start      : int
    label             : int
    missing_annotator : str


def generate_heatmaps(dataset: VideoDataset, net: Net, gpu: torch.device, out_dir: Path, task: str, variable: str, snippets: pd.DataFrame) -> None:
    cli.section("Generating Heatmaps")

    snippets = snippets.loc[(task, variable)]
    for idx, (_, entry) in enumerate(snippets.iterrows()): # Ignored variable is Pandas index, which in this case is just (task, variable) for all entries.

        cli.chapter(f"Entry {idx}")

        session           = entry["session"]
        sample_start      = entry["start_ms"]
        label             = entry["label"]
        missing_annotator = entry["annotator"]

        assert type(session)               == str, f"Bad type found in entry ({task}, {variable}, {idx}) for key 'session': expected str, found {type(session)}."
        assert type(missing_annotator)     == str, f"Bad type found in entry ({task}, {variable}, {idx}) for key 'missing_annotator': expected str, found {type(missing_annotator)}."
        assert type(sample_start) in [int, float], f"Bad type found in entry ({task}, {variable}, {idx}) for key 'sample_start': expected int or float, found {type(sample_start)}."
        assert type(label)        in [int, float], f"Bad type found in entry ({task}, {variable}, {idx}) for key 'label': expected int or float, found {type(label)}."

        if type(sample_start) == float:
            assert sample_start.is_integer()
            sample_start = int(sample_start)

        if type(label) == float:
            assert label.is_integer()
            label = int(label)

        snippet_data = SnippetData(session, sample_start, label, missing_annotator)
        cli.print(asdict(snippet_data))

        # NOTE: The snippets were taken so the label is the same for all present annotators.
        #       Therefore, the choice doesn't matter. Here, we just pick the first available one alphabetically.
        #       This is more brittle than random.choice([ ... ]), but should be fine.
        annotator = "ew" if (missing_annotator != "ew") else "mf"
        frames, _ = dataset.get_sample(session, annotator, sample_start)
        frames    = frames.to(gpu).unsqueeze(0)
        # label     = torch.tensor(label, device=gpu)

        make_heatmaps(frames, 0, snippet_data, net, out_dir, dataset.fps, gpu)
        make_heatmaps(frames, 1, snippet_data, net, out_dir, dataset.fps, gpu)


def make_heatmaps(frames: Tensor, target_label: int, snippet_data: SnippetData, net: Net, out_root: Path, fps: float, gpu: torch.device) -> None:
    matching_text : str = "matching" if (snippet_data.label == target_label) else "opposite"

    cli.subchapter(f"target: {target_label}, ground truth: {snippet_data.label} ({matching_text})")

    outer_dir_name : str = f"{snippet_data.session}_{snippet_data.sample_start}ms"
    inner_dir_name : str = f"{matching_text}__target_{target_label}__true_{snippet_data.label}"
    out_dir = out_root / outer_dir_name / inner_dir_name
    out_dir.mkdir(parents=True, exist_ok=False)

    save_vid(out_dir / "original.mp4", frames, fps)

    calculate_gradient(frames, target_label, net, out_dir, fps)
    gradcam  = calculate_gradcam(frames, target_label, net, out_dir, fps)
    guided_backprop = calculate_guided_backprop(frames, target_label, net, out_dir, fps)
    calculate_guided_gradcam(frames, gradcam, guided_backprop, out_dir, fps)
    calculate_occlusion(frames, target_label, net, out_dir, fps, gpu)


def calculate_gradient(frames: Tensor, target_label: int, net: nn.Module, target_dir: Path, fps: float) -> Tensor:
    cli.print("Gradient saliency...", end="")

    gradient = heatmap.vanilla_backprop(net, frames, class_idx=target_label)
    saliency = heatmap.saliency(gradient)
    overlay = heatmap.colormap_overlay(frames, saliency)
    overlay = enhanced_overlay(base=frames, overlay=saliency)

    save_vid(target_dir / "gradient.mp4", gradient, fps)
    save_vid(target_dir / "gradient_saliency.mp4", saliency, fps)
    save_vid(target_dir / "gradient_saliency_overlay.mp4", overlay, fps)

    torch.save(gradient, target_dir /  "gradient.pt")
    torch.save(saliency, target_dir /  "gradient_saliency.pt")

    cli.print("Done.")
    return saliency


def calculate_gradcam(frames: torch.Tensor, target_label: int, net: Net, target_dir: Path, fps: float) -> torch.Tensor:
    cli.print("Grad-CAM...", end="")

    target_layer: nn.Module = net.encoder.base_model.layer4[-1].conv2 # Last conv layer.
    gradcam = heatmap.gradcam(net=net, target_layer=target_layer, input=frames, class_idx=target_label)
    overlay = enhanced_overlay(base=frames, overlay=gradcam)

    save_vid(target_dir / "gradcam.mp4", gradcam, fps)
    save_vid(target_dir / "gradcam_overlay.mp4", overlay, fps)

    torch.save(gradcam, target_dir /  "gradcam.pt")

    cli.print("Done.")
    return gradcam


def calculate_guided_backprop(frames: Tensor, target_label: int, net: nn.Module, target_dir: Path, fps: float) -> Tensor:
    cli.print("Guided Backpropagation...", end="")

    guided_backprop = heatmap.guided_backprop(net=net, input=frames, class_idx=target_label)
    saliency = heatmap.saliency(guided_backprop)
    overlay = enhanced_overlay(base=frames, overlay=saliency)

    save_vid(target_dir / "guided_backprop.mp4", guided_backprop, fps)
    save_vid(target_dir / "guided_backprop_saliency.mp4", saliency, fps)
    save_vid(target_dir / "guided_backprop_saliency_overlay.mp4", overlay, fps)

    torch.save(guided_backprop, target_dir /  "guided_backprop.pt")
    torch.save(saliency, target_dir /  "guided_backprop_saliency.pt")

    cli.print("Done.")
    return guided_backprop


def calculate_guided_gradcam(frames: Tensor, gradcam: Tensor, guided_backprop: Tensor, target_dir: Path, fps: float) -> Tensor:
    cli.print("Guided Grad-CAM...", end="")

    guided_gradcam = heatmap.scale_to_match(frames, gradcam, mode="linear") * guided_backprop
    saliency = heatmap.saliency(guided_gradcam)
    overlay = enhanced_overlay(base=frames, overlay=saliency)

    save_vid(target_dir / "guided_gradcam.mp4", guided_gradcam, fps)
    save_vid(target_dir / "guided_gradcam_saliency.mp4", saliency, fps)
    save_vid(target_dir / "guided_gradcam_overlay.mp4", overlay, fps)

    torch.save(guided_gradcam, target_dir /  "guided_gradcam.pt")
    torch.save(saliency, target_dir /  "guided_gradcam_saliency.pt")

    cli.print("Done.")
    return guided_gradcam


def calculate_occlusion(frames: Tensor, target_label: int, net: nn.Module, target_dir: Path, fps: float, gpu: torch.device) -> Tensor:
    cli.print("Occlusion...", end="")

    occlusion = heatmap.occlusion_3d(frames, net, device=gpu, occluder_radius=(1, 16, 16), occluder_stride=(1, 16, 16), class_idx=target_label)
    overlay = enhanced_overlay(base=frames, overlay=occlusion)

    save_vid(target_dir / "occlusion.mp4", occlusion, fps)
    save_vid(target_dir / "occlusion_overlay.mp4", overlay, fps)

    torch.save(occlusion, target_dir / "occlusion.pt")

    cli.print("Done.")
    return occlusion


# TODO: Fix e2e.util.save_vid() and use that instead of this custom variation?
def save_vid(path: Path, vid: Tensor, fps: float) -> None:
    vid = heatmap.common.clean(vid)

    if len(vid.shape) == 3: # Grayscale
        vid = util.remap(vid)
        vid = heatmap.viz.colormap(vid)

    if vid.shape[3] > 3: # C is not in last pos -> PyTorch dim order (CTHW) needs to be mapped to NumPy order (THWC).
        vid = vid.permute(1, 2, 3, 0)

    vid = vid.numpy()
    vid = util.remap(vid, dest=(0, 256))
    vid = vid.clip(0, 255)
    vid = vid.astype(np.uint8)

    iio.imwrite(path, vid, fps=fps)


# TODO: This is copied from heatmap_task.py and should be abstracted. Maybe extended features on heatmap.colormap_overlay()?
def enhanced_overlay(base: Tensor, overlay: Tensor) -> Tensor:
    """
    Overlays `overlay` on `base`, with added intensity tweaking.
    """

    def pump_it(x: Tensor) -> Tensor:
        return util.remap(x.squeeze(), source=(x.quantile(0.01), x.quantile(0.999))).clamp(0, 1)

    combined = heatmap.colormap_overlay(pump_it(base), pump_it(overlay), mode="linear")

    return combined


# ================ KICKSTART ================ #


if __name__ == "__main__":
    main()
