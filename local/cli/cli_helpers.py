import socket
import sys, os
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import json
from typing import Dict, Optional, Tuple, Any, List

import numpy as np
import torch
import pandas as pd

from local.cli import PrettyCli
from local.transforms import get_default_transforms
from local.util import count_params
from local.training import TensorMap
from local.datasets import TensorManager, VideoManager


class Environment:
    """Environment variables set by managing software (SLURM, Singularity), grouped by origin."""

    @dataclass
    class Slurm:
        """Environment variables initialized by SLURM when running an array job."""
        job_id    : int
        task_id   : int
        num_tasks : int

    @dataclass
    class Singularity:
        """Environment variables initialized by Singularity when running an image."""
        image_name : str
        command    : str

    command     : Optional[List[str]]
    hostname    : Optional[str]
    slurm       : Optional[Slurm]
    singularity : Optional[Singularity]

    def __init__(self, command: Optional[List[str]], hostname: Optional[str] = None, slurm: Optional[Slurm] = None, singularity: Optional[Singularity] = None):
        self.command     = command
        self.hostname    = hostname
        self.slurm       = slurm
        self.singularity = singularity

    def to_dict(self) -> dict:
        """
        For display purposes with PrettyCli. Observe the nonstandard use of False for missing optionals.
        """
        return {
            "Command"     : self.command             if self.command     else False,
            "Hostname"    : self.hostname            if self.hostname    else False,
            "SLURM"       : asdict(self.slurm      ) if self.slurm       else False,
            "Singularity" : asdict(self.singularity) if self.singularity else False,
        }


class CliHelper:
    """
    Helper class for training scripts.

    * Contains static variables TASKS, PEOPLE, DRUMS, EGGS.
    * Contains validate-and-print methods report_gpu(), report_input_sources().
    """

    PEOPLE = "people"
    DRUMS  = "drums"
    EGGS   = "eggs"
    TASKS  = [PEOPLE, DRUMS, EGGS]

    cli: PrettyCli
    env: Optional[Environment]

    def __init__(self, cli: PrettyCli = PrettyCli()):
        self.cli = cli

    def json_read(self, path: Path) -> Any:
        """
        Reads a JSON object from a file.

        * Mostly here for symmetry with `json_write()`
        """
        assert path.is_file(), f"File not found: {path}"

        with open(path, "r") as file:
            data = json.load(file)

        return data

    def json_write(self, obj: Any, path: Path) -> None:
        """
        Pretty-writes JSON to a file.

        * Ensures NumPy data is handled correctly.
        """
        with open(path, "w") as file:
            json.dump(obj, file, indent=4, default=json_default)

    def report_gpu(self) -> torch.device:
        """
        Grab and report GPU.

        * Returns a CUDA device, (panics if not available).
        * Prints the device details.
        """
        assert torch.cuda.is_available()
        gpu = torch.device("cuda")
        props = torch.cuda.get_device_properties(gpu)

        self.cli.section("GPU Details")
        self.cli.print({
            "Name": props.name,
            "Version": f"{props.major}.{props.minor}",
            "Memory": f"{props.total_memory / (1024 * 1024 * 1024):4.02f} GB",
            "Cores": props.multi_processor_count,
        })

        return gpu

    def report_environment(self) -> Environment:
        """
        Returns a report on the current environment variables checked by managing software.

        * Checks for SLURM and Singularity.
        * If an output field is present, then all its fields should be filled with valid values.
        """

        self.cli.section("Environment")

        command     : List[str]                         = sys.argv
        slurm       : Optional[Environment.Slurm      ] = None
        singularity : Optional[Environment.Singularity] = None
        hostname    : str                               = socket.gethostname()

        job_id    = os.getenv("SLURM_ARRAY_JOB_ID")
        task_id   = os.getenv("SLURM_ARRAY_TASK_ID")
        num_tasks = os.getenv("SLURM_ARRAY_TASK_COUNT")

        if (job_id is not None) and (task_id is not None) and (num_tasks is not None):
            slurm = Environment.Slurm(int(job_id), int(task_id), int(num_tasks))

        image_name          = os.getenv("SINGULARITY_NAME")
        singularity_command = os.getenv("SINGULARITY_COMMAND")

        if (image_name is not None) and (singularity_command is not None):
            singularity = Environment.Singularity(image_name, singularity_command)

        output = Environment(command, hostname, slurm, singularity)
        self.cli.print(output.to_dict())
        self.env = output
        return output

    def report_input_sources(self, dirs: Dict[str, Path], files: Dict[str, Path]) -> None:
        """
        Validates all input sources are valid, and prints their locations.

        * Asserts all dirs are existing directories in the filesystem.
        * Asserts all files are existing files in the filesystem.
        """

        for (name, dir) in dirs.items():
            assert dir.is_dir(), f"[NOT A DIR] Key: {name}; Value: {dir}"

        for (name, file) in files.items():
            assert file.is_file(), f"[NOT A FILE] Key: {name}; Value: {file}"

        self.cli.section("Input Sources")
        self.cli.print({
            "Directories": dirs,
            "Files": files,
        })

    def setup_output_dir(self, output_root: Path, script_params: dict, add_timestamp: bool = True) -> Tuple[datetime, Path]:
        """
        Creates timestamped output folder and basic output files.

        * Returns (start_time, output_path).
        * Saves script params, creates a notes file.
        * Sets <output root>/current.txt as a pointer to the current output folder.
        * `add_timestamp` indicates if a sub-folder should be created with the current timestamp in ISO-8601 format (yyyymmddThhmmss). Defaults to true.
        * If the script is running in a SLURM environment, creates a `task_$ID` subfolder. This should be mutually exclusive with adding an automatic timestamp.
        """

        start_time = datetime.now()

        self.cli.section("Start Time")
        self.cli.print(start_time)

        output_path = output_root
        if add_timestamp:
            output_path = output_root / start_time.strftime("%Y%m%dT%H%M%S")

        if self.env and self.env.slurm:
            output_path = output_path / "tasks" / f"{self.env.slurm.task_id:03d}"

        output_path.mkdir(parents=True, exist_ok=False)

        self.cli.section("Output Path")
        self.cli.print(str(output_path))

        # This is handy to have a wrapping script copy the logs to the output dir (handled through `tee`).
        with open(output_root / "current.txt", "w") as file:
            file.write(str(output_path))

        (output_path / "notes.md").touch()

        self.cli.section("Script Parameters")
        self.cli.print(script_params)
        self.json_write(script_params, output_path / "script_params.json")

        return (start_time, output_path)

    def load_pickled_net(self, path: Path, gpu: Optional[torch.device] = None, section: Optional[str] = None) -> torch.nn.Module:
        """
        Loads a net from `path`.

        * If `gpu` provided, loads the net to the device.
        * Prints net summary.
        * If `section` provided, prints it as a section head.
        """
        section = section or "Loading Network"
        self.cli.section(section)

        assert path.is_file(), f"File not found: {path}"

        net : torch.nn.Module = torch.load(path)
        if gpu != None:
            net = net.to(gpu)
        net.eval()

        params = {
            "Total Parameters": count_params(net),
            "Parameters per Layer": { type(layer).__name__ : count_params(layer) for layer in net.children() },
        }
        self.cli.print(params)

        return net

    def report_tensor_manager(self, feature_root: Path, task: str, variable: str, device: torch.device) -> TensorManager:
        self.cli.section("K-Folds Tensor Manager")

        manager = TensorManager(feature_root, task, variable, device)

        self.cli.print({
            "Num Folds" : manager.num_folds(),
            "Sample Counts" : {
                "Train" : [ len(data.Y) for data in manager._train_folds ],
                "Test"  : len(manager._test_fold.Y),
            },
        })
        return manager

    def report_video_manager(
        self,
        video_root: Path, annotation_file: Path, stats_file: Path,
        task: str, variable: str, snippet_duration: float, snippet_subdivision: int,
        batch_size: Optional[int] = None, num_workers: int = 0, verbose: bool = True,
        train_transform: Optional[TensorMap] = None, test_transform: Optional[TensorMap] = None,
    ) -> VideoManager:

        self.cli.section("K-Folds Video Manager")

        default_train_transform, default_test_transform = get_default_transforms()
        train_transform = train_transform or default_train_transform
        test_transform  = test_transform  or default_test_transform

        manager = VideoManager(
            video_root, annotation_file, stats_file,
            task, variable, snippet_duration, snippet_subdivision,
            train_transform, test_transform,
            batch_size, num_workers, verbose
        )

        self.cli.print({
            "Num Folds"         : manager.num_folds(),
            "Training Sessions" : { k: [ sample.session for sample in manager._train_folds[k].samples ] for k in range(manager.num_folds()) }, # TODO: Avoid raw access to private members.
            "Testing Sessions"  : [ sample.session for sample in manager._test_fold.samples ], # TODO: Avoid raw access to private members.
            "Annotators"        : manager._annotators, # TODO: Avoid raw access to private members.
        })
        return manager

    def report_snippet_info(self, snippets_file: Path) -> pd.DataFrame:
        self.cli.section("Snippet Information")

        snippets = pd.read_csv(snippets_file)
        snippets = snippets.set_index(["task", "variable"]).sort_index()

        self.cli.print(snippets)
        return snippets


def json_default(obj):
    """
    Set this as the default() callback when calling json.dump() to ensure NumPy arrays are handled properly.
    """
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    # raise TypeError('Unknown type:', type(obj))
    return str(obj)
