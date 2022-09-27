import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List

import pandas as pd
import imageio.v3 as iio
from tqdm import trange

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from local.types import TensorMap
from local.training import KFoldsManager


@dataclass
class VideoSample:
    session    : str
    annotators : List[str]
    label_data : pd.DataFrame
    frames     : torch.Tensor


@dataclass
class VideoFold:
    """
    Info needed by `VideoManager` to merge folds into a `VideoDataset`.
    """
    samples               : List[VideoSample]
    empirical_probability : float
    pixel_mean            : Tensor
    pixel_std             : Tensor


class VideoDataset(Dataset):
    samples                : List[VideoSample]
    # transform              : TensorMap
    fps                    : float
    subdivision            : int

    sample_duration_ms     : float
    sample_duration_frames : int

    def __init__(
        self,
        samples         : List[VideoSample],
        transform       : TensorMap,
        fps             : float, # frames / second
        sample_duration : float, # Seconds
        subdivision     : int,   # Number of samples to yield per video. Sample i will be drawn from video i // subdivision, with start time in the (i % subdivision)-th chunk.
    ):
        assert len(samples) > 0
        assert fps > 0.0
        assert sample_duration > 1.0 # NOTE: Arbitrary!
        assert subdivision > 0

        self.samples     = samples
        self.transform   = transform
        self.fps         = fps
        self.subdivision = subdivision

        self.sample_duration_ms = 1000.0 * sample_duration
        self.sample_duration_frames = int(sample_duration * fps)

        assert self.sample_duration_ms > 1000.0 # NOTE: Arbitrary!
        assert self.sample_duration_frames > 5  # NOTE: Arbitrary!

    def __len__(self) -> int:
        return self.subdivision * len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        * Takes `self.subdivision` samples from each session. The provided index corresponds to the sample and not the session.
        * Sample start and end times are chosen randomly within the corresponding session, with earlier samples being picked from earlier starting times.
        * There can be overlap between consecutive samples.
        """

        # idx => session
        sample_idx = (idx // self.subdivision)
        start_time_idx = (idx % self.subdivision)

        assert 0 <= idx < len(self)
        assert 0 <= sample_idx < len(self.samples)

        sample = self.samples[sample_idx]

        # Choose an annotator
        annotator = sample.annotators[torch.randint(len(sample.annotators), ())]
        label_data = sample.label_data.loc[annotator]

        # Get the data
        sample_start_ms, sample_end_ms = self._get_times(label_data, start_time_idx)
        frames = self._get_frames(sample, sample_start_ms)
        label  = self._get_label(label_data, sample_start_ms, sample_end_ms)

        return (frames, label)

    def _get_times(self, label_data: pd.DataFrame, idx: int) -> Tuple[float, float]:
        """
        * Determines task length from the `label_data`.
        * Chooses a start time in the `(idx / subdivision)`-th piece of the task length (adjusted so the end time is within bounds).
        * Chooses an end time determined by `self.sample_duration_ms`.
        * Works best if `(task length) / self.subdivision` is greater than `self.sample_duration_ms`.
        """
        # NOTE: All times in milliseconds.

        task_start : int = label_data["start_ms"].min()
        task_end   : int = label_data["end_ms"  ].max()

        task_duration = task_end - task_start
        adjusted_task_duration = task_duration - self.sample_duration_ms

        subdivision_duration = adjusted_task_duration / self.subdivision
        subdivision_start = task_start + idx * subdivision_duration

        sample_start = subdivision_start + torch.rand(()).item() * subdivision_duration
        sample_end   = sample_start + self.sample_duration_ms

        return sample_start, sample_end

    def _get_frames(self, sample: VideoSample, sample_start_ms: float) -> torch.Tensor:
        """
        Extracts the corresponding frames, given a sample and a starting time.
        * Snaps starting frame index to the closest frame boundary (except for rounding?)
        * Ensures the snippet has the correct number of frames, so snippets can be stacked on top of each other.
        """

        sample_start_idx = int((sample_start_ms * self.fps) / 1000)
        sample_end_idx   = sample_start_idx + self.sample_duration_frames

        frames = sample.frames[:, sample_start_idx:sample_end_idx, :, :] # CTHW
        frames = self.transform(frames)
        assert frames.shape[1] == self.sample_duration_frames # Ensure we get the correct number of frames so we can stack.

        return frames

    def _get_label(self, label_data: pd.DataFrame, sample_start_ms: float, sample_end_ms: float) -> torch.Tensor:
        """
        Calculates the binary label for a snippet, given the interval annotations from one annotator.
        """

        clipped_start : pd.DataFrame = label_data["start_ms"].clip(sample_start_ms, sample_end_ms)
        clipped_end   : pd.DataFrame = label_data["end_ms"  ].clip(sample_start_ms, sample_end_ms)
        engaged_duration = (label_data["engaged"] * (clipped_end - clipped_start)).sum()

        # Engaged if positive labels are more than half the length.
        label = (engaged_duration > self.sample_duration_ms * 0.5)
        label = torch.tensor(label).long()

        return label

    def _seek_sample(self, session: str) -> VideoSample:
        """
        Finds and returns the sample corresponding to the provided `session`.
        * Panics if the `session` is missing.
        """
        sample : Optional[VideoSample] = None
        for s in self.samples:
            if s.session == session:
                sample = s
                break

        assert sample is not None
        return sample

    def get_sample(self, session: str, annotator: str, sample_start_ms: float) -> Tuple[Tensor, Tensor]:
        """
        Get one deterministic sample, given its identifying information (session, annotator, sample start).
        * Returns `(sample, label)`.
        """

        sample = self._seek_sample(session)
        assert annotator in sample.annotators

        label_data = sample.label_data.loc[annotator]

        task_start : int = label_data["start_ms"].min()
        task_end   : int = label_data["end_ms"  ].max()

        sample_end_ms = sample_start_ms + self.sample_duration_ms

        assert task_start <= sample_start_ms, f"Starting time out of bounds: Task starts at {task_start}ms, requested {sample_start_ms}ms."
        assert task_end   >= sample_end_ms  , f"End time out of bounds: Task ends at {task_end}ms, sample ends at {sample_end_ms}ms (requested starting time: {sample_start_ms}ms)."

        frames = self._get_frames(sample, sample_start_ms)
        label  = self._get_label(label_data, sample_start_ms, sample_end_ms)

        return (frames, label)

    def get_sample_times(self, session: str, annotator: str) -> Tuple[float, float]:
        """
        Get the start and end times (in ms) for the relevant task in the given `(session, annotator)` pair.
        """

        sample = self._seek_sample(session)
        assert annotator in sample.annotators

        label_data = sample.label_data.loc[annotator]

        task_start : int = label_data["start_ms"].min()
        task_end   : int = label_data["end_ms"  ].max()

        return (task_start, task_end)

    def get_common_times(self, session: str) -> Tuple[float, float]:
        """
        Get the common timespan that all available annotators have covered.
        * Returns `(start_ms, end_ms)` timestamps as floating-point milliseconds.
        """

        sample = self._seek_sample(session)

        task_start = float("-inf")
        task_end   = float("+inf")

        for annotator in sample.annotators:
            annotator_start, annotator_end = self.get_sample_times(session, annotator)
            task_start = max(task_start, annotator_start)
            task_end   = min(task_end, annotator_end)

        assert 0 < task_start
        assert task_start < task_end
        assert task_end < 1000 * 60 * 60 # Less than one hour (known upper bound).

        return task_start, task_end

    def get_random_common_starting_time(self, session: str) -> float:
        task_start, task_end = self.get_common_times(session)
        task_duration = task_end - task_start
        adjusted_duration = task_duration - self.sample_duration_ms
        sample_start = task_start + adjusted_duration * torch.rand(()).item()
        return sample_start

    def get_annotators(self, session: str) -> List[str]:
        sample = self._seek_sample(session)
        return [ *sample.annotators ] # Safer to return a copy.


class VideoManager(KFoldsManager):
    """
    Handles a collection of folds. Instantiates `VideoDataset` as needed.

    * Use `leave_one_out(k)` to obtain k-folds train and validation sets.
    * `test_transform` is used for validation and test sets.
    """

    # External parameters, assigned in constructor.
    video_root      : Path
    annotation_file : Path
    stats_file      : Path
    task            : str
    variable        : str
    sample_duration : float
    subdivision     : int
    # mypy is too grumpy to allow me to assign to a callable, it thinks I'm trying to assign to a member function.
    # train_transform : TensorMap
    # test_transform  : TensorMap
    batch_size      : Optional[int]
    num_workers     : int
    verbose         : bool

    # Internal data, filled in _init_folds()
    _train_folds : List[VideoFold]
    _test_fold   : VideoFold
    _num_folds   : int
    _sessions    : List[str]
    _annotators  : List[str]
    _fps         : float

    def __init__(
        self,
        video_root      : Path,
        annotation_file : Path,
        stats_file      : Path,
        task            : str,
        variable        : str,
        sample_duration : float,
        subdivision     : int,
        train_transform : TensorMap,
        test_transform  : TensorMap,
        batch_size      : Optional[int] = None,
        num_workers     : int           = 0,
        verbose         : bool          = True,
    ):
        assert video_root.is_dir()
        assert annotation_file.is_file()
        assert stats_file.is_file()
        assert task in [ "people", "eggs", "drums" ]
        assert variable in [ "attending", "participating" ], f"Expected `variable` to be either 'attending' or 'participating'. Found: '{variable}'."
        assert sample_duration > 0.0
        assert subdivision > 0

        self.video_root      = video_root
        self.annotation_file = annotation_file
        self.stats_file      = stats_file
        self.task            = task
        self.variable        = variable
        self.sample_duration = sample_duration
        self.subdivision     = subdivision
        self.train_transform = train_transform
        self.test_transform  = test_transform
        self.batch_size      = batch_size
        self.num_workers     = num_workers
        self.verbose         = verbose

        self._init_folds()

    def _get_fold_info(self, annotations: pd.DataFrame, stats: dict, k: int) -> VideoFold:
        """
        Loads the information for a fold from its annotation file and path.
        """

        entry = stats["stats"][k]

        empirical_probability = float(entry["empirical_probability"][self.task][self.variable])
        pixel_mean = torch.tensor(entry["pixel_values"]["mean"])
        pixel_std  = torch.tensor(entry["pixel_values"]["std" ])

        annotations = annotations.loc[k]
        sessions = annotations.index.get_level_values("session").sort_values().unique().tolist()

        samples : List[VideoSample] = []
        for session in sessions:
            label_data = annotations.loc[session]

            annotators = label_data.index.get_level_values("annotator").sort_values().unique().tolist()
            assert (len(annotators) == 2) or (len(annotators) == 3), f"At session '{session}': expected 2 or 3 annotators, found {len(annotators)}: {annotators}"

            with iio.imopen(self.video_root / f"{session}.mp4", "r") as file:
                frames = file.read()
                meta   = file.metadata(exclude_applied=False)

            assert (len(frames.shape) == 4) and (frames.shape[3] == 3) # THWC

            frames = torch.tensor(frames).float() # THWC
            frames = (frames - pixel_mean) / pixel_std
            frames = frames.permute(3, 0, 1, 2) # THWC => CTHW
            frames = frames.contiguous()

            assert ("fps" in meta) and (type(meta["fps"]) == float) and (meta["fps"] > 0)

            fps = meta["fps"]
            if self._fps < 0: # Signal value.
                self._fps = fps
            else:
                assert self._fps == fps, f"Expected all videos to have the same framerate. So far found {self._fps}, but session '{session}' has {fps}."

            samples.append(VideoSample(session, annotators, label_data, frames))

        return VideoFold(samples, empirical_probability, pixel_mean, pixel_std)

    def _init_folds(self) -> None:
        """
        Reads all the data in, and classifies it for usage.

        * LOADS ALL VIDEOS INTO MEMORY! This might take a while!
        * Reads annotation file and breaks it down into per-session chunks.
        """
        annotations : pd.DataFrame = pd.read_csv(self.annotation_file, index_col=["task", "variable"]).sort_index().loc[(self.task, self.variable)].reset_index(drop=True)

        num_folds  : int       = len(annotations["fold"].unique()) # Including the test fold.
        sessions   : List[str] = annotations["session"  ].sort_values().unique().tolist()
        annotators : List[str] = annotations["annotator"].sort_values().unique().tolist()

        annotations = annotations.set_index(["fold", "session", "annotator"]).sort_index()

        with open(self.stats_file, "r") as file:
            stats = json.load(file)

        self._fps = -1.0 # Signal value. Should be overwritten in _get_fold_info().

        iterator = trange(num_folds, desc="Loading Fold", leave=False) if self.verbose else range(num_folds)
        self._train_folds = []
        for k in iterator:
            fold = self._get_fold_info(annotations, stats, k)
            if k == num_folds - 1:
                self._test_fold = fold
            else:
                self._train_folds.append(fold)

        assert self._fps > 0.0
        assert len(self._train_folds) == num_folds - 1 # All but the last (test).

        self._num_folds  = len(self._train_folds) # Excluding the test fold.
        self._sessions   = sessions
        self._annotators = annotators

    def _make_dataset(self, fold_info: List[VideoFold], is_train: bool) -> VideoDataset:
        """
        * Creates a new `VideoDataset` instance from the given `fold_info` list.
        * Uses `is_train` to determine which transform to use.
        """

        samples : List[VideoSample] = []

        for fold in fold_info:
            samples.extend(fold.samples)

        transform = self.train_transform if is_train else self.test_transform
        dataset   = VideoDataset(samples, transform, self._fps, self.sample_duration, self.subdivision)

        return dataset

    def _make_loader(self, fold_info: List[VideoFold], is_train: bool) -> DataLoader:
        """
        * Creates a new `VideoDataset` instance from the given `fold_info` list.
        * Returns a `DataLoader` that wraps the dataset (settings determined by this class's constructor).
        * Uses `is_train` to determine which transform to use and if the data should be shuffled.
        """

        shuffle = True if is_train else False

        dataset = self._make_dataset(fold_info, is_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle , num_workers=self.num_workers)

        return loader

    def num_folds(self) -> int:
        """
        Number of training / validation folds (not counting the testing fold).
        """
        return self._num_folds

    def one_fold_loader(self, k: int, is_train=True) -> DataLoader:
        """
        * Returns a loader for fold `k`.
        * Use `is_train` to determine the data augmentation pipeline.
        """

        assert self._num_folds > 0
        assert 0 <= k < self._num_folds

        return self._make_loader([self._train_folds[k]], is_train)

    def leave_one_out(self, k: int) -> Tuple[DataLoader, DataLoader]:
        """
        * Keeps fold `k` as a validation set.
        * Merges all other folds into a training set.
        * Returns `(train_loader, val_loader)`
        """

        assert self._num_folds > 0
        assert 0 <= k < self._num_folds

        train_info : List[VideoFold] = []
        val_info   : VideoFold

        for idx in range(self._num_folds):
            if idx == k:
                val_info = self._train_folds[idx]
            else:
                train_info.append(self._train_folds[idx])

        train_loader : DataLoader = self._make_loader(train_info, is_train=True)
        val_loader   : DataLoader = self._make_loader([val_info], is_train=False)

        return (train_loader, val_loader)

    def test_set(self) -> VideoDataset:
        """
        Creates a new instance of the test `VideoDataset`.
        """
        return self._make_dataset([self._test_fold], is_train=False)

    def test_loader(self) -> DataLoader:
        """
        Creates a new instance of the test `VideoDataset` and wraps it in a `DataLoader`.
        """
        return self._make_loader([self._test_fold], is_train=False)


    def full_train_loader(self) -> DataLoader:
        """
        Joins all non-test folds into a single training fold.

        * Use only for final training! For validation tasks, use `leave_one_out(k)`.
        """
        return self._make_loader(self._train_folds, is_train=True)

    def leave_one_out_prob(self, k: int) -> float:
        """
        * Returns the empirical probability of the training set returned by `leave_one_out()`.
        """

        assert self._num_folds > 0
        assert 0 <= k < self._num_folds

        sample_count     : int   = 0
        positive_samples : float = 0

        for idx in range(self._num_folds):
            if idx == k:
                continue
            else:
                info = self._train_folds[idx]
                sample_count += len(info.samples)
                positive_samples += info.empirical_probability * len(info.samples)

        return positive_samples / sample_count

    def full_train_prob(self) -> float:
        """
        * Returns the empirical probability of the full training set returned by `full_train_loader()`.
        """

        assert self._num_folds > 0

        sample_count     : int   = 0
        positive_samples : float = 0

        for idx in range(self._num_folds):
            info = self._train_folds[idx]
            sample_count += len(info.samples)
            positive_samples += info.empirical_probability * len(info.samples)

        return positive_samples / sample_count
