from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from imageio import v3 as iio

import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
from matplotlib.backend_bases import MouseEvent, MouseButton
from matplotlib.patches import Circle

import ipywidgets as widgets

from .painting import gaussian_kernel, add_template
from .timer import MakeTimer
from local.cli.cli_helpers import json_default


FULL_WIDTH  = widgets.Layout(width="100%")
FIT_WIDTH   = widgets.Layout(width="fit-content")
CONTAINER   = widgets.Layout(min_width="30rem", width="fit-content", overflow="visible")
FLEX_CENTER = widgets.Layout(display="flex", justify_content="center", width="100%")

INITIAL_CURSOR_FRACTION : float = 1 / 25
MOUSEWHEEL_SCALE_FACTOR = 1.15


class Annotator:

    sample_duration : float # Setting passed in constructor; snippet duration in seconds.
    video_root      : Path  # Setting passed in constructor; folder containing all session videos.
    annotation_root : Path  # Setting passed in constructor; folder used to save the attention painting annotations created with this tool.
    snippet_file    : Path  # Setting passed in constructor; CSV file containing the candidate snippet data.

    snippet_data    : pd.DataFrame # Candidate snippet data.
    snippet_idx     : int          # Index into snippet_data corresponding to the current item.

    sample_path     : Path       # Path to the video of the current session (the MP4 that the current snippet comes from).
    sample_start    : float      # Current snippet offset from the beginning of the session video (in seconds).
    session         : str        # Session that the current snippet comes from.
    annotator       : str        # Annotator intended to paint on this sample (should be the only annotator who hasn't classified this snippet in the ELAN phase).
    variable        : str        # Variable that was targeted when the current snippet was selected (other annotators showed consensus).
    label           : float      # Consensus label that the other annotators selected for the current snippet and variable.
    fps             : float      # fps of the current snippet.
    vid             : np.ndarray # Frames showing the current snippet.
    heatmap         : np.ndarray # Frames that we paint the attention on.
    frame_idx       : int        # Current frame being displayed.

    kernel_size     : float      # Variable used to control the paint brush size; standard deviation of the Gaussian kernel.
    kernel          : np.ndarray # Scaled Gaussian kernel, ready to be copy-pasted into the heatmap image.

    participating   : str # Currently chosen "participating" value for this snippet in the interface.
    attending       : str # Currently chosen "attending" value for this snippet in the interface.

    fig             : plt.Figure # Handle to the PyPlot figure showing the current frame and heatmap.
    img             : AxesImage  # Handle to the image data for the figure.
    circle          : Circle     # Handle to the mouse pointer (indicates kernel size).

    playing         : bool # Used for media controls; if true, the current frame changes in real time according to the fps.

    out             : widgets.Output    # Used to capture exceptions and other console output into the widget.
    slider          : widgets.IntSlider # Frame slider. Used to scrub through the video.
    widget          : widgets.Widget    # Main widget handle.
    mouse_pressed   : bool              # Used to control mouse behavior in the plot (painting).


    def __init__(self, sample_duration: float, video_root: Path, annotation_root: Path, snippet_file: Path):
        assert sample_duration > 0.0
        assert video_root.is_dir()
        assert annotation_root.is_dir()
        assert snippet_file.is_file()

        self.sample_duration = sample_duration
        self.video_root      = video_root
        self.annotation_root = annotation_root
        self.snippet_file    = snippet_file

        self.snippet_data  = pd.read_csv(self.snippet_file).set_index(["annotator", "session", "start_ms"]).sort_index().reset_index()
        assert len(self.snippet_data) > 0

        self.snippet_idx   = 0
        self.frame_idx     = 0
        self.mouse_pressed = False
        self.mouse_add     = True # True: add to heatmap. False: remove from heatmap.
        self.playing       = False

        self.session   = None
        self.annotator = None
        self.variable  = None

        self.out    = None
        self.slider = None
        self.widget = None

        self.fig    = None
        self.img    = None
        self.circle = None

        self.participating = "no"
        self.attending     = "no"

        self.load_sample(0)


    def load_sample(self, idx: int) -> None:
        assert 0 <= idx < len(self.snippet_data)

        entry = self.snippet_data.iloc[idx]

        sample_path = self.video_root / f"{entry['session']}.mp4"
        start_time  = entry["start_ms"] / 1_000
        session     = entry["session"]
        annotator   = entry["annotator"]
        variable    = entry["variable"]
        label       = entry["label"]

        assert sample_path.is_file()
        assert 0 <= start_time < 60 * 60 # 1 hour (known upper bound)

        meta = iio.immeta(sample_path, exclude_applied=False)

        fps = meta["fps"]
        assert (type(fps) == float) and (1 < fps)

        duration_frames = int(self.sample_duration * fps)
        start_idx = int(fps * start_time)
        end_idx = start_idx + duration_frames
        end_time = end_idx / fps

        if "duration" in meta:
            vid_duration = meta["duration"]
            assert (type(vid_duration) == float) and (end_time < vid_duration)

        vid = iio.imread(sample_path, index=None)
        vid = vid[start_idx:end_idx]

        (t, h, w, c) = vid.shape
        heatmap = np.zeros((t, h, w), dtype=np.float32)

        # TODO: Pass the instantaneous intensity as as parameter?
        self.set_kernel_size(INITIAL_CURSOR_FRACTION * min(w, h))

        self.snippet_idx  = idx
        self.sample_path  = sample_path
        self.sample_start = start_time
        self.session      = session
        self.annotator    = annotator
        self.variable     = variable
        self.label        = label
        self.fps          = fps
        self.vid          = vid
        self.heatmap      = heatmap
        self.frame_idx    = 0
        self.playing      = False

        if self.widget:
            self.slider.max = len(self.vid) - 1
            self.redraw()
        else:
            self.assemble_widget()


    def set_kernel_size(self, kernel_size: float) -> None:
        kernel = gaussian_kernel(kernel_size)
        kernel = (0.1 * kernel) / kernel.max() # Adjust intensity.

        self.kernel_size = kernel_size
        self.kernel      = kernel

        if self.circle is not None:
            self.circle.radius = self.kernel_size


    def assemble_widget(self) -> None:
        self.out = widgets.Output()

        plot              = self.assemble_plot()
        slider            = self.assemble_slider()
        playback_controls = self.assemble_playback_controls()
        paint_controls    = self.assemble_paint_controls()
        label_chooser     = self.assemble_label_chooser()
        file_controls     = self.assemble_file_controls()

        _widget = widgets.VBox(
            [plot, slider, playback_controls, paint_controls, label_chooser, file_controls],
            layout=CONTAINER
        )

        self.widget = widgets.VBox([ _widget, self.out ])

        self.redraw()


    def assemble_plot(self) -> widgets.Widget:
        plt.ioff()

        fig, ax = plt.subplots(1)

        fig.canvas.layout         = FULL_WIDTH
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        fig.canvas.resizable      = False
        fig.canvas.capture_scroll = True

        ax.axis("off")
        img = ax.imshow(self.vid[self.frame_idx])
        circle = Circle((0, 0), self.kernel_size, fill=False)
        ax.add_artist(circle)

        plt.connect("motion_notify_event", self.on_move)
        plt.connect("button_press_event", self.on_mouse_pressed)
        plt.connect("button_release_event", self.on_mouse_released)
        plt.connect("scroll_event", self.on_mouse_wheel)

        self.img = img
        self.fig = fig
        self.circle = circle

        return fig.canvas


    def assemble_slider(self) -> widgets.Widget:
        slider = widgets.IntSlider(max=len(self.vid)-1, description="Frame", layout=FULL_WIDTH)
        slider.observe(lambda change: self.change_frame(change["new"]), names="value")

        self.slider = slider
        return slider


    def assemble_playback_controls(self) -> widgets.Widget:
        play  = widgets.Button(icon="play")
        pause = widgets.Button(icon="pause")
        stop  = widgets.Button(icon="stop")

        controls = widgets.HBox([play, pause, stop], layout=FLEX_CENTER)

        def callback():
            if self.playing:
                self.slider.value = (self.frame_idx + 1) % len(self.vid)

        timer = MakeTimer(1 / self.fps, callback)
        timer.start()

        play .on_click(lambda _: self.play_media())
        pause.on_click(lambda _: self.pause_media())
        stop .on_click(lambda _: self.stop_media())

        return controls


    def assemble_paint_controls(self) -> widgets.Widget:
        clear   = widgets.Button(description="Clear Frame")
        back    = widgets.Button(icon="step-backward")
        forward = widgets.Button(icon="step-forward")

        controls = widgets.HBox([clear, back, forward], layout=FLEX_CENTER)

        def back_callback():
            self.slider.value = (self.slider.value - 1) % len(self.vid)

        def forward_callback():
            self.slider.value = (self.slider.value + 1) % len(self.vid)

        clear.on_click(lambda _: self.clear_current_frame())
        back.on_click(lambda _: back_callback())
        forward.on_click(lambda _: forward_callback())

        return controls


    def assemble_label_chooser(self) -> widgets.Widget:
        participating = widgets.ToggleButtons(
            options=["No", "Self", "Joint"],
            description="Participating"
        )
        attending = widgets.ToggleButtons(
            options=["No", "Attending", "Excited"],
            description="Attending"
        )
        container = widgets.VBox([participating, attending], layout=FIT_WIDTH)

        def participating_callback(change):
            with self.out:
                val = change["new"].lower()
                self.participating = val

        def attending_callback(change):
            with self.out:
                val = change["new"].lower()
                self.attending = val

        participating.observe(participating_callback, names="value")
        attending    .observe(attending_callback    , names="value")

        return container


    def assemble_file_controls(self) -> widgets.Widget:

        load_values = [(f"{entry['annotator'].upper()}: {entry['session']} @ {int(entry['start_ms'] / 1_000)}s ({entry['variable']})", idx) for (idx, entry) in self.snippet_data.iterrows() ]

        sample_label    = widgets.Label(value="Current Sample:")
        sample_selector = widgets.Dropdown(options=load_values)
        save_button     = widgets.Button(description="Save")
        container       = widgets.HBox([sample_label, sample_selector, save_button], layout=FLEX_CENTER)

        def load_callback(change):
            with self.out:
                value = change.new
                self.load_sample(value)

        def save_callback():
            with self.out:
                path = self.annotation_root / f"snippet_{self.snippet_idx}__{self.annotator}_{self.session}_{int(self.sample_start)}s_{self.variable}"
                self.save_data(path)

        sample_selector.observe(load_callback, names="value")
        save_button.on_click(lambda _: save_callback())

        return container


    def redraw(self) -> None:
        merged = self.merge_frame(self.frame_idx)
        self.img.set_data(merged)
        self.fig.canvas.draw_idle()


    def merge_frame(self, idx: int) -> np.ndarray:
        frame = self.vid    [idx]
        map   = self.heatmap[idx]

        # NOTE: Currently they are both the same size, so this is a waste of compute and resolution.
        # (h, w, c) = frame.shape
        # overlay = cv2.resize(map, (w, h), interpolation=cv2.INTER_NEAREST)

        color = np.clip(256 * map, 0, 255).astype(np.uint8)
        color = cv2.applyColorMap(color, cv2.COLORMAP_JET)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

        merged = 0.75 * frame + 0.25 * color
        merged = merged.astype(np.uint8)

        return merged


    def save_data(self, path: Path) -> None:
        if not path.exists():
            path.mkdir(parents=False, exist_ok=False)
        assert path.is_dir()

        info = {
            "saved": str(datetime.now()),
            "sample": {
                "session":         self.session,
                "annotator":       self.annotator,
                "start_time_s":    self.sample_start,
                "target_variable": self.variable,
                "consensus_label": self.label,
            },
            "labels": {
                "participating": self.participating,
                "attending":     self.attending,
            },
        }

        with open(path / "info.json", "w") as file:
            json.dump(info, file, indent=4, default=json_default)

        np.save(path / "heatmap.npy", self.heatmap)

        vid = np.stack([ self.merge_frame(i) for i in range(len(self.vid)) ])
        iio.imwrite(path / "visualization.mp4", vid, fps=self.fps)


    def on_mouse_pressed(self, event: MouseEvent) -> None:
        with self.out:
            if event.button == MouseButton.LEFT:
                self.mouse_pressed = True
                self.mouse_add     = True
            elif event.button == MouseButton.RIGHT:
                self.mouse_pressed = True
                self.mouse_add     = False
            else:
                print(f"Mouse button pressed: {event.button}")


    def on_mouse_released(self, event: MouseEvent) -> None:
        with self.out:
            self.mouse_pressed = False


    def on_mouse_wheel(self, event: MouseEvent) -> None:
        # TODO: Test
        # TODO: Add GUI indicator of size.
        with self.out:
            new_size: float
            if event.button == "up":
                new_size = self.kernel_size * MOUSEWHEEL_SCALE_FACTOR
                self.set_kernel_size(new_size)
            elif event.button == "down":
                new_size = self.kernel_size / MOUSEWHEEL_SCALE_FACTOR
                self.set_kernel_size(new_size)


    def on_move(self, event) -> None:
        with self.out:
            if (event.xdata == None) or (event.ydata == None):
                return
            x, y = event.xdata, event.ydata

            self.circle.center = (x, y)

            if self.mouse_pressed:
                sign = +1 if self.mouse_add else -1
                add_template(self.heatmap, sign * self.kernel, x, y, self.frame_idx)
                self.redraw()
            else:
                self.fig.canvas.draw_idle() # Cheaper version of self.redraw() ?


    def change_frame(self, idx: int) -> None:
        with self.out:
            self.frame_idx = idx
            self.redraw()


    def play_media(self) -> None:
        with self.out:
            self.playing = True


    def pause_media(self) -> None:
        with self.out:
            self.playing = False


    def stop_media(self) -> None:
        with self.out:
            self.playing = False
            self.slider.value = 0


    def clear_current_frame(self) -> None:
        self.heatmap[self.frame_idx] = 0
        self.redraw()


    def display(self) -> widgets.Widget:
        if self.widget == None:
            self.assemble_widget()

        return self.widget
