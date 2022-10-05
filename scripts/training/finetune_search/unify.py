#!/bin/env -S python3 -u


import sys
from pathlib import Path
import shutil
import json

import pandas as pd

from local import search


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <output folder>", file=sys.stderr)
        exit(1)

    out_root = Path(sys.argv[1])
    assert out_root.is_dir()

    (out_root / "notes.md").touch()

    task_root = out_root / "tasks"
    assert task_root.is_dir()
    tasks = [ item for item in task_root.iterdir() if item.is_dir() ]
    tasks.sort()
    assert len(tasks) > 1

    shutil.copy(tasks[0] / "script_params.json", out_root / "script_params.json")

    data      : pd.DataFrame = pd.DataFrame()
    best_f1   : float        = -1.0
    best_task : Path

    for path in tasks:
        task_id = int(path.stem) # Assumption: task artifacts saved in <output folder>/tasks/<numerical task ID>/

        task_data = pd.read_csv(path / "stats.csv")
        task_data["task_id"] = task_id
        data = pd.concat([data, task_data], ignore_index=True)

        with open(path / "results.json", "r") as f:
            summary = json.load(f)

        task_f1 = float(summary["Best_Avg_Val_F1"])
        if task_f1 > best_f1:
            best_f1   = task_f1
            best_task = path

    data.to_csv(out_root / "stats.csv", index=False)
    shutil.copy(best_task / "best_net.pt", out_root / "best_net.pt")

    search.stats_and_plots(out_root, data)


if __name__ == "__main__":
    main()
