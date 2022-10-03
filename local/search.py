from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from torch import nn

from local.cli.cli_helpers import CliHelper
from local.training_helper import TrainingHelper


@dataclass
class SearchResults:
    best_avg_val_f1 : float
    best_params     : TrainingHelper.Params
    best_net        : nn.Module
    stats           : pd.DataFrame


def report_results(helper: CliHelper, out_path: Path, start_time: datetime, search_results: SearchResults) -> None:
    """
    1. Save stats to <out_path>/stats.csv
    2. Report durations and basic training statistics.
    3. Save training summary to <out_path>/results.json
    """

    stats = search_results.stats

    end_time = datetime.now()
    elapsed = end_time - start_time

    stats.to_csv(out_path / "stats.csv", index=False)

    summary = {
        "General": {
            "Started"      : start_time,
            "Ended"        : end_time,
            "Total_Time"   : str(elapsed),
            "Time_per_Rep" : stats["duration"].describe().to_dict(),
            "F1" : {
                "Train"      : stats["train_f1"     ].describe().to_dict(),
                "Validation" : stats["validation_f1"].describe().to_dict(),
                "Test"       : stats["test_f1"      ].describe().to_dict(),
            },
        },
        "Best_Avg_Val_F1" : search_results.best_avg_val_f1,
        "Best_Params"     : asdict(search_results.best_params),
    }

    helper.cli.section("Final Results")
    helper.cli.print(summary)
    helper.json_write(summary, out_path / "results.json")


def stats_and_plots(out_path: Path, stats: pd.DataFrame) -> None:
    """
    1. Find the existing hyper-parameters in stats (optional parameters might be missing).
    2. Extract the average validation F1 score per hyper-parameter combination from stats.
    3. Save the top 10 hyper-parameter combinations to <out_path>/top_hyperparams.csv
    4. Plot each individual run as a point in (validation F1 score as function of hyper-parameter value) for each hyper-parameter. Save in <out_path>/figures/param_*.png
    5. Plot each hyper-parameter combination as a point in (average validation F1 score as function of hyper-parameter value) for each hyper-parameter. Save in <out_path>/figures/average_*.png
    """

    param_keys = [ key for key in TrainingHelper.param_names() if key in stats ] # Safety filter: avoid skipped params (optimizer_name).

    val_scores: pd.DataFrame = stats[[ *param_keys, "average_validation_f1" ]].drop_duplicates()
    val_scores = val_scores.sort_values(by="average_validation_f1", ascending=False).reset_index(drop=True)
    val_scores.iloc[:10].to_csv(out_path / "top_hyperparams.csv", index=False) # Top 10 judged by avg. val. F1

    fig_path = out_path / "figures"
    fig_path.mkdir(exist_ok=False, parents=False)

    fig, ax = plt.subplots()
    sns.violinplot(data=val_scores["average_validation_f1"], orient="h", ax=ax)
    plt.title("Average validation F1 score per Parameter Combination")
    plt.xlim(0, 1)
    plt.xlabel("F1 Score")
    fig.savefig(fig_path / "mean_f1_distribution.png")
    plt.close(fig)

    for param in param_keys:
        fig, ax = plt.subplots()
        sns.scatterplot(x=param, y="validation_f1", data=stats, ax=ax)
        xscale = "linear" if (param == "use_class_weights") else "log"
        ax.set(xscale=xscale)
        fig.savefig(fig_path / f"param_{param}.png")
        plt.close(fig)

    for param in param_keys:
        fig, ax = plt.subplots()
        sns.scatterplot(x=param, y="average_validation_f1", data=val_scores, ax=ax)
        xscale = "linear" if (param == "use_class_weights") else "log"
        ax.set(xscale=xscale)
        fig.savefig(fig_path / f"average_{param}.png")
        plt.close(fig)
