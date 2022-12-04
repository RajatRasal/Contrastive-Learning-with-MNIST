import logging
from typing import Dict, List, Optional, Union

from pytorch_lightning import Callback, Trainer, LightningModule
from ray import air, tune
from ray.tune import CLIReporter

from components import _ACTIVATIONS, _POOLING
from train import get_trainer, train_mnist_classifier

logger = logging.getLogger(__name__)


class TuneReportCallbackOnValidationEnd(Callback):

    def __init__(self, metrics: Optional[Union[str, List[str], Dict[str, str]]] = None):
        if isinstance(metrics, str):
            metrics = [metrics]
        self._metrics = metrics

    def _get_report_dict(self, trainer: Trainer, pl_module: LightningModule):
        # Don't report if just doing initial validation sanity checks.
        if trainer.sanity_checking:
            return
        if not self._metrics:
            report_dict = {k: v.item() for k, v in trainer.callback_metrics.items()}
        else:
            report_dict = {}
            for key in self._metrics:
                if isinstance(self._metrics, dict):
                    metric = self._metrics[key]
                else:
                    metric = key
                if metric in trainer.callback_metrics:
                    report_dict[key] = trainer.callback_metrics[metric].item()
                else:
                    logger.warning(
                        f"Metric {metric} does not exist in "
                        "`trainer.callback_metrics."
                    )

        return report_dict

    def on_validation_end(self, trainer, pl_module):
        report_dict = self._get_report_dict(trainer, pl_module)
        if report_dict is not None:
            tune.report(**report_dict)


def _train_mnist_classifier(config):
    trainer = get_trainer(
        max_epochs=config["max_epochs"],
        val_check_freq=config["val_check_freq"],
        callbacks=[
            TuneReportCallbackOnValidationEnd(
                metrics={
                    "val_acc": "Validation Accuracy",
                    "train_acc": "Training Accuracy",
                },
            ),
        ],
    )
    del config["max_epochs"]
    del config["val_check_freq"]
    train_mnist_classifier(trainer, **config)

def tune_mnist_classifier():
    max_epochs = 20
    config = {
        "activation": tune.grid_search(list(_ACTIVATIONS.keys())),
        "pooling": tune.grid_search(list(_POOLING.keys())),
        "batch_size": 256,
        "lr": tune.grid_search([0.01, 0.03, 0.05, 0.07, 0.09]),
        "max_epochs": max_epochs,
        "val_check_freq": min(5, max_epochs),
        "seed": 1234,
    }

    reporter = CLIReporter(
        parameter_columns=["activation", "pooling", "lr"],
        metric_columns=["val_acc", "train_acc"],
    )

    tuner = tune.Tuner(
        _train_mnist_classifier,
        tune_config=tune.TuneConfig(
            metric="val_acc",
            mode="max",
            num_samples=1,
            max_concurrent_trials=3,
        ),
        run_config=air.RunConfig(
            name="tune_mnist_cls",
            local_dir="./ray_logs",
            progress_reporter=reporter,
        ),
        param_space=config,
    )
    results = tuner.fit()
    print(results.get_best_result())


if __name__ == "__main__":
    tune_mnist_classifier()
