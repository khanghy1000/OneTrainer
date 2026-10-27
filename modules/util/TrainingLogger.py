import os
from typing import Any

from modules.util.config.TrainConfig import TrainConfig

from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import pil_to_tensor

import trackio
from PIL import Image


class TrainingLogger:
    """Logs training metrics and samples."""

    def __init__(self, config: TrainConfig, tensorboard_log_dir: str):
        self.config = config
        self.tensorboard: SummaryWriter | None = None
        self.trackio = None

        os.makedirs(tensorboard_log_dir, exist_ok=True)
        self.tensorboard = SummaryWriter(
            os.path.join(tensorboard_log_dir, config.run_name)
        )

        if config.trackio:
            def strip_version_keys(obj, key="__version"):
                if isinstance(obj, dict):
                    return {k: strip_version_keys(v, key) for k, v in obj.items() if k != key}
                if isinstance(obj, list):
                    return [strip_version_keys(i, key) for i in obj]
                return obj

            project = self.config.trackio_project.strip() or "OneTrainer"
            run_name = self.config.run_name

            self.trackio = trackio.init(
                project=project,
                name=run_name,
                config=strip_version_keys(self.config.to_pack_dict(secrets=False)), # type: ignore
            )

    def add_scalar(self, tag: str, value: float, step: int):
        if self.tensorboard is not None:
            self.tensorboard.add_scalar(tag, value, step)
        if self.trackio is not None:
            self.trackio.log({tag: value}, step=step)

    def add_image(self, tag: str, image: Any, step: int):
        if self.tensorboard is not None:
            self.tensorboard.add_image(
                tag,
                pil_to_tensor(image) if isinstance(image, Image.Image) else image,
                step,
            )
        if self.trackio is not None:
            self.trackio.log({tag: trackio.Image(image)}, step=step)

    def close(self):
        if self.tensorboard is not None:
            self.tensorboard.close()
            self.tensorboard = None
        if self.trackio is not None:
            self.trackio.finish()
            self.trackio = None
