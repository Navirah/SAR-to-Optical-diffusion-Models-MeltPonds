# datasets/reader.py
import os, random, sys, logging, contextlib, io
from typing import Tuple, List

#global noise suppression (tensorflow / absl)

# 0 = all logs, 1 = hide INFO, 2 = hide WARNING, 3 = hide ERROR
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except Exception:
    pass

# silence python-side TF logger if present
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# local project imports
from ..experiment import Experiment
from utils import toolbox as tbx  # kept for legacy compatibility

#project logger

LOGGER_NAME = "s1s2.data"
logger = logging.getLogger(LOGGER_NAME)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.WARNING)


def _apply_debug_level_to_logger(debug_level: int):
    """Map numeric debug level to python logging verbosity."""
    level = (
        logging.WARNING if debug_level <= 0
        else logging.INFO if debug_level == 1
        else logging.DEBUG
    )
    logger.setLevel(level)


@contextlib.contextmanager
def _maybe_silence_stdio(enabled: bool):
    """Optionally suppress stdout/stderr for noisy parsers."""
    if not enabled:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


#helpers

def _bilinear_resize(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Resize tensor (C,H,W) using bilinear interpolation."""
    return F.interpolate(
        x.unsqueeze(0),
        size=(h, w),
        mode="bilinear",
        align_corners=False
    ).squeeze(0)


def _validate_tensor(x: torch.Tensor, name: str, eps: float = 1e-6):
    """Basic sanity checks on loaded tensors."""
    if not torch.is_tensor(x):
        raise TypeError(f"{name} is not a torch.Tensor")
    if x.ndim != 3:
        raise ValueError(f"{name} must be (C,H,W), got {tuple(x.shape)}")
    if torch.isnan(x).any():
        raise ValueError(f"{name} contains NaNs")
    if torch.isinf(x).any():
        raise ValueError(f"{name} contains Infs")
    if x.abs().sum().item() < eps:
        raise ValueError(f"{name} appears to be all zeros (sum<{eps})")


def _auto_order_pair(
    t1: torch.Tensor,
    t2: torch.Tensor,
    exp: Experiment
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Infer which tensor is input (S1) and which is target (S2)
    based on expected channel counts.
    """
    c1, c2 = t1.shape[0], t2.shape[0]
    ic, oc = exp.INPUT_CHANNELS, exp.OUTPUT_CHANNELS

    if c1 == oc and c2 == ic:
        return t1, t2
    if c1 == ic and c2 == oc:
        return t2, t1

    raise ValueError(
        f"Cannot infer S1/S2 order from channels: got {c1} & {c2}, "
        f"expected input={ic}, target={oc}"
    )


def _has_stats(exp: Experiment) -> bool:
    return all(hasattr(exp, a) for a in ("S1_MEAN", "S1_STD", "S2_MEAN", "S2_STD"))


#dataset

class TFRecordDataset(Dataset):
    """
    S1 → S2 TFRecord dataset.
    Returns (input=S1, target=S2).
    """

    def __init__(
        self,
        file_list: List[str],
        exp: Experiment,
        is_train: bool = True,
        debug_level: int = 1
    ):
        if len(file_list) == 0:
            raise FileNotFoundError("No TFRecord files provided.")

        self.file_list = file_list
        self.exp = exp
        self.is_train = is_train
        self.debug_level = debug_level
        self._printed_header = False

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.file_list[idx]

        if not self._printed_header:
            logger.debug("=== DATASET DEBUG INFO ===")
            logger.debug(f"Path root: {os.path.dirname(path)}")
            logger.debug(f"Example file: {os.path.basename(path)}")
            logger.debug(
                f"Expected channels -> input={self.exp.INPUT_CHANNELS}, "
                f"target={self.exp.OUTPUT_CHANNELS}"
            )
            self._printed_header = True

        gt_image, input_image = self._load_pair(path)

        _validate_tensor(gt_image, "gt_image(raw)")
        _validate_tensor(input_image, "input_image(raw)")

        H, W = self.exp.IMG_HEIGHT, self.exp.IMG_WIDTH
        gt_image = _bilinear_resize(gt_image, H, W)
        input_image = _bilinear_resize(input_image, H, W)

        if self.is_train:
            gt_image, input_image = self._random_jitter(gt_image, input_image)

        assert input_image.shape[0] == self.exp.INPUT_CHANNELS
        assert gt_image.shape[0] == self.exp.OUTPUT_CHANNELS

        if self.debug_level >= 2 and idx % 50 == 0:
            logger.debug(
                f"[{idx}] S1 {tuple(input_image.shape)} "
                f"[{float(input_image.min()):.3g},{float(input_image.max()):.3g}] | "
                f"S2 {tuple(gt_image.shape)} "
                f"[{float(gt_image.min()):.3g},{float(gt_image.max()):.3g}]"
            )

        return input_image, gt_image

    #loading

    def _load_pair(self, file_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Try multiple TFRecord parsers and infer correct ordering."""
        errors = []

        try:
            with _maybe_silence_stdio(self.debug_level <= 1):
                from utils.pure_pytorch_tfrecord import load_float_list_tfrecord
                t1, t2 = load_float_list_tfrecord(
                    file_path,
                    self.exp.TILESIZE,
                    self.exp.TILESIZE
                )
            t1, t2 = t1.float().contiguous(), t2.float().contiguous()
            return _auto_order_pair(t1, t2, self.exp)
        except Exception as e:
            errors.append(f"float_list parser: {e}")

        try:
            with _maybe_silence_stdio(self.debug_level <= 1):
                from utils.pure_pytorch_tfrecord import load_tfrecord_as_tensor
                t1, t2 = load_tfrecord_as_tensor(
                    file_path,
                    self.exp.TILESIZE,
                    self.exp.TILESIZE
                )
            t1, t2 = t1.float().contiguous(), t2.float().contiguous()
            return _auto_order_pair(t1, t2, self.exp)
        except Exception as e:
            errors.append(f"generic parser: {e}")

        try:
            with _maybe_silence_stdio(self.debug_level <= 1):
                t1, t2 = tbx.parse_tfrecord_pytorch(file_path, self.exp.TILESIZE)
            t1, t2 = t1.float().contiguous(), t2.float().contiguous()
            return _auto_order_pair(t1, t2, self.exp)
        except Exception as e:
            errors.append(f"tbx.default: {e}")

        try:
            with _maybe_silence_stdio(self.debug_level <= 1):
                t1, t2 = tbx.parse_tfrecord_alt_pytorch(file_path, self.exp.TILESIZE)
            t1, t2 = t1.float().contiguous(), t2.float().contiguous()
            return _auto_order_pair(t1, t2, self.exp)
        except Exception as e:
            errors.append(f"tbx.alt: {e}")

        raise RuntimeError(
            f"All parsers failed for {os.path.basename(file_path)}:\n"
            + "\n".join(errors)
        )

    #augmentations

    def _random_jitter(
        self,
        gt: torch.Tensor,
        inp: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        H, W = self.exp.IMG_HEIGHT, self.exp.IMG_WIDTH
        scale = getattr(self.exp, "RANDOM_RESIZE", 1.0)

        if scale and scale > 1.0:
            h2, w2 = int(H * scale), int(W * scale)
            gt = _bilinear_resize(gt, h2, w2)
            inp = _bilinear_resize(inp, h2, w2)
            top = random.randint(0, h2 - H)
            left = random.randint(0, w2 - W)
            gt = gt[:, top:top+H, left:left+W]
            inp = inp[:, top:top+H, left:left+W]

        if getattr(self.exp, "RANDOM_ROTATE", False) and random.random() < 0.5:
            k = random.randint(1, 3)
            gt = torch.rot90(gt, k, dims=[1, 2])
            inp = torch.rot90(inp, k, dims=[1, 2])

        if random.random() < 0.5:
            gt = torch.flip(gt, dims=[2])
            inp = torch.flip(inp, dims=[2])

        return gt, inp


#reader

class Reader:
    """Build train/val dataloaders from a TFRecord root."""

    def __init__(self, experiment: Experiment, caller: str):
        self.exp = experiment
        dbg = getattr(self.exp, "DEBUG_LEVEL", 1)
        _apply_debug_level_to_logger(dbg)

        logger.info(f"Reader initialised by {caller}")

        dataset_path = self._resolve_dataset_path()
        self.TRAIN_DIR = os.path.join(dataset_path, "train")
        self.VAL_DIR = os.path.join(dataset_path, "val")

        train_files = self._collect_files(self.TRAIN_DIR)
        val_files = self._collect_files(self.VAL_DIR)

        self.train_dataset = TFRecordDataset(train_files, self.exp, True, dbg)
        self.test_dataset = TFRecordDataset(val_files, self.exp, False, dbg)

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.exp.BATCH_SIZE,
            shuffle=getattr(self.exp, "SHUFFLE", True),
            num_workers=getattr(self.exp, "NUM_WORKERS", 0),
            pin_memory=True
        )

        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.exp.BATCH_SIZE,
            shuffle=False,
            num_workers=getattr(self.exp, "NUM_WORKERS", 0),
            pin_memory=True
        )

        self.BUFFER_SIZE = len(train_files)

    def __len__(self):
        return self.BUFFER_SIZE

    def _resolve_dataset_path(self) -> str:
        dr = getattr(self.exp, "DATA_ROOT", None)
        if dr and os.path.exists(dr):
            return dr

        known_paths = [
            "/path/to/data/curated/256",
            "/path/to/data/alt12/256",
            "/path/to/data/alt30/256",
        ]
        for p in known_paths:
            if os.path.exists(p):
                return p

        raise FileNotFoundError(
            "Could not resolve dataset path. "
            "Set exp.DATA_ROOT to a valid TFRecord root."
        )
