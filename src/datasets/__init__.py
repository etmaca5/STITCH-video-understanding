from .base import EvalSample, EvalDataset
from .qvhighlights import QVHighlightsDataset
from .ego4d_nlq import Ego4dNlqDataset
from .ego4d_mq import Ego4dMqDataset
from .activitynet import ActivityNetCaptionsDataset
from .activitynet_qa import ActivityNetQADataset
from .custom_dataset import CustomDataset
from .lovr import LoVRDataset
from .longvideobench import LongVideoBenchDataset
from .lvbench import LVBenchDataset
from .videomme import VideoMMEDataset
from .mlvu import MLVUDataset
from .kinetics_gebd import KineticsGEBDDataset

DATASETS = {
    "qvhighlights": QVHighlightsDataset,
    "ego4d_nlq": Ego4dNlqDataset,
    "ego4d_mq": Ego4dMqDataset,
    "activitynet": ActivityNetCaptionsDataset,
    "activitynet_qa": ActivityNetQADataset,
    "custom_dataset": CustomDataset,
    "lovr": LoVRDataset,
    "longvideobench": LongVideoBenchDataset,
    "lvbench": LVBenchDataset,
    "videomme": VideoMMEDataset,
    "mlvu": MLVUDataset,
    "kinetics_gebd": KineticsGEBDDataset,
}


def build_dataset(cfg):
    """Instantiate the dataset specified by *cfg.dataset.name*."""
    cls = DATASETS[cfg.dataset.name]
    return cls(cfg)
