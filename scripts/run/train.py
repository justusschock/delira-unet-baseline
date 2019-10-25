import torch
from batchgenerators.transforms import RangeTransform, Compose, \
    ZeroMeanUnitVarianceTransform
from delira_unet import UNetTorch, RAdam, dice_score_including_background, \
    SoftDiceLossPyTorch
from delira.training.callbacks import ReduceLROnPlateauCallbackPyTorch
from delira.data_loading.sampler import SequentialSampler, RandomSampler
from delira.data_loading import DataManager
from delira import get_current_debug_mode
from delira.utils import DeliraConfig
from delira.training import PyTorchExperiment
from sklearn.model_selection import train_test_split
import os
import logging
import sys
logging.basicConfig(level=logging.INFO,
                    handlers=[logging.StreamHandler(sys.stderr)])
logger = logging.getLogger("Execute_Logger")


# PARAMETERS TO CHANGE:
# TODO: Paths and create suitable config
data_path = ""
save_path = ""

config = DeliraConfig(
    fixed_model={
        "in_channels": 1,
        "num_classes": 3,
        "norm_layer": "Instance",
        "per_class": False
    },
    fixed_training={
        "optimizer_cls": RAdam,
        "optimizer_params": {"lr": 1e-3},
        "lr_sched_cls": ReduceLROnPlateauCallbackPyTorch,
        "lr_sched_params": {"mode": "max", "patience": 5},
        "losses": {
            "ce": torch.nn.CrossEntropyLoss(),
            "soft_dice": SoftDiceLossPyTorch(non_lin=torch.nn.Softmax(dim=1))
        },
        "metrics": {"dice": dice_score_including_background},
        "num_epochs": 300,  # number of epochs to train
    },
    val_split=0.2,
    seed=0,
    batchsize=1,
    checkpoint_freq=1,
    gpu_ids=[0],
    val_score_key="dice",
    val_score_mode="highest",
    verbose=True,
    metric_keys={"dice": ("pred", "label")}
)

base_transforms = [
    # HistogramEqualization(),
    RangeTransform((-1, 1))
]
train_specific_transforms = []
test_specific_transforms = []


train_transforms = Compose(base_transforms + train_specific_transforms)
test_transforms = Compose(base_transforms + test_specific_transforms)


train_dir = "train"
test_dir = "test"

data_path = os.path.expanduser(data_path)
save_path = os.path.expanduser(save_path)

train_path = os.path.join(data_path, train_dir)
test_path = os.path.join(data_path, test_dir)

# TODO: Change this to your custom Dataset (derived from
# delira.data_loading.AbstractDataset)
dset = None
dset_test = None

idx_train, idx_val = train_test_split(
    list(range(len(dset))), test_size=config.val_split,
    random_state=config.seed)

dset_train = dset.get_subset(idx_train)
dset_val = dset.get_subset(idx_val)

mgr_train = DataManager(dset_train, config.batchsize, 4, train_transforms,
                        sampler_cls=RandomSampler)
mgr_val = DataManager(dset_val, config.batchsize, 4, test_transforms,
                      sampler_cls=SequentialSampler)
mgr_test = DataManager(dset_test, config.batchsize, 4, test_transforms,
                       sampler_cls=SequentialSampler)

experiment = PyTorchExperiment(config, UNetTorch, name="BaselineUnet",
                               save_path=save_path,
                               checkpoint_freq=config.checkpoint_freq,
                               gpu_ids=config.gpu_ids,
                               val_score_key=config.val_score_key,
                               metric_keys=config.metric_keys)

experiment.save()
net = experiment.run(mgr_train, mgr_val, val_score_mode=config.val_score_mode,
                     verbose=config.verbose)
net.eval()
experiment.test(net, mgr_test, verbose=config.verbose,
                metrics=config.nested_get("metrics"),
                metric_keys=config.metric_keys)
