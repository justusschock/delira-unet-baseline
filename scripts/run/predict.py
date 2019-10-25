from delira_unet import UNetTorch
from delira.training import Predictor
import numpy as np
import torch
import os
from tqdm import tqdm
from batchgenerators.transforms import RangeTransform, Compose
from delira.training.backends import convert_torch_to_numpy
from functools import partial
from delira.data_loading import DataManager, SequentialSampler

if __name__ == '__main__':

    # PARAMETERS TO CHANGE
    # TODO: Change paths and create suitable Dataset
    checkpoint_path = ""
    data_path = ""
    save_path = ""
    dset = None

    checkpoint_path = os.path.expanduser(checkpoint_path)
    data_path = os.path.expanduser(data_path)
    save_path = os.path.expanduser(save_path)

    transforms = Compose(
        [
            RangeTransform((-1, 1))
        ]
    )

    model = UNetTorch(3, 1, norm_layer="Instance", per_class=False)
    model.load_state_dict(torch.load(checkpoint_path)["model"])
    model.eval()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)

    dmgr = DataManager(dset, 1, 4, transforms,
                       sampler_cls=SequentialSampler)

    predictor = Predictor(model, key_mapping={"x": "data"},
                          convert_batch_to_npy_fn=convert_torch_to_numpy,
                          prepare_batch_fn=partial(model.prepare_batch,
                                                   input_device=device,
                                                   output_device=device)
                          )

    os.makedirs(os.path.join(save_path, "per_class_all_masks"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "per_mask_all_classes"), exist_ok=True)

    with torch.no_grad():
        for idx, (preds_batch, metrics_batch) in enumerate(
                predictor.predict_data_mgr(dmgr, verbose=True)):
            # TODO: Add whatever you want to do here with each batch
            pass
