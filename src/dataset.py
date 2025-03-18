"""Module to load the breast cancer dataset."""

from pathlib import Path
from typing import Dict
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from pandas import DataFrame


class CancerDataset(Dataset):
    """
    Class setting up a breast cancer Dataset.
    """

    def __init__(
        self,
        data_loc: Path,
        train: bool = False,
    ):
        self.dataset: pd.DataFrame = pd.read_csv(data_loc.joinpath("breast-cancer.csv"))

        self.split = _split_data(self.dataset)["train" if train is True else "valid"]

    def __len__(self):
        return len(self.split)

    def __getitem__(self) -> None:
        """Return an instance.

        """
        return None

def _split_data(dataset: DataFrame) -> Dict[str, DataFrame]:
    """Split the ``types_df`` into a training and validation set.

    Parameters
    ----------
    types_dataset : DataFrame
        The full types data set.

    Returns
    -------
    Dict[str, DataFrame]
        Dictionary holding the ``"train"`` and ``"valid"`` splits.

    """
    valid_df = dataset.sample(
        n=int(np.floor(0.2 * dataset.shape[0])),
        random_state=321,
    )

    # The training items are simply the items *not* in the valid split
    train_df = dataset.loc[~dataset.index.isin(valid_df.index)]

    return {"train": train_df, "valid": valid_df}
