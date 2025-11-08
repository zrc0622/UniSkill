from torch.utils.data import Dataset as TorchDataset

from . import *

class MiniCombinedDataset(TorchDataset):
    def __init__(
        self,
        datasets=["libero_hdf5", "sthsthv2_webm"],
        **kwargs,
    ):
        self.datasets = []
        dataset_classes = {
            "libero_hdf5": LIBERODatasetHDF5,
            "sthsthv2_webm": SthSthv2DatasetWEBM,
        }

        for dataset_name in datasets:
            if dataset_name in dataset_classes:
                self.datasets.append(dataset_classes[dataset_name](**kwargs))

        self.dataset_lengths = [len(dataset) for dataset in self.datasets]
            
    def __len__(self):
        return sum(self.dataset_lengths)
    
    def __getitem__(self, idx):
        for i, dataset in enumerate(self.datasets):
            if idx < self.dataset_lengths[i]:
                return dataset[idx]
            idx -= self.dataset_lengths[i]
        raise ValueError(f"Index {idx} out of bounds for CombinedDataset")