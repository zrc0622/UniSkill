import os
import json

from .base_dataset import BaseDataset

class BridgeDatasetVIDEO(BaseDataset):
    def __init__(
        self,
        data_path: str='workspace/datasets/BridgeDataV2',
        **kwargs,
    ):
        kwargs['min_predict_future_horizon'] = 5
        kwargs['max_predict_future_horizon'] = 10
        super().__init__(data_path, **kwargs)

    def _prepare_data(self, data_path: str):
        split_name = 'train' if self.train else 'test'
        metadata_filename = f"metadata_{split_name}.json"
        metadata_path = os.path.join(data_path, metadata_filename)

        with open(metadata_path, 'r') as f:
            all_videos_metadata = json.load(f)

        self.image_pair = [
            video for video in all_videos_metadata
            if video['length'] >= self.min_predict_future_horizon
        ]