import os
import h5py
import numpy as np
from PIL import Image

from .base_dataset import BaseDataset

class LIBERODatasetHDF5(BaseDataset):
    def __init__(
        self,
        data_path: str='workspace/datasets/LIBER0',
        **kwargs,
    ):
        kwargs['min_predict_future_horizon'] = 20
        kwargs['max_predict_future_horizon'] = 40
        super().__init__(data_path, **kwargs)
        
    def _prepare_data(self, data_path):
        videos = []
        view_key = 'agentview_rgb'

        for suite in os.listdir(data_path):
            if not suite.startswith('libero'):
                continue
            suite_path = os.path.join(data_path, suite)
            
            if not os.path.isdir(suite_path):
                continue
                
            hdf5_files_in_suite = [f for f in os.listdir(suite_path) if f.endswith('.hdf5')]
            
            if self.train:
                demos_files = sorted(hdf5_files_in_suite)[:int(len(hdf5_files_in_suite)*0.9)]
            else:
                demos_files = sorted(hdf5_files_in_suite)[int(len(hdf5_files_in_suite)*0.9):]
            
            for demo_file in demos_files:
                file_path = os.path.join(suite_path, demo_file)
                try:
                    with h5py.File(file_path, 'r') as f:
                        for demo_id in f['data'].keys():
                            internal_key = f'data/{demo_id}/obs/{view_key}'
                            
                            if internal_key in f:
                                vid_len = f[internal_key].shape[0]

                                if vid_len < self.min_predict_future_horizon:
                                    continue
                                
                                combined_path = f"{file_path}||{internal_key}"

                                videos.append({
                                    'path': combined_path,
                                    'length': vid_len,
                                })
                except Exception:
                    pass
            
        self.image_pair = videos
        
    def read_images(self, video_path, prev_idx, next_idx):
        file_path, internal_key = video_path.split('||')

        with h5py.File(file_path, 'r') as f:
            video_dataset = f[internal_key]
            conditioning_frame = video_dataset[prev_idx]
            frame = video_dataset[next_idx]

        curr_image = Image.fromarray(conditioning_frame)
        next_image = Image.fromarray(frame)
        
        return curr_image, next_image