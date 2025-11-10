import os
import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
import numpy as np
from tqdm import tqdm

def convert_dataset_to_videos(raw_data_path, output_path, fps=30):
    print(f"从 '{raw_data_path}' 加载 TFDS builder...")
    builder = tfds.builder_from_directory(builder_dir=raw_data_path)

    for split in ['train', 'test']:
        print(f"\n--- 正在处理 {split} 分区 ---")
        dataset = builder.as_dataset(split=split)
        
        split_output_path = os.path.join(output_path, split)
        os.makedirs(split_output_path, exist_ok=True)
        
        for i, episode in tqdm(enumerate(dataset), desc=f"转换 {split} episodes"):
            frames = []
            for step in episode['steps']:
                image_tensor = step['observation']['image']
                image_np_rgb = image_tensor.numpy()
                image_np_bgr = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2BGR)
                frames.append(image_np_bgr)

            if not frames:
                continue

            height, width, _ = frames[0].shape
            video_filename = f"episode_{i:06d}.mp4"
            video_filepath = os.path.join(split_output_path, video_filename)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(video_filepath, fourcc, fps, (width, height))

            for frame in frames:
                writer.write(frame)

            writer.release()

    print("\n所有数据转换完成！")


if __name__ == '__main__':
    convert_dataset_to_videos('/mnt/mnt/public_zgc/BridgeDataV2/OpenDataLab___BridgeData_V2/raw/bridge/0.1.0/', '/mnt/mnt/public_zgc/BridgeDataV2/OpenDataLab___BridgeData_V2/raw/bridge/video/', 15)