import os
import json
import argparse
import cv2
from tqdm import tqdm

def generate_metadata(video_data_path: str, output_path: str):
    os.makedirs(output_path, exist_ok=True)

    for split in ['train', 'test']:
        split_video_path = os.path.join(video_data_path, split)
        
        if not os.path.isdir(split_video_path):
            continue

        split_metadata = []
        video_files = sorted([f for f in os.listdir(split_video_path) if f.endswith('.mp4')])

        for video_filename in tqdm(video_files, desc=f"扫描 {split} 视频"):
            video_filepath = os.path.join(split_video_path, video_filename)
            
            try:
                cap = cv2.VideoCapture(video_filepath)
                if not cap.isOpened():
                    continue
                
                # 获取视频的总帧数
                vid_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                split_metadata.append({
                    'path': video_filepath, # 存储完整路径以方便读取
                    'length': vid_len,
                })
            except Exception as e:
                print(f"处理文件 {video_filepath} 时出错: {e}")

        output_file = os.path.join(output_path, f"metadata_{split}.json")
        
        print(f"元数据生成完毕。共找到 {len(split_metadata)} 个视频。")
        print(f"正在将结果保存到 '{output_file}'...")
        with open(output_file, 'w') as f:
            json.dump(split_metadata, f, indent=4)
        print(f"{split} 分区处理完成！\n")

    print("所有分区处理完毕。")

if __name__ == '__main__':
    generate_metadata('/mnt/mnt/public_zgc/BridgeDataV2/OpenDataLab___BridgeData_V2/raw/bridge/video', '/mnt/mnt/public/fangzhirui/zrc/UniSkill/diffusion/workspace/datasets/BridgeDataV2')