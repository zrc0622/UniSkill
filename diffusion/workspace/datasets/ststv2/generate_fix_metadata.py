import os
import json
import decord
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

DATASET_ROOT = '/mnt/mnt/public/fangzhirui/zrc/UniSkill/diffusion/workspace/datasets/ststv2'
VIDEO_DIRECTORY = os.path.join(DATASET_ROOT, '20bn-something-something-v2-fix')

def process_video_item(item, video_dir):
    video_id = item.get('id')
    template = item.get('template')

    if not video_id or not template:
        return None, None  # 返回 None 表示无效条目

    video_path = os.path.join(video_dir, f"{video_id}.webm")

    if not os.path.exists(video_path):
        return None, None

    try:
        vr = decord.VideoReader(video_path)
        video_len = len(vr)

        if video_len < 2:
            return None, None

        # 校验首尾帧，确保视频可读
        _ = vr[0].asnumpy()
        _ = vr[video_len - 1].asnumpy()

        metadata = {
            'template': template,
            'length': video_len
        }
        return video_id, metadata

    except (decord.DECORDError, Exception):
        # 任何读取错误都视为文件损坏，返回损坏文件路径用于记录
        return "bad_file", video_path


def create_new_metadata_file_parallel(original_path, new_path, video_dir):
    if not os.path.exists(original_path):
        print(f"输入文件不存在，已跳过: {original_path}")
        return

    with open(original_path, 'r') as f:
        old_metadata_list = json.load(f)

    new_metadata_dict = {}
    bad_files = []

    num_processes = cpu_count()
    print(f"使用 {num_processes} 个进程并行处理...")

    with Pool(processes=num_processes) as pool:

        process_func = partial(process_video_item, video_dir=video_dir)

        desc = f"校验并处理 {os.path.basename(original_path)}"
        results = list(tqdm(pool.imap_unordered(process_func, old_metadata_list), total=len(old_metadata_list), desc=desc))

    for video_id, result_data in results:
        if video_id == "bad_file":
            bad_files.append(result_data)
        elif video_id is not None:
            new_metadata_dict[video_id] = result_data

    with open(new_path, 'w') as f:
        json.dump(new_metadata_dict, f, indent=4)

    print(f"处理完成, 新文件已保存至: {new_path}")
    if bad_files:
        print("\n检测到并排除了以下损坏的视频文件:")
        for f in bad_files:
            print(f" - {f}")


if __name__ == '__main__':
    labels_dir = os.path.join(DATASET_ROOT, 'labels')

    file_mapping = {
        os.path.join(labels_dir, 'train.json'): os.path.join(labels_dir, 'metadata_train.json'),
        os.path.join(labels_dir, 'validation.json'): os.path.join(labels_dir, 'metadata_val.json')
    }

    for input_file, output_file in file_mapping.items():
        create_new_metadata_file_parallel(input_file, output_file, VIDEO_DIRECTORY)

    print("\n所有元数据文件处理完毕。")