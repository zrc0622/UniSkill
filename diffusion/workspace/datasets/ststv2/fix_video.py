import os
import subprocess
import glob
from tqdm import tqdm
import multiprocessing

def reencode_worker(args):
    input_path, output_path = args
    
    # 构建 ffmpeg 命令
    command = [
        'ffmpeg',
        '-i', input_path,
        '-y',  # 覆盖输出文件
        '-loglevel', 'error', # 只在发生错误时打印日志
        output_path
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return None
    except subprocess.CalledProcessError as e:
        filename = os.path.basename(input_path)
        error_message = e.stderr.strip()
        return (filename, error_message)

def batch_reencode_videos_parallel(source_dir, dest_dir):
    if not os.path.isdir(source_dir):
        print(f"错误: 源目录 '{source_dir}' 不存在。")
        return

    os.makedirs(dest_dir, exist_ok=True)

    video_files = glob.glob(os.path.join(source_dir, '*.webm'))
    if not video_files:
        print(f"警告: 在目录 '{source_dir}' 中没有找到任何 .webm 文件。")
        return
        
    tasks = [(input_path, os.path.join(dest_dir, os.path.basename(input_path))) for input_path in video_files]

    print(f"找到了 {len(tasks)} 个 .webm 文件，准备开始并行处理...")

    num_processes = max(1, multiprocessing.cpu_count() - 1)
    print(f"将使用 {num_processes} 个 CPU 核心进行加速...")

    failed_files = []

    with multiprocessing.Pool(processes=num_processes) as pool:
        with tqdm(total=len(tasks), desc="视频并行修复进度") as pbar:
            for result in pool.imap_unordered(reencode_worker, tasks):
                if result is not None:
                    failed_files.append(result)
                pbar.update(1)

    print("\n--- ✨ 全部处理完成 ---")
    success_count = len(tasks) - len(failed_files)
    print(f"总计文件数: {len(tasks)}")
    print(f"成功处理: {success_count}")
    print(f"失败处理: {len(failed_files)}")

    if failed_files:
        print("\n以下文件在处理过程中失败了:")
        for filename, error_msg in failed_files:
            print(f" - 文件: {filename}")
            print(f"   错误: {error_msg}")


if __name__ == "__main__":
    SOURCE_DIRECTORY = '20bn-something-something-v2'
    DESTINATION_DIRECTORY = '20bn-something-something-v2-fix'

    batch_reencode_videos_parallel(SOURCE_DIRECTORY, DESTINATION_DIRECTORY)