import os
import subprocess
import multiprocessing

# Chỉnh codec phù hợp với GPU của bạn
CODEC = "libx265"  # Thay bằng "h264_nvenc" nếu có NVIDIA, "hevc_amf" nếu dùng AMD

def convert_video(input_path, output_path):
    """ Chuyển đổi video sang 5fps và lưu vào thư mục output. """
    if os.path.exists(output_path):
        print(f"❌ Bỏ qua {input_path}, đã tồn tại!")
        return

    command = [
        "ffmpeg",
        "-hwaccel", "auto",
        "-i", input_path,
        "-vf", "fps=5",
        "-c:v", CODEC,
        "-crf", "20",
        "-preset", "fast",
        output_path
    ]

    print(f"🚀 Đang xử lý: {input_path}")
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"✅ Hoàn tất: {input_path} -> {output_path}")

def process_folder(input_root, output_root, num_workers=4):
    """ Duyệt qua tất cả thư mục trong input_root và xử lý video """
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    tasks = []
    
    for root, _, files in os.walk(input_root):
        relative_path = os.path.relpath(root, input_root)
        output_dir = os.path.join(output_root, relative_path)
        os.makedirs(output_dir, exist_ok=True)

        video_files = [f for f in files if f.lower().endswith(('.mp4', '.mkv', '.avi', '.mov'))]
        for video in video_files:
            input_path = os.path.join(root, video)
            output_path = os.path.join(output_dir, f"{os.path.splitext(video)[0]}_5fps{os.path.splitext(video)[1]}")
            tasks.append((input_path, output_path))
    
    print(f"📂 Tìm thấy {len(tasks)} video cần xử lý.")
    
    with multiprocessing.Pool(num_workers) as pool:
        pool.starmap(convert_video, tasks)

if __name__ == "__main__":
    input_folder = "UCFCrimeDataset/Test"  # Thư mục gốc chứa video
    output_folder = "5fps/Test"  # Thư mục gốc lưu kết quả
    num_workers = 8  # Số luồng xử lý song song
    process_folder(input_folder, output_folder, num_workers)
