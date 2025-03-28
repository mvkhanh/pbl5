import os
import subprocess
import multiprocessing
import cv2

# Số frame mỗi segment
SEGMENT_SIZE = 32

def split_video(args):
    """ Tách video thành từng segment 32 frame """
    input_path, output_folder, is_normal = args
    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)

    # Lấy thông tin video
    total_frames, fps = int(cv2.VideoCapture(input_path).get(cv2.CAP_PROP_FRAME_COUNT)), 5
    if total_frames is None or fps is None:
        print(f"❌ Không thể đọc {filename}, bỏ qua!")
        return

    num_segments = total_frames // SEGMENT_SIZE
    if num_segments < 1:
        print(f"⚠️ {filename} quá ngắn ({total_frames} frames), bỏ qua!")
        return

    # Nếu không phải "NormalVideos", cắt bớt 20% đầu & cuối
    start_idx, end_idx = 0, num_segments
    # if not is_normal and num_segments > 4:
    #     trim_size = num_segments // 5  # 20% số segment
    #     start_idx += trim_size
    #     end_idx -= trim_size

    print(f"🎬 {filename}: {total_frames} frames, {end_idx - start_idx} segments")

    for i in range(start_idx, end_idx):
        new_index = i - start_idx  # Reset index về 0
        segment_filename = f"{name}_{new_index:04d}{ext}"
        segment_output = os.path.join(output_folder, segment_filename)

        if os.path.exists(segment_output):
            print(f"✔️ Đã có {segment_filename}, bỏ qua!")
            continue

        start_frame = i * SEGMENT_SIZE  # Giữ nguyên start_frame theo index gốc

        command = [
            "ffmpeg",
            "-hwaccel", "auto",
            "-i", input_path,
            "-ss", str(start_frame / fps),
            "-t", str(SEGMENT_SIZE / fps),
            "-c:v", "libx264", "-crf", "23", "-preset", "fast",
            segment_output
        ]

        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✅ Đã tạo segment: {segment_output}")

def process_folder(input_folder, output_folder, num_workers=4):
    """ Duyệt qua thư mục và tách video song song """
    task_list = []

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.mp4', '.mkv', '.avi', '.mov')):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_folder)
                is_normal = "Normal" in relative_path

                # Tạo thư mục đích
                segment_output_folder = os.path.join(output_folder, relative_path)
                os.makedirs(segment_output_folder, exist_ok=True)

                task_list.append((input_path, segment_output_folder, is_normal))

    # Giới hạn số luồng tối đa
    num_workers = min(8, os.cpu_count() // 2)
    print(f"🚀 Chạy {num_workers} luồng song song...")

    with multiprocessing.Pool(num_workers) as pool:
        pool.map(split_video, task_list)

if __name__ == "__main__":
    input_folder = "5fps/Train"
    output_folder = "5fps_Segments/Train"
    os.makedirs(output_folder, exist_ok=True)

    process_folder(input_folder, output_folder)