import argparse
import os
import os.path
import numpy as np
import torch
import open3d as o3d
from multiprocessing import Pool, set_start_method
import xlrd
import time
import csv

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pad_point_cloud_torch(tensor, target_size):
    current_size = tensor.shape[0]
    if current_size >= target_size:
        return tensor

    repeat_factor = target_size // current_size
    remaining_size = target_size % current_size

    padded_tensor = tensor.tile((repeat_factor, 1))

    if remaining_size > 0:
        remaining_part = tensor[:remaining_size, :]
        padded_tensor = torch.cat((padded_tensor, remaining_part), dim=0)

    return padded_tensor


def fps_torch(pts, k):
    with torch.no_grad():
        n_points = pts.shape[0]
        device = pts.device
        farthest_pts_indices = torch.zeros(k, dtype=torch.long, device=device)
        farthest_pts_indices[0] = torch.randint(n_points, (1,), device=device).item()
        distances = ((pts[farthest_pts_indices[0]] - pts) ** 2).sum(dim=-1)

        for i in range(1, k):
            farthest_pts_indices[i] = torch.argmax(distances)
            distances = torch.min(distances, ((pts[farthest_pts_indices[i]] - pts) ** 2).sum(dim=-1))
        return farthest_pts_indices


def calculate_entropy_torch_batch(patches_batch):
    with torch.no_grad():
        B, N, _ = patches_batch.shape
        device = patches_batch.device
        if N == 0:
            return torch.zeros(B, device=device, dtype=torch.float32)

        rgb_values = patches_batch[..., 3:6].long()
        combined_rgb = (rgb_values[..., 0] << 16) + (rgb_values[..., 1] << 8) + (rgb_values[..., 2])
        entropies = torch.empty(B, device=device, dtype=torch.float32)
        for i in range(B):
            counts = torch.bincount(combined_rgb[i])
            counts = counts[counts > 0]
            probabilities = counts.float() / N
            entropies[i] = -torch.sum(probabilities * torch.log2(probabilities))
        return entropies


def fps_torch_batch(pts_batch, k):
    with torch.no_grad():
        B, N, D = pts_batch.shape
        device = pts_batch.device
        farthest_indices = torch.zeros((B, k), dtype=torch.long, device=device)
        start_indices = torch.randint(0, N, (B,), dtype=torch.long, device=device)
        farthest_indices[:, 0] = start_indices

        min_distances = torch.full((B, N), float('inf'), device=device, dtype=torch.float32)

        first_points = torch.gather(pts_batch, 1, start_indices.view(B, 1, 1).expand(-1, -1, D))
        dists = ((pts_batch - first_points) ** 2).sum(dim=-1)
        min_distances = torch.min(min_distances, dists)

        for i in range(1, k):
            next_indices = torch.argmax(min_distances, dim=1)
            farthest_indices[:, i] = next_indices
            next_points = torch.gather(pts_batch, 1, next_indices.view(B, 1, 1).expand(-1, -1, D))
            dists = ((pts_batch - next_points) ** 2).sum(dim=-1)
            min_distances = torch.min(min_distances, dists)

        return farthest_indices


def knn_gpu_bruteforce(points, queries, k, batch_size=1024):
    with torch.no_grad():
        all_indices = []
        num_queries = queries.shape[0]

        for start_idx in range(0, num_queries, batch_size):
            end_idx = min(start_idx + batch_size, num_queries)
            query_batch = queries[start_idx:end_idx]
            dists = torch.cdist(query_batch, points)
            knn_indices = torch.topk(dists, k, dim=-1, largest=False, sorted=True)[1]
            all_indices.append(knn_indices)

        return torch.cat(all_indices, dim=0)


def read_xlrd(excelFile):
    try:
        data = xlrd.open_workbook(excelFile)
        table = data.sheet_by_index(0)
        dataFile = []
        for rowNum in range(table.nrows):
            if rowNum > 0:
                dataFile.append(table.row_values(rowNum))
        return dataFile
    except FileNotFoundError:
        print(f"Error: Excel file not found at {excelFile}")
        return []


def create_patch_fast(id, path, args, device_str):
    device = torch.device(device_str)
    total_start_time = time.time()
    ply_str = os.path.splitext(os.path.basename(path))[0]
    folder_big = os.path.join(args.data_dir, args.patch_dir_big, ply_str)
    folder_small = os.path.join(args.data_dir, args.patch_dir_small, ply_str)

    if os.path.exists(folder_big):
        return {'filename': path, 'total_time': 0, 'entropy_time': 0, 'status': 'skipped'}

    os.makedirs(folder_big, exist_ok=True)
    os.makedirs(folder_small, exist_ok=True)

    PC_dir = os.path.join(args.data_dir, args.ply_dir, path)
    pcd = o3d.io.read_point_cloud(PC_dir)
    xyz_np = np.asarray(pcd.points)

    if len(xyz_np) == 0:
        print(f"Warning: File {path} is empty, skipping.")
        return {'filename': path, 'total_time': 0, 'entropy_time': 0, 'status': 'empty'}

    rgb_np = np.asarray(pcd.colors) * 255
    xyz_normalized_np = xyz_1_2001(xyz_np)

    point_cloud_torch = torch.from_numpy(
        np.concatenate((xyz_normalized_np, rgb_np), axis=1)
    ).to(device, dtype=torch.float32)

    if point_cloud_torch.shape[0] < args.knn_big:
        print(f"Info: File {path} has too few points ({point_cloud_torch.shape[0]}), padding to {args.knn_big}.")
        point_cloud_torch = pad_point_cloud_torch(point_cloud_torch, args.knn_big)

    point_cloud_xyz_torch = point_cloud_torch[:, :3]

    total_points = point_cloud_xyz_torch.shape[0]
    _center_num = max(total_points // 8192, 64)
    center_num_cap = 256
    center_num = min(_center_num, center_num_cap)
    center_num = min(center_num, total_points)

    center_indices_fps = fps_torch(point_cloud_xyz_torch, center_num)

    entropy_start_time = time.time()
    centers_for_entropy_torch = point_cloud_xyz_torch[center_indices_fps]
    entropy_patch_indices_torch = knn_gpu_bruteforce(
        point_cloud_xyz_torch, centers_for_entropy_torch, args.knn_small
    )
    entropy_patches_batch = point_cloud_torch[entropy_patch_indices_torch]
    entropies_torch = calculate_entropy_torch_batch(entropy_patches_batch)
    entropy_end_time = time.time()
    entropy_processing_time = entropy_end_time - entropy_start_time

    exps = torch.exp(entropies_torch)
    probs = exps / torch.sum(exps) if torch.sum(exps) > 0 else torch.full_like(exps, 1.0 / len(exps))

    num_to_sample = min(args.center_points, len(center_indices_fps))
    if num_to_sample < args.center_points:
        print(f"Warning: File {path} has insufficient initial centers, sampling {num_to_sample}.")

    sampled_indices_in_list = torch.multinomial(probs, num_to_sample, replacement=False)
    final_center_indices = center_indices_fps[sampled_indices_in_list]
    final_center_coords_torch = point_cloud_xyz_torch[final_center_indices]

    big_patch_indices_torch = knn_gpu_bruteforce(
        point_cloud_xyz_torch, final_center_coords_torch, args.knn_big
    )
    small_patch_indices_torch = knn_gpu_bruteforce(
        point_cloud_xyz_torch, final_center_coords_torch, args.knn_small
    )

    big_patches_batch = point_cloud_torch[big_patch_indices_torch]
    small_patches_batch = point_cloud_torch[small_patch_indices_torch]

    patch_fps_indices_batch = fps_torch_batch(big_patches_batch[:, :, :3], 1024)

    patch_fps_1024_batch = torch.gather(big_patches_batch, 1, patch_fps_indices_batch.unsqueeze(-1).expand(-1, -1, 6))

    final_big_patches_np = patch_fps_1024_batch.cpu().numpy()
    final_small_patches_np = small_patches_batch.cpu().numpy()

    for m in range(num_to_sample):
        filename = f"{ply_str}__{m}"
        np.save(os.path.join(folder_big, filename), final_big_patches_np[m])
        np.save(os.path.join(folder_small, filename), final_small_patches_np[m])

    total_end_time = time.time()
    total_processing_time = total_end_time - total_start_time

    print(
        f'File {path} processed. Total time: {total_processing_time:.3f}s, Entropy time: {entropy_processing_time:.3f}s')

    del point_cloud_torch, big_patches_batch, small_patches_batch, patch_fps_1024_batch
    torch.cuda.empty_cache()

    return {'filename': path, 'total_time': total_processing_time, 'entropy_time': entropy_processing_time,
            'status': 'processed'}


if __name__ == '__main__':
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description='Expriment setting')
    parser.add_argument('--knn_big', type=int, required=True)
    parser.add_argument('--knn_small', type=int, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--ply_dir', type=str, required=True)
    parser.add_argument('--patch_dir_big', type=str, required=True)
    parser.add_argument('--patch_dir_small', type=str, required=True)
    parser.add_argument('--center_points', type=int, required=True)
    parser.add_argument('--parallel_cpu', type=int, default=1)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.data_dir, args.patch_dir_big), exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, args.patch_dir_small), exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")
    if str(device) == 'cpu':
        print("Warning: CUDA device not detected. Running on CPU, which will be very slow.")

    print('-------Starting patch creation--------')
    excel_path = os.path.join(args.data_dir, 'mos.xls')
    exle_file = read_xlrd(excel_path)

    if not exle_file:
        print("Excel file is empty or unreadable. Exiting.")
        exit()

    print(f'Save dir: {args.data_dir}/{args.patch_dir_big} and {args.data_dir}/{args.patch_dir_small}')
    print(f'There are {len(exle_file)} files waiting for process... ')
    print(f'USE {args.parallel_cpu} CPU core(s) to process the task in parallel.')

    all_results = []
    device_str = str(device)
    with Pool(args.parallel_cpu) as pool:
        results_async = [pool.apply_async(func=create_patch_fast, args=(id, name, args, device_str))
                         for id, [name, mos] in enumerate(exle_file)]
        all_results = [res.get() for res in results_async]

    print(f'-------All patch splitting completed-------')

    timing_records = [record for record in all_results if record and record.get('status') == 'processed']

    if timing_records:
        log_dir = os.path.join(args.data_dir, 'processing_logs')
        os.makedirs(log_dir, exist_ok=True)
        csv_log_path = os.path.join(log_dir, 'timing_details.csv')
        summary_report_path = os.path.join(log_dir, 'summary_report.txt')

        try:
            with open(csv_log_path, 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['Filename', 'Total Time (s)', 'Entropy Time (s)'])
                for record in timing_records:
                    csv_writer.writerow([
                        record['filename'],
                        f"{record['total_time']:.4f}",
                        f"{record['entropy_time']:.4f}"
                    ])
            print(f"\nDetailed timing log saved to: {csv_log_path}")
        except IOError as e:
            print(f"Error: Could not write CSV file {csv_log_path}. Reason: {e}")

        total_times = [r['total_time'] for r in timing_records]
        entropy_times = [r['entropy_time'] for r in timing_records]
        avg_total_time = np.mean(total_times)
        avg_entropy_time = np.mean(entropy_times)
        max_total_time = np.max(total_times)
        min_total_time = np.min(total_times)
        total_processing_time = np.sum(total_times)

        report_content = (
            f"{'=' * 20} Performance Statistics Report {'=' * 20}\n"
            f"Report generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'-' * 55}\n"
            f"Files processed successfully: {len(timing_records)}\n"
            f"Total processing time: {total_processing_time:.2f} seconds\n\n"
            f"Average total time/file: {avg_total_time:.4f} seconds\n"
            f"Average entropy time/file: {avg_entropy_time:.4f} seconds\n"
            f"Max processing time: {max_total_time:.4f} seconds\n"
            f"Min processing time: {min_total_time:.4f} seconds\n"
            f"{'=' * 55}\n"
        )

        try:
            with open(summary_report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"Performance summary report saved to: {summary_report_path}")
        except IOError as e:
            print(f"Error: Could not write summary file {summary_report_path}. Reason: {e}")

        print("\n--- Performance Summary ---")
        print(report_content)
        print("\nPerformance report generated.")
    else:
        print("\nNo files were processed successfully. Cannot generate performance report.")