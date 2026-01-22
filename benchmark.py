#!/usr/bin/env python3
"""
WanAnimatePreprocess 性能基准测试脚本
用于测量优化前后的性能差异
"""

import time
import numpy as np
import os
import sys
import argparse

def benchmark_bbox_detection(detector, images, num_runs=1):
    """
    基准测试 BBOX 检测性能
    
    Args:
        detector: YOLO 检测器
        images: 输入图像 (N, H, W, 3)
        num_runs: 运行次数 (取平均值)
    
    Returns:
        dict: 包含耗时、吞吐量等性能指标
    """
    import cv2
    
    H, W = images.shape[1], images.shape[2]
    shape = np.array([H, W])[None]
    
    # 预热
    print("  预热 GPU...")
    test_img = cv2.resize(images[0], (640, 640)).transpose(2, 0, 1)[None]
    detector(test_img, shape)
    
    # 测试
    print(f"  运行 {num_runs} 次...")
    times = []
    
    for run in range(num_runs):
        start = time.time()
        
        # 逐帧检测 (原始方式)
        for img in images:
            detector(
                cv2.resize(img, (640, 640)).transpose(2, 0, 1)[None],
                shape
            )
        
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"    运行 {run+1}: {elapsed:.2f}s")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    throughput = len(images) / avg_time  # 帧/秒
    
    return {
        'total_time': avg_time,
        'std_dev': std_time,
        'throughput': throughput,
        'fps': throughput,
        'per_frame_ms': (avg_time / len(images)) * 1000
    }

def benchmark_keypoint_extraction(pose_model, images, bboxes, num_runs=1):
    """
    基准测试关键点提取性能
    
    Args:
        pose_model: ViTPose 模型
        images: 输入图像 (N, H, W, 3)
        bboxes: 检测框列表
        num_runs: 运行次数
    
    Returns:
        dict: 性能指标
    """
    from ..pose_utils.pose2d_utils import crop, bbox_from_detector
    import cv2
    
    IMG_NORM_MEAN = np.array([0.485, 0.456, 0.406])
    IMG_NORM_STD = np.array([0.229, 0.224, 0.225])
    input_resolution = (256, 192)
    rescale = 1.25
    
    # 预热
    print("  预热 GPU...")
    bbox = bboxes[0]
    if bbox is None:
        bbox = np.array([0, 0, images.shape[2], images.shape[1]])
    center, scale = bbox_from_detector(bbox, input_resolution, rescale=rescale)
    cropped = crop(images[0], center, scale, (input_resolution[0], input_resolution[1]))[0]
    img_norm = (cropped - IMG_NORM_MEAN) / IMG_NORM_STD
    img_norm = img_norm.transpose(2, 0, 1).astype(np.float32)
    pose_model(img_norm[None], np.array(center)[None], np.array(scale)[None])
    
    # 测试
    print(f"  运行 {num_runs} 次...")
    times = []
    
    for run in range(num_runs):
        start = time.time()
        
        # 逐帧提取 (原始方式)
        for img, bbox in zip(images, bboxes):
            if bbox is None:
                bbox = np.array([0, 0, img.shape[1], img.shape[0]])
            
            center, scale = bbox_from_detector(bbox, input_resolution, rescale=rescale)
            cropped_img = crop(img, center, scale, (input_resolution[0], input_resolution[1]))[0]
            
            img_norm = (cropped_img - IMG_NORM_MEAN) / IMG_NORM_STD
            img_norm = img_norm.transpose(2, 0, 1).astype(np.float32)
            
            pose_model(img_norm[None], np.array(center)[None], np.array(scale)[None])
        
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"    运行 {run+1}: {elapsed:.2f}s")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    throughput = len(images) / avg_time
    
    return {
        'total_time': avg_time,
        'std_dev': std_time,
        'throughput': throughput,
        'fps': throughput,
        'per_frame_ms': (avg_time / len(images)) * 1000
    }

def run_benchmark(video_path=None, num_frames=50, frame_size=(832, 480)):
    """
    运行完整基准测试
    
    Args:
        video_path: 视频文件路径 (如果为 None，使用随机数据)
        num_frames: 要测试的帧数
        frame_size: 帧分辨率 (宽, 高)
    """
    print("=" * 60)
    print("WanAnimatePreprocess 性能基准测试")
    print("=" * 60)
    
    # 加载模型
    print("\n[1] 加载模型...")
    try:
        import torch
        from models.onnx_models import ViTPose, Yolo
        from pose_utils.pose2d_utils import load_pose_metas_from_kp2ds_seq
        
        # 注意: 这里需要实际的模型路径
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'detection')
        
        if not os.path.exists(model_dir):
            print("  警告: 模型目录不存在，跳过基准测试")
            return
        
        # 寻找模型文件
        model_files = os.listdir(model_dir)
        yolo_model = next((f for f in model_files if 'yolo' in f.lower()), None)
        vitpose_model = next((f for f in model_files if 'vitpose' in f.lower()), None)
        
        if not yolo_model or not vitpose_model:
            print("  警告: 找不到必要的模型文件，跳过基准测试")
            print(f"  在 {model_dir} 中找到的文件: {model_files}")
            return
        
        detector = Yolo(os.path.join(model_dir, yolo_model), device='CUDAExecutionProvider')
        pose_model = ViTPose(os.path.join(model_dir, vitpose_model), device='CUDAExecutionProvider')
        print(f"  ✓ 模型已加载")
        
    except Exception as e:
        print(f"  ✗ 模型加载失败: {e}")
        return
    
    # 准备测试数据
    print(f"\n[2] 准备测试数据 ({num_frames} 帧, {frame_size[0]}x{frame_size[1]})...")
    
    if video_path and os.path.exists(video_path):
        # 从视频读取
        import cv2
        cap = cv2.VideoCapture(video_path)
        images = []
        while len(images) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            # 调整为指定大小
            frame = cv2.resize(frame, frame_size)
            images.append(frame)
        cap.release()
        if not images:
            print(f"  ✗ 无法从视频读取帧")
            return
        images = np.array(images)
    else:
        # 使用随机数据
        images = np.random.randint(0, 255, (num_frames, frame_size[1], frame_size[0], 3), dtype=np.uint8)
        print(f"  生成 {num_frames} 帧随机测试图像")
    
    print(f"  ✓ 测试数据已准备 (形状: {images.shape})")
    
    # 运行基准测试
    print(f"\n[3] 运行基准测试 (BBOX 检测)...")
    try:
        detector.reinit()
        bbox_results = benchmark_bbox_detection(detector, images, num_runs=1)
        detector.cleanup()
        
        print(f"  总耗时: {bbox_results['total_time']:.2f}s")
        print(f"  吞吐量: {bbox_results['throughput']:.1f} fps")
        print(f"  单帧耗时: {bbox_results['per_frame_ms']:.1f} ms")
    except Exception as e:
        print(f"  ✗ 基准测试失败: {e}")
        return
    
    # 模拟检测框用于关键点测试
    import cv2
    print(f"\n[4] 运行基准测试 (关键点提取)...")
    
    bboxes = []
    shape = np.array([frame_size[1], frame_size[0]])[None]
    
    try:
        detector.reinit()
        for img in images:
            bbox_result = detector(
                cv2.resize(img, (640, 640)).transpose(2, 0, 1)[None],
                shape
            )
            bboxes.append(bbox_result[0][0]["bbox"])
        detector.cleanup()
        
        pose_model.reinit()
        kp_results = benchmark_keypoint_extraction(pose_model, images, bboxes, num_runs=1)
        pose_model.cleanup()
        
        print(f"  总耗时: {kp_results['total_time']:.2f}s")
        print(f"  吞吐量: {kp_results['throughput']:.1f} fps")
        print(f"  单帧耗时: {kp_results['per_frame_ms']:.1f} ms")
    except Exception as e:
        print(f"  ✗ 关键点提取基准测试失败: {e}")
    
    # 输出总结
    print("\n" + "=" * 60)
    print("基准测试总结")
    print("=" * 60)
    
    try:
        total_time = bbox_results['total_time'] + kp_results['total_time']
        total_frames = len(images)
        
        print(f"总处理时间: {total_time:.2f}s")
        print(f"总帧数: {total_frames}")
        print(f"平均 FPS: {total_frames / total_time:.1f}")
        print(f"平均单帧耗时: {(total_time / total_frames) * 1000:.1f} ms")
        
        print("\n性能预期:")
        print(f"  30 帧视频: ~{(total_time / total_frames) * 30:.1f}s")
        print(f"  60 帧视频: ~{(total_time / total_frames) * 60:.1f}s")
        print(f"  120 帧视频: ~{(total_time / total_frames) * 120:.1f}s")
        
    except:
        pass
    
    print("\n提示: 基准测试结果因系统配置而异")
    print("      多次运行以获得更准确的平均值\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WanAnimatePreprocess 性能基准测试')
    parser.add_argument('--video', type=str, default=None, help='视频文件路径')
    parser.add_argument('--frames', type=int, default=50, help='测试帧数')
    parser.add_argument('--width', type=int, default=832, help='帧宽度')
    parser.add_argument('--height', type=int, default=480, help='帧高度')
    
    args = parser.parse_args()
    
    run_benchmark(
        video_path=args.video,
        num_frames=args.frames,
        frame_size=(args.width, args.height)
    )
