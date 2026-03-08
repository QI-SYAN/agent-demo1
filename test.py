import cv2
import numpy as np
import yaml
import os
from ultralytics import YOLO

def _clamp_axis(value, upper_bound):
    if upper_bound <= 0:
        return 0
    return max(0, min(upper_bound - 1, int(value)))

def detect_and_track_stage2(video_path, model_path, output_path="output_stage2.mp4"):
    # === 1. 创建更强的 Tracker 配置 ===
    # === 1. 创建更强的 Tracker 配置 ===
    tracker_config = {
        'tracker_type': 'bytetrack',
        'track_high_thresh': 0.25, # 高分阈值
        'track_low_thresh': 0.1,   # 低分阈值
        'new_track_thresh': 0.25,  # 新轨迹阈值
        'track_buffer': 120,       # 缓存时间(帧) - 增加到 120 帧 (约4秒) 以解决 2s 删除测试
        'match_thresh': 0.8,       # 匹配阈值
        'gating_thresh': 0.8,      # 门控阈值
        'fuse_score': True,         # <--- 必须添加这个！
    }
    
    # 使用绝对路径，确保 YOLO 能找到
    tracker_yaml = os.path.abspath("custom_tracker.yaml")
    with open(tracker_yaml, 'w') as f:
        yaml.dump(tracker_config, f)

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30
    
    # 保持和 Agent 一致的 Resize
    TARGET_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    TARGET_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (TARGET_W, TARGET_H))

    # ROI 定义 (保持不变)
    w = TARGET_W
    h = TARGET_H
    # roi_contour = np.array([
    #     [0, _clamp_axis(1000, h)],
    #     [0, _clamp_axis(675, h)],
    #     [_clamp_axis(270, w), _clamp_axis(435, h)],
    #     [_clamp_axis(w - 1, w), _clamp_axis(435, h)],
    #     [_clamp_axis(w - 1, w), _clamp_axis(1000, h)],
    # ], dtype=np.int32)
    roi_contour = np.array([
        [0, _clamp_axis(450, h)],
        [0, _clamp_axis(300, h)],
        [_clamp_axis(110, w), _clamp_axis(220, h)],
        [_clamp_axis(585, w), _clamp_axis(220, h)],
        [_clamp_axis(585, w), _clamp_axis(450, h)],
    ], dtype=np.int32)
    print(f"开始 Stage 2 测试... 使用 Tracker: {tracker_yaml}")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if roi_contour.size:
            cv2.polylines(frame, [roi_contour], True, (0, 255, 255), 2)

        # === 2. 加载自定义 Tracker ===
        # conf=0.1: 允许低分框出现，防止断连
        # persist=True: 必须开启
        try:
            results = model.track(frame, persist=True, verbose=False, 
                                tracker=tracker_yaml, 
                                conf=0.1) 
        except Exception as e:
            print(f"Tracker error: {e}")
            break

        if results and results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes
            for idx, box in enumerate(boxes.xyxy):
                x1, y1, x2, y2 = map(int, box)
                track_id = int(boxes.id[idx])
                
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                is_in_roi = cv2.pointPolygonTest(roi_contour, (center_x, center_y), False) >= 0

                color = (0, 255, 0) if is_in_roi else (0, 0, 255)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # 显示 ID，看看是不是还会在绿线附近变号
                cv2.putText(frame, f"ID:{track_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Stage 2 Check", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    my_video = r"D:\resouce_bs\1_part002.mp4"
    my_model = r"D:\resouce_bs\best2.pt" 
    detect_and_track_stage2(my_video, my_model)