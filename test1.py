import cv2
from ultralytics import YOLO

def detect_and_track(video_path, model_path, output_path="output.mp4"):
    """
    使用 YOLO 进行目标检测与跟踪
    """
    # 1. 加载模型
    try:
        model = YOLO(model_path)
        print(f"成功加载模型: {model_path}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    # 2. 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    # 获取视频属性以保存结果（可选）
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 初始化视频写入器
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    print("开始处理视频... 按 'q' 键退出预览")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 3. 运行 YOLO 跟踪
        # persist=True 是跟踪的关键，它告诉模型在帧之间保持 ID
        results = model.track(frame, persist=True, verbose=False)

        # 4. 可视化结果
        # plot() 方法会在帧上绘制边界框、类别和跟踪 ID
        annotated_frame = results[0].plot()

        # 显示
        cv2.imshow("YOLO Tracking", annotated_frame)
        
        # 写入输出文件
        out.write(annotated_frame)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"处理完成，结果已保存至: {output_path}")

if __name__ == "__main__":
    # === 配置区域 ===
    # 替换为你的视频路径
    my_video = r"D:\resouce_bs\data\2ca6e53f120d48d8e5ca87df746bd819.mp4" 
    
    # 替换为你的模型路径 (例如 "yolov8n.pt", "yolov8s.pt" 或你训练好的 "best.pt")
    # 如果本地没有，它会自动下载 yolov8n.pt
    my_model = r"D:\resouce_bs\best2.pt" 
    
    
    detect_and_track(my_video, my_model)# filepath: d:\bishe\LangGraph\video_tampering_agent\test_yolo_track.py
