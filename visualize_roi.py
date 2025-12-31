import cv2
import matplotlib.pyplot as plt
import sys
import os

def show_frame_with_grid(video_path):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Read the first frame
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not read frame.")
        return

    # Convert BGR to RGB for matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    h, w, _ = frame_rgb.shape
    print(f"Video Resolution: Width={w}, Height={h}")

    plt.figure(figsize=(12, 8))
    plt.imshow(frame_rgb)
    plt.title(f"First Frame - Resolution: {w}x{h}")
    
    # Show coordinates on hover (standard matplotlib feature)
    plt.show()

if __name__ == "__main__":
    # Default path or from command line
    default_video = r"d:\bishe\LangGraph\video_tampering_agent\.cache\car_detection\demo-forgery\annotated.mp4"
    
    target_video = sys.argv[1] if len(sys.argv) > 1 else default_video
    
    print(f"Opening: {target_video}")
    show_frame_with_grid(target_video)
