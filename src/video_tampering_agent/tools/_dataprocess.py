import copy
import cv2
import numpy as np

class VehicleTracker:
    def __init__(self, xymap, video_path):
        self.xymap = xymap
        self.video_path = video_path
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        # self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        #                                        np.float32) * 0.03
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                               np.float32) * 0.05
        self.cap = cv2.VideoCapture(video_path)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def init_kalman(self, initial_measurement):
        # 添加调试输出
        # print(
        #     f"Debug: Initial measurement received is {initial_measurement} with type {type(initial_measurement)} and shape {initial_measurement.shape if isinstance(initial_measurement, np.ndarray) else 'N/A'}.")

        # 确保 initial_measurement 是一个形状为 (2, 1) 或 (2,) 的 NumPy 数组
        if isinstance(initial_measurement, np.ndarray):
            if initial_measurement.shape == (2, 1):
                initial_measurement = initial_measurement.flatten()  # 将其转换为一维数组
            elif initial_measurement.shape != (2,):
                raise ValueError("initial_measurement must be a NumPy array of shape (2, 1) or (2,).")

        # 确保 initial_measurement 是一个长度为2的列表或元组，并且每个元素都是数值类型
        if not isinstance(initial_measurement, (list, tuple, np.ndarray)) or len(initial_measurement) != 2:
            raise ValueError("initial_measurement must be a list, tuple, or NumPy array of length 2.")

        for item in initial_measurement:
            if not isinstance(item, (int, float, np.number)):
                raise TypeError("Each element in initial_measurement should be an int, float, or numpy number.")

        # 将 initial_measurement 的元素转换为浮点数，并初始化 statePre 和 statePost
        self.kalman.statePre = np.array([
            [float(initial_measurement[0])],  # x 坐标
            [float(initial_measurement[1])],  # y 坐标
            [0.0],  # 初始速度 vx
            [0.0]  # 初始速度 vy
        ], dtype=np.float32)

        self.kalman.statePost = np.array([
            [float(initial_measurement[0])],  # x 坐标
            [float(initial_measurement[1])],  # y 坐标
            [0.0],  # 初始速度 vx
            [0.0]  # 初始速度 vy
        ], dtype=np.float32)

        # # 添加调试输出以确认初始化成功
        # print(f"Initialized statePre with: \n{self.kalman.statePre}")
        # print(f"Initialized statePost with: \n{self.kalman.statePost}")
    def track_vehicle(self):
        result = []
        # Skip frames until the last known position.
        for i in range(len(self.xymap)):
            ret, frame = self.cap.read()
            if not ret:
                break
            center_x = (self.xymap[i][0] + self.xymap[i][2]) // 2
            center_y = (self.xymap[i][1] + self.xymap[i][3]) // 2
            measurement = np.array([[np.float32(center_x)], [np.float32(center_y)]])
            if i == len(self.xymap) - 1:
                self.init_kalman(measurement)

        # Start tracking from the next frame after the last known position.
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            prediction = self.kalman.predict()

            # Use Optical Flow to find new points
            old_frame = frame.copy()
            old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
            p0 = np.array([[prediction[0]], [prediction[1]]], dtype=np.float32).reshape(-1, 1, 2)
            lk_params = dict(winSize=(35, 35), maxLevel=0,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), p0, None,**lk_params)

            if p1 is not None and st[0]:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                # Update Kalman Filter with the new measurement
                measurement = np.array([[np.float32(good_new[0][0])], [np.float32(good_new[0][1])]])
                self.kalman.correct(measurement)

                # Check if the vehicle is within a reasonable area
                # This is a simple heuristic check; you might want to refine it.
                if 0 <= good_new[0][0] < self.frame_width and 0 <= good_new[0][1] < self.frame_height:
                    result.append(1)
                else:
                    result.append(0)
            else:
                result.append(0)

            if len(result) == self.cap.get(cv2.CAP_PROP_FRAME_COUNT) - len(self.xymap):
                break

        self.cap.release()
        return result


def track_after_vanish(transposed_xymap, check_map, video_path):
    # video_path = 'path_to_video.mp4'
    track_map = {}
    for id,frame in check_map.items():
        start_index, end_index = frame[0], frame[1]
        xylist_before_vanish = transposed_xymap[id][:start_index] #[-1,-1,-1,-1]之前的所有xy
        tracker = VehicleTracker(xylist_before_vanish, video_path)
        result = tracker.track_vehicle()
        # 检查 result 是否包含 0
        if 0 in result:
            track_map[id] = result
        # track_map[id] = result
    return track_map


def find_all_zero_blocks(transposed_xymap):
    # 转置 xymap 以便按车辆 ID 处理
    # transposed_xymap = list(map(list, zip(*xymap)))
    zero_block_map = {}

    for vehicle_id, lst in enumerate(transposed_xymap):
        zero_blocks = []
        n = len(lst)
        i = 0
        while i < n:
            if lst[i] == [-1, -1, -1, -1]:
                start_index = i
                while i < n and lst[i] == [-1, -1, -1, -1]:
                    i += 1
                end_index = i - 1
                # 确保找到的零块不在列表的开头或结尾，并且至少有一个元素,相差大于2 比如1 3 就不行
                # 增加时长阈值：从 2 改为 15 (约0.5秒)，过滤掉绝大多数检测抖动
                if start_index != 0 and end_index != n - 1 and (end_index - start_index > 6):
                    zero_blocks.append((start_index, end_index))
                if end_index == n - 1:
                    break
            else:
                i += 1

        if zero_blocks:
            zero_block_map[vehicle_id] = zero_blocks
    return zero_block_map

def check_adjacent_frame(zero_block_map, parallelogram_id_map):
    # zero_block_map  车辆id：消失时间帧索引列表 {1: [(12, 32),(34,38)], 18: [(35, 38)], 26: [(12, 19)], 30: [(14, 20)], 32: [(12, 23), (25, 32)], 33: [(10, 26)]}
    # parallelogram_id_map车辆周围三个平行四边形中车辆id存储，行为帧数,列为不同的车辆id，里面为其周围车辆id[0,3,5,…]
    # parallelogram_id_map = [[[] for _ in range(150)] for _ in range(total_frames)]
    check_map = {}
    # print(zero_block_map)
    for id, framelist in zero_block_map.items():
        flag = 0
        for frame in framelist:
            start_index, end_index = frame[0], frame[1]
            for other_id in range(len(parallelogram_id_map[start_index])):
                if id in parallelogram_id_map[start_index-1][other_id] and id not in parallelogram_id_map[start_index][other_id] \
                        and id in parallelogram_id_map[end_index+1][other_id] and id not in parallelogram_id_map[end_index][other_id]:
                    flag = 1
                    check_map[id] = frame
                    # print((other_id,id))
                    break
            if flag: #默认每个车辆的篡改只有一处
                break
    return check_map

# total_frames = 20
# height = 700
# edge = 5
# xymap = [[[-1.0, -1.0, -1.0, -1.0] for _ in range(150)] for _ in range(total_frames)]
# transposed_xymap = list(map(list, zip(*xymap)))
# print(len(transposed_xymap))
# print(len(transposed_xymap[0]))
# print(transposed_xymap)
# check_first_last_position(transposed_xymap, height, edge)

# def check_first_last_position(transposed_xymap, height, edge):
#     first_block_map = {}
#     last_block_map = {}
#     # 遍历某一个车辆的轨迹
#     for vehicle_id, lst in enumerate(transposed_xymap):
#         n = len(lst)
#         i = 0
#         while i < n:
#             if lst[i] != [-1, -1, -1, -1]:
#                 first_idx = i
#                 # 判断目标检测框的左上角或右下角的坐标是否有一个在图像上边缘或下边缘 x1,y1,x2,y2 分别左上和右下
#                 if not lst[first_idx][1] <= edge and not lst[first_idx][3] >= height - edge:
#                     # 若满足该条件 则说明第一次出现的位置不在图像边缘，有异常
#                     j = first_idx
#                     result = [first_idx,n-1]
#                     while j < n:
#                         if lst[j] != [-1, -1, -1, -1]:
#                             j += 1
#                         else:
#                             result[1] = j-1
#                             break  # 遇到 [-1, -1, -1, -1] 停止
#                     first_block_map[vehicle_id] = result
#                 break
#             i+=1
#
#         i = n-1
#         while i >= 0:
#             if lst[i] != [-1, -1, -1, -1]:
#                 last_idx = i
#                 # 判断目标检测框的左上角或右下角的坐标是否有一个在图像上边缘或下边缘 x1,y1,x2,y2 分别左上和右下
#                 if not lst[last_idx][1] <= edge and not lst[last_idx][3] >= height - edge:
#                     # 若满足该条件 则说明第一次出现的位置不在图像边缘，有异常
#                     j = last_idx
#                     result = [0,last_idx]
#                     while j >= 0:
#                         if lst[j] != [-1, -1, -1, -1]:
#                             j -= 1
#                         else:
#                             result[0] = j+1
#                             break  # 遇到 [-1, -1, -1, -1] 停止
#                     last_block_map[vehicle_id] = result
#                 break
#             i -= 1
#     return first_block_map, last_block_map