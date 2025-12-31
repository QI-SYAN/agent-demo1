from ultralytics import YOLO
from ultralytics.solutions import speed_estimation, object_counter
import cv2
import argparse
import copy
import imageio.v3 as iio
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import shutil
import json

from _dataprocess import *


CAR_TOOL_DISPLAY = os.environ.get("CAR_TOOL_DISPLAY", "0") == "1"


def _ensure_capacity(container, index, factory):
    missing = index - len(container) + 1
    if missing > 0:
        container.extend(factory() for _ in range(missing))


class _StaticTracker:
    """Adapter that feeds precomputed tracking results back into Ultralytics solutions."""

    def __init__(self, results: list):
        self._results = results

    def track(self, *args, **kwargs):  # pragma: no cover - simple adapter
        return self._results


def _apply_object_counter(counter_obj, frame, results_list):
    """Run the object counter solution using cached tracking outputs."""

    if counter_obj is None:
        return frame

    original_model = counter_obj.model
    try:
        counter_obj.model = _StaticTracker(results_list)
        solution_result = counter_obj.process(frame)
    finally:
        counter_obj.model = original_model

    return getattr(solution_result, "plot_im", frame)


def _apply_speed_estimator(speed_obj, frame, results_list, speed_row):
    """Run the speed estimation solution using cached tracking outputs."""

    original_model = speed_obj.model
    try:
        speed_obj.model = _StaticTracker(results_list)
        solution_result = speed_obj.process(frame)
    finally:
        speed_obj.model = original_model

    for track_id, speed in speed_obj.spd.items():
        if track_id < 0:
            continue
        _ensure_capacity(speed_row, track_id, lambda: -1.0)
        speed_row[track_id] = float(speed)

    return getattr(solution_result, "plot_im", frame)

def adjust_frame_size(frame, macro_block_size=16):
    h, w = frame.shape[:2]
    new_w = ((w + macro_block_size - 1) // macro_block_size) * macro_block_size
    new_h = ((h + macro_block_size - 1) // macro_block_size) * macro_block_size
    if (w, h) != (new_w, new_h):
        print(f"Resizing frame from ({w}, {h}) to ({new_w}, {new_h})")
        frame = cv2.resize(frame, (new_w, new_h))
    return frame

def count_vehicles_in_quadrants(xymap, circle_count_map, i, rmultiple):
    # vehicles0当前帧的所有车辆坐标
    vehicles0 = xymap[i][:]
    # 从后向前遍历列表，找到第一个非空元素的位置
    for j in range(len(vehicles0) - 1, -1, -1):
        if vehicles0[j] != [-1.0, -1.0, -1.0, -1.0]:
            break
    # 返回从开始到该位置的子列表
    vehicles = vehicles0[:j + 1]

    for track_id in range(len(vehicles)):
        vehicle = vehicles[track_id]
        x1, y1, x2, y2 = vehicle
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        # 动态计算r为当前目标的检测框上下边界的1.5倍
        r = abs(y2 - y1) * rmultiple

        # 初始化四个象限的计数器
        quadrant_counts = [0, 0, 0, 0]
        for other_track_id in range(len(vehicles)):
            if other_track_id == track_id:
                continue
            other_vehicle = vehicles[other_track_id]
            ox1, oy1, ox2, oy2 = other_vehicle
            other_center_x = (ox1 + ox2) / 2
            other_center_y = (oy1 + oy2) / 2
            dx = other_center_x - center_x
            dy = other_center_y - center_y

            if dx ** 2 + dy ** 2 <= r ** 2:  # 检查是否在半径r内
                if dx >= 0 and dy >= 0:
                    quadrant_counts[0] += 1
                elif dx < 0 and dy >= 0:
                    quadrant_counts[1] += 1
                elif dx < 0 and dy < 0:
                    quadrant_counts[2] += 1
                else:
                    quadrant_counts[3] += 1
        circle_count_map[i][track_id] = copy.deepcopy(quadrant_counts)

def draw_parallelograms_on_image(image, vehicle, angle_degrees, height_multiple, extend_width_multiple):
    """
    在给定的图像上绘制平行四边形。
    参数:
    - image: 输入的BGR格式图像（NumPy数组）。
    - vehicle: 车辆检测框坐标 [x1, y1, x2, y2]。
    - angle_degrees: 平行线与水平线之间的角度（度）。
    - height_multiple: 平行四边形高度相对于原始检测框高度的比例系数。
    - extend_width: 左右扩展平行四边形的宽度。
    """
    img_h, img_w = image.shape[:2]
    x1, y1, x2, y2 = vehicle
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    delta_x = abs(x2-x1)
    delta_y = abs(y2-y1)
    height = delta_y * height_multiple #所有平行四边形的宽度
    extend_width = delta_x * extend_width_multiple #旁边两个平行四边形的宽度

    # 将角度从度转换为弧度
    angle_radians = math.radians(angle_degrees)
    cos_angle = math.cos(angle_radians)
    sin_angle = math.sin(angle_radians)
    tan_angle = math.tan(angle_radians)
    width = delta_x + (delta_y/2)/tan_angle+50

    # 左右平行四边形的中心点x
    center_x_left = center_x-width/2-extend_width/2
    center_x_right = center_x+width/2+extend_width/2

    x1_left = max(center_x-width/2-(height/2)/tan_angle-extend_width,0)
    x2_left = max(center_x-width/2+(height/2)/tan_angle,0)
    x3_left = max(x1_left+extend_width,0)
    x4_left = max(x2_left-extend_width,0)

    x1_right = min(x3_left+width,img_w)
    x2_right = min(x2_left+width+extend_width,img_w)
    x3_right = min(x3_left+width+extend_width,img_w)
    x4_right = min(x2_left+width,img_w)
    y_up = max(0,center_y-height/2)
    y_down = min(img_h,center_y+height/2)

    # 定义颜色和线条厚度
    color = (0, 0, 255)  # 红色
    thickness = 2
    # 绘制左侧平行四边形
    pts_left = np.array([[x1_left, y_up], [x3_left, y_up], [x2_left, y_down], [x4_left, y_down]], np.int32)
    pts_left = pts_left.reshape((-1, 1, 2))
    cv2.polylines(image, [pts_left], isClosed=True, color=color, thickness=thickness)

    # 绘制右侧平行四边形
    pts_right = np.array([[x1_right, y_up],    [x3_right, y_up], [x2_right, y_down], [x4_right, y_down]], np.int32)
    pts_right = pts_right.reshape((-1, 1, 2))
    cv2.polylines(image, [pts_right], isClosed=True, color=color, thickness=thickness)

    # 绘制中心平行四边形
    pts_center = np.array(
        [[x3_left, y_up], [x1_right, y_up], [x4_right, y_down], [x2_left, y_down]],
        np.int32)
    pts_center = pts_center.reshape((-1, 1, 2))
    cv2.polylines(image, [pts_center], isClosed=True, color=color, thickness=thickness)
    # 显示图像
    if CAR_TOOL_DISPLAY:
        cv2.imwrite('Parallelograms.png', image)
        cv2.imshow('Parallelograms', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # points = [(x1_left,y_up),(x3_left,y_up),(x4_left,y_down),(x2_left,y_down),(x1_right,y_up),(x3_right,y_up),(x4_right,y_down),(x2_right,y_down)]
    return image

def count_vehicles_in_parallelograms(xymap, circle_count_map, i, left_angle_degrees, right_angle_degrees,height_multiple, extend_width_multiple,left_right_xboundary,image = None):
    """
    计算在特定平行四边形区域内的其他目标数量。

    参数:
    - xymap: 包含所有帧中所有车辆位置信息的列表，每个元素是单个帧中的车辆坐标列表。
    - circle_count_map: 用于存储每个车辆在不同帧中位于各个平行四边形内的情况。
    - i: 当前处理的帧索引。
    - angle_degrees: 定义平行线与水平线之间的角度（度）。
    - height_multiple: 平行四边形高度相对于原始检测框高度的比例系数。
    - extend_width: 左右扩展平行四边形的宽度。
    """
    # 获取当前帧的所有车辆坐标，排除无效数据(-1.0, -1.0, -1.0, -1.0)
    vehicles0 = xymap[i][:]
    for j in range(len(vehicles0) - 1, -1, -1):
        if vehicles0[j] != [-1.0, -1.0, -1.0, -1.0]:
            break
    vehicles = vehicles0[:j + 1]

    # 将角度从度转换为弧度，以便使用数学函数计算
    # angle_radians = math.radians(left_angle_degrees)
    # cos_angle = math.cos(angle_radians)
    # sin_angle = math.sin(angle_radians)
    # 如果提供了图像，则绘制平行四边形
    if image is not None:
        for vehicle in vehicles:
            if all(coord != -1.0 for coord in vehicle):  # 确保坐标不是无效数据
                x1, y1, x2, y2 = vehicle
                if (vehicle[0] + vehicle[2]) / 2 <= left_right_xboundary: #车辆位置在左边 角度取反
                    image = draw_parallelograms_on_image(image, vehicle, left_angle_degrees, height_multiple, extend_width_multiple)
                else:#车辆位置在右边
                    image = draw_parallelograms_on_image(image, vehicle, right_angle_degrees, height_multiple,
                                                         extend_width_multiple)
                break

        # 显示或保存带有绘制平行四边形的图像
        if CAR_TOOL_DISPLAY:
            cv2.imshow('Parallelograms', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        # 或者保存图像
        # cv2.imwrite('parallelograms_frame.png', image)

def draw_frame_number(image, frame_num):
    """
    在给定的图像右上角绘制当前帧序号以及汉字。

    参数:
        image: 要在其上绘制文本的图像。
        frame_num: 当前帧序号。

    返回:
        包含绘制了帧序号和汉字的新图像。
    """
    # 设置字体、位置、颜色等参数
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2  # 增大字体大小
    color = (0, 0, 255)  # BGR, red color for more visibility
    thickness = 3  # 加粗线宽
    text = f'{frame_num}frame'  # 添加文字

    # 获取文本尺寸以确保它不会超出图像边界
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    img_height, img_width, _ = image.shape

    # 设置文本位置为右上角，同时保证不被裁剪
    x = img_width - text_size[0] - 20  # 留出更多边缘空间
    y = text_size[1] + 20  # 确保文本底部有更多空间

    # 绘制文本到图像
    cv2.putText(image, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return image


# def write_to_file(filename, xymap, total_frames, i, names, classmap, speedmap, parallelogram_count_map, parallelogram_id_map, track_map):
#     with open(filename, 'w', encoding='utf-8') as file:
#         content = (
#             # f"frames:58-82\n"
#             # f"xymap: {xymap}\n"
#             f"track_map: {track_map}\n"
#             # f"total_frames: {total_frames}\n"
#             # f"i: {i}\n"
#             # f"names: {names}\n"
#             # f"classmap: {classmap[58:83]}\n"
#             # f"speedmap: {speedmap[58:83]}\n"
#             # f"circle_count_map: {parallelogram_count_map[58:83]}\n"
#             # f"parallelogram_count_map: {parallelogram_count_map}\n"
#             # f"parallelogram_id_map: {parallelogram_id_map}\n"
#         )
#         file.write(content)

def write_to_file(filename, xymap, total_frames, i, names, classmap, speedmap, parallelogram_count_map, parallelogram_id_map, track_map):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write("track_map:\n")  # 写入track_map头部信息
        for key, value in track_map.items():
            # 对于字典中的每个键值对，创建一个新行，键在行首，接着是值的列表
            file.write(f"{key}: {value}\n\n")  # 注意这里的'\n\n'会在每个键值对间添加一个空行

# 对每一帧的结果进行处理
def process_results(results, image, names, w, h, i,xymap,firstclass, roi_contour=None):
    numlist = [0,0,0,0]
    if hasattr(results, 'boxes') and results.boxes is not None:
        boxes = results.boxes
        for idx, box in enumerate(boxes.xyxy):
            x1, y1, x2, y2 = map(int, box) # 转换为整数坐标
            conf = boxes.conf[idx].item() # 获取置信度分数
            cls_id = int(boxes.cls[idx]) # 获取类别ID
            track_id = int(boxes.id[idx]) - 1  # !!!track_id才是每个车唯一的编号 从0开始
            if track_id < 0:
                continue

            if roi_contour is not None:
                bottom_center = (int((x1 + x2) / 2), y2)
                if cv2.pointPolygonTest(roi_contour, bottom_center, False) < 0:
                    continue

            if track_id in firstclass:
                class_name = firstclass[track_id]
            else:
                class_name = names[cls_id] # 根据类别ID查找对应的类别名称
                firstclass[track_id] = class_name

            _ensure_capacity(xymap[i], track_id, lambda: [-1.0, -1.0, -1.0, -1.0])
            xymap[i][track_id] = list(map(lambda x: x.item(), box))
            if class_name=='car':
                numlist[0] += 1
            elif class_name=='truck':
                numlist[1] += 1
            elif class_name=='bus':
                numlist[2] += 1
            else:
                numlist[3] += 1

            # 在图像上绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 如果启用了跟踪功能，打印并绘制跟踪ID
            if hasattr(boxes, 'id') and boxes.id is not None:
                label = f'ID: {track_id}'
                # 计算文本的宽度和高度
                (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                # 文本的位置：框的底部中央
                label_x = x1 + (x2 - x1 - label_width) // 2
                label_y = y2 + label_height + 5
                # 确保标签不超出图像边界
                label_x = max(0, min(w - label_width, label_x))
                label_y = max(label_height, min(h, label_y))
                # 在图像上绘制标签
                cv2.putText(image, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # print(
                #     f"Track ID {track_id}: ({x1}, {y1}) -> ({x2}, {y2}), Confidence: {conf:.2f}, Class: {class_name}, Track ID: {track_id}")
    return numlist

def manage_folder(folder_path):
    if os.path.exists(folder_path):
        # 如果文件夹存在，清空文件夹内的所有内容
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # 删除文件或符号链接
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # 删除子文件夹及其内容
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        # 如果文件夹不存在，则创建文件夹
        try:
            os.makedirs(folder_path)
            print(f"Folder created at {folder_path}")
        except Exception as e:
            print(f'Failed to create folder {folder_path}. Reason: {e}')
def draw_and_count(image, xymap, parallelogram_count_map, parallelogram_id_map, i, left_angle_degrees, right_angle_degrees, height_multiple,
                   extend_width_multiple, left_right_xboundary, draw_filename, draw=True):
    # 针对第i帧（从0开始）图像，已经遍历过了
    """
    在给定的图像上绘制平行四边形，并计算特定区域内的其他目标数量。

    参数:
    - image: 输入的BGR格式图像（NumPy数组），如果为None则不进行绘图。
    - xymap: 包含所有帧中所有车辆位置信息的列表，每个元素是单个帧中的车辆坐标列表。
    - i: 当前处理的帧索引。
    - vehicle: 车辆检测框坐标 [x1, y1, x2, y2]。
    - left_angle_degrees: 左侧平行线与水平线之间的角度（度）。
    - right_angle_degrees: 右侧平行线与水平线之间的角度（度）。
    - height_multiple: 平行四边形高度相对于原始检测框高度的比例系数。
    - extend_width_multiple: 左右扩展平行四边形的宽度比例系数。
    - draw: 是否在图像上绘制平行四边形。

    返回:
    - 绘制了平行四边形的图像（如果提供了image且draw为True）。
    - 三个平行四边形内的其他目标数量。
    """
    if image is None:
        return

    # 获取当前帧的所有车辆坐标，排除末尾的连续无效数据(-1.0, -1.0, -1.0, -1.0)
    vehicles0 = xymap[i][:]
    for j in range(len(vehicles0) - 1, -1, -1):
        if vehicles0[j] != [-1.0, -1.0, -1.0, -1.0]:
            break
    vehicles = vehicles0[:j + 1]
    for track_id in range(len(vehicles)):
        image_copy = image.copy()
        vehicle = vehicles[track_id]
        # 遍历每一个车辆目标框
        if not all(coord != -1.0 for coord in vehicle):  # 确保坐标不是无效数据
            continue
        # 获取当前车辆的坐标和尺寸
        x1, y1, x2, y2 = vehicle
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        delta_x = abs(x2 - x1)
        delta_y = abs(y2 - y1)
        height = delta_y * height_multiple # 所有平行四边形的高度
        # 根据车辆中心点的位置选择角度
        angle_degrees = left_angle_degrees if center_x <= left_right_xboundary else right_angle_degrees
        # 将角度从度转换为弧度
        angle_radians = math.radians(angle_degrees)
        tan_angle = math.tan(angle_radians)
        width = delta_x + (delta_y / 2) / tan_angle + 50 # 中间平行四边形的宽度
        extend_width = delta_x * extend_width_multiple # 旁边两个平行四边形的宽度
        img_h, img_w = image.shape[:2]

        x1_left = max(center_x - width / 2 - (height / 2) / tan_angle - extend_width, 0)
        x2_left = max(center_x - width / 2 + (height / 2) / tan_angle, 0)
        x3_left = max(x1_left + extend_width, 0)
        x4_left = max(x2_left - extend_width, 0)

        x1_right = min(x3_left + width, img_w)
        x2_right = min(x2_left + width + extend_width, img_w)
        x3_right = min(x3_left + width + extend_width, img_w)
        x4_right = min(x2_left + width, img_w)
        y_up = max(0, center_y - height / 2)
        y_down = min(img_h, center_y + height / 2)

        # # 定义颜色和线条厚度
        # color = (0, 0, 255)  # 红色
        # thickness = 2
        # 左侧平行四边形
        pts_left = np.array([[x1_left, y_up], [x3_left, y_up], [x2_left, y_down], [x4_left, y_down]], np.int32)
        # 右侧平行四边形
        pts_right = np.array([[x1_right, y_up], [x3_right, y_up], [x2_right, y_down], [x4_right, y_down]],
                             np.int32)
        # 中心平行四边形
        pts_center = np.array(
            [[x3_left, y_up], [x1_right, y_up], [x4_right, y_down], [x2_left, y_down]],
            np.int32)

        parallelograms = [pts_left, pts_center, pts_right] #左中右
        parallelogram_counts = [0, 0, 0] #左中右
        parallelogram_ids = []
        # 遍历同一帧中的其他车辆，检查它们是否位于任何一个平行四边形内
        for other_track_id in range(len(vehicles)):
            if other_track_id == track_id:
                continue  # 跳过自身比较
            other_vehicle = vehicles[other_track_id]
            if not all(coord != -1.0 for coord in other_vehicle):  # 确保坐标不是无效数据 注意！！！！！
                continue
            ox1, oy1, ox2, oy2 = other_vehicle
            other_center_x = (ox1 + ox2) / 2  # 其他车辆的中心点X坐标
            other_center_y = (oy1 + oy2) / 2  # 其他车辆的中心点Y坐标
            for idx, pg in enumerate(parallelograms):
                if cv2.pointPolygonTest(pg, (other_center_x, other_center_y), False) >= 0:
                    parallelogram_counts[idx] += 1 #修改计数器
                    parallelogram_ids.append(other_track_id)
                    break  # 一旦找到匹配的平行四边形就跳出循环，避免重复计数与增加时间
            # 更新parallelogram_count_map，保存当前车辆在各平行四边形内的计数结果

        _ensure_capacity(parallelogram_count_map[i], track_id, list)
        _ensure_capacity(parallelogram_id_map[i], track_id, list)
        parallelogram_count_map[i][track_id] = copy.deepcopy(parallelogram_counts)
        parallelogram_id_map[i][track_id] = copy.deepcopy(parallelogram_ids)
        if draw:
            for idx, pg in enumerate(parallelograms):
                cv2.polylines(image_copy, [pg], isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.imwrite(draw_filename+'/'+str(i)+'_id'+str(track_id)+'.png', image_copy)
def data_process(xymap, parallelogram_id_map, video_path, height, edge, total_frames):
    transposed_xymap = list(map(list, zip(*xymap)))

    # 非连续存在情况(中间消失)
    # 找到所有不在最前和最后的车辆不存在的列表边缘索引
    zero_block_map = find_all_zero_blocks(transposed_xymap)  # {1: [(12, 32)], 18: [(35, 38)], 26: [(12, 19)], 30: [(14, 20)], 32: [(12, 23), (25, 32)], 33: [(10, 26)]}
    check_map = check_adjacent_frame(zero_block_map, parallelogram_id_map)
    # check_map: {1: (12, 32), 18: (35, 38), 26: (12, 19), 30: (14, 20), 32: (12, 23), 33: (10, 26)}
    track_map = track_after_vanish(transposed_xymap, check_map, video_path)

    # 同时返回 track_map（卡尔曼跟踪结果）和 check_map（原始消失起止帧）
    return track_map, check_map

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default=r"D:\resouce_bs\best2.pt")
    # parser.add_argument('--weight', type=str, default='yolov10s.pt')
    parser.add_argument('--save_video', action='store_true', help='Save the processed video')
    # parser.add_argument('--input_video_path', type=str,
    #                     default='./videos/input/1202rootdatasets_1210_01_45_results.mp4', help='source video path.')
    # parser.add_argument('--output_video_path', type=str,
    #                     default='./videos/output/1202rootdatasets_test21.mp4',
    #                     help='output video inference result storage path.')

    # parser.add_argument('--input_video_path', type=str,
    #                     default='./videos/input/tiny1202rootdatasets_1210_01_45_results.mp4', help='source video path.') #可改
    # parser.add_argument('--output_video_path', type=str,
    #                     default='./videos/output/tiny1202rootdatasets_1210_test8.mp4',
    #                     help='output video inference result storage path.') #要改
    # parser.add_argument('--input_video_path', type=str,
    #                     default='./videos/input/tiny1202rootdatasets_0202_23_49_results2.mp4',
    #                     help='source video path.') #可改
    # parser.add_argument('--output_video_path', type=str,
    #                     default='./videos/outputnew/20250302_3/tiny1202rootdatasets_0202_23_49_results2_output.mp4',
    #                     help='output video inference result storage path.') #要改


    parser.add_argument('--input_video_path', type=str,
                        default='./videos/input/1202rootdatasets_1210_01_45_results.mp4',
                        help='source video path.') #可改
    parser.add_argument('--output_video_path', type=str,
                        default='./videos/outputnew/20250303long_NoiseCov005_win35_maxlevel0/1202rootdatasets_1210_01_45_results_output.mp4',
                        help='output video inference result storage path.') #要改!!!!
    # parser.add_argument('--input_video_path', type=str,
    #                     default='./videos/input/tiny1202rootdatasets_1210_01_45_results.mp4',
    #                     help='source video path.') #可改
    # parser.add_argument('--output_video_path', type=str,
    #                     default='./videos/outputnew/20250303_processNoiseCov005/tiny1202rootdatasets_1210_01_45_results_output.mp4',
    #                     help='output video inference result storage path.') #要改!!!!
    opt = parser.parse_args()
    return opt

def main():
    processed_frames = []
    line_pts = [(0, 615), (1920, 615)]
    opt = parse_opt()
    rootdir = os.path.dirname(opt.output_video_path)
    manage_folder(rootdir)
    txtoutputpath = os.path.join(rootdir, 'result.txt')
    draw_filename = os.path.join(rootdir, 'images').replace('\\', '/') #重要
    # txtoutputpath = 'videos/output0302_1.txt'#要改
    # draw_filename = 'videos/output/0302_1/'#要改
    # 使用函数时，提供参数如角度、高度倍数以及扩展宽度
    # rmultiple = 2
    left_angle_degrees = -70
    right_angle_degrees = 75  # 平行线与水平线之间的夹角（度）,这个角度是以度为单位的，并且是按照标准数学坐标系中的逆时针方向来计算的。
    # 如果 angle_degrees 是正数（例如 45 度），那么平行线是从水平线向左上方倾斜的，即相对于水平线左偏；如果是负数（例如 -45 度），那么平行线是从水平线向右上方倾斜的，即相对于水平线右偏。
    height_multiple = 4  # 平行四边形的高度是检测框上下边界距离的5倍
    extend_width_multiple = 1.2 # 左右扩展平行四边形的宽度
    left_right_xboundary = 800 #左右不同方向的车辆x坐标划分线
    edge = 5
    model = YOLO(opt.weight)
    names = model.model.names
    cap = cv2.VideoCapture(opt.input_video_path)
    assert cap.isOpened(), "Illegal or non-existing video file"
    ret, frame = cap.read()
    height = 1000
    # 获取视频的总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if ret:
        # 获取帧的宽度和高度
        width = frame.shape[1]
        height = frame.shape[0]
        print(f"视频帧的宽度: {width}, 高度: {height}")

    else:
        print("无法读取视频帧")

    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    def _clamp_axis(value, upper_bound):
        if upper_bound <= 0:
            return 0
        return max(0, min(upper_bound - 1, int(value)))

    roi_contour = np.array([
        [0, _clamp_axis(1000, h)],
        [0, _clamp_axis(675, h)],
        [_clamp_axis(270, w), _clamp_axis(435, h)],
        [_clamp_axis(w - 1, w), _clamp_axis(435, h)],
        [_clamp_axis(w - 1, w), _clamp_axis(1000, h)],
    ], dtype=np.int32)

    # Ensure the Ultralytics helpers reuse the provided weights and avoid downloading defaults.
    speed_obj = speed_estimation.SpeedEstimator(model=opt.weight, show=CAR_TOOL_DISPLAY)
    # Newer Ultralytics releases expose configuration via the CFG dictionary instead of set_args.
    speed_obj.CFG["region"] = line_pts
    speed_obj.region = line_pts
    speed_obj.initialize_region()
    speed_obj.names = names

    counter_obj = object_counter.ObjectCounter(model=opt.weight, show=CAR_TOOL_DISPLAY)
    counter_obj.CFG["region"] = line_pts
    counter_obj.region = line_pts
    counter_obj.initialize_region()
    counter_obj.region_initialized = True
    counter_obj.names = names

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 车辆位置存储表，行为帧数,列为不同的车辆id，里面为x1, y1, x2, y2
    xymap = [[[-1.0, -1.0, -1.0, -1.0] for _ in range(150)] for _ in range(total_frames)]
    # xymap = [[[-1.0, -1.0, -1.0, -1.0] for _ in range(total_frames*20)] for _ in range(total_frames)]
    # 车辆类型记录列表
    classmap = [[0,0,0,0] for _ in range(total_frames)]
    # 车辆首个类型存储列表
    firstclass = {}
    # # 车辆的速度之和列表
    # sumspeed = []
    # 车辆速度存储表，行为帧数，列为不同的车辆id，预计共有150个不同的车辆，第一行没有速度因为上一帧不存在
    speedmap = [[-1.0 for _ in range(150)] for _ in range(total_frames)]
    # # 车辆周围r半径的圆内四个象限的车辆数统计表，行为帧数,列为不同的车辆id，里面为四个象限的统计数
    # circle_count_map = [[[0, 0, 0, 0] for _ in range(150)] for _ in range(total_frames)]
    # 车辆周围三个平行四边形中车辆数统计表，行为帧数,列为不同的车辆id，里面为统计数[0,0,0]
    parallelogram_count_map = [[[] for _ in range(150)] for _ in range(total_frames)]
    # 车辆周围三个平行四边形中车辆id存储，行为帧数,列为不同的车辆id，里面为车辆id[0,3,5,…]
    parallelogram_id_map = [[[] for _ in range(150)] for _ in range(total_frames)]
    manage_folder(draw_filename)
    i = 0
    while cap.isOpened():
        # 每一帧
        success, im0 = cap.read()
        if not success:
            break
        if roi_contour.size:
            cv2.polylines(im0, [roi_contour], True, (0, 255, 255), 2)
        tracking_results = model.track(im0, persist=True, show=False)
        results_list = tracking_results if isinstance(tracking_results, list) else [tracking_results]
        numlist = [0, 0, 0, 0]
        for results in results_list:
            frame_counts = process_results(results, im0, names, w, h, i, xymap, firstclass, roi_contour)
            for idx in range(len(numlist)):
                numlist[idx] += frame_counts[idx]
        classmap[i] = copy.deepcopy(numlist)
        # 当前帧的xymap已经做好了 为xymap[i],可以找到当前帧的车辆之间相对位置
        # 车辆位置存储表单行，行为不同的车辆id，里面为x1, y1, x2, y2
        # count_vehicles_in_quadrants(xymap, circle_count_map, i, rmultiple)
        # count_vehicles_in_parallelograms(xymap, parallelogram_count_map, i, left_angle_degrees, right_angle_degrees, height_multiple, extend_width_multiple,left_right_xboundary,im0)
        # draw_and_count(xymap, parallelogram_count_map, i, left_angle_degrees, right_angle_degrees,
        #                height_multiple, extend_width_multiple,left_right_xboundary,im0)

        draw_and_count(im0, xymap, parallelogram_count_map, parallelogram_id_map, i, left_angle_degrees, right_angle_degrees,
                       height_multiple, extend_width_multiple, left_right_xboundary, draw_filename,draw=False)
        # 让 Ultralytics Solutions 使用缓存的 tracking 结果完成计数与速度标注。
        im0 = _apply_object_counter(counter_obj, im0, results_list)
        im0 = _apply_speed_estimator(speed_obj, im0, results_list, speedmap[i])

        rgb_frame = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
        rgb_frame = adjust_frame_size(rgb_frame)  # 调整帧大小
        # 在右上角绘制当前帧序号 i
        rgb_frame = draw_frame_number(rgb_frame, i)

        processed_frames.append(rgb_frame)
        print(f"Processed frame {i}")
        i += 1
    cap.release()

    if xymap:
        max_track_slots = max(len(row) for row in xymap)
        if max_track_slots > 0:
            target_index = max_track_slots - 1
            for row in xymap:
                _ensure_capacity(row, target_index, lambda: [-1.0, -1.0, -1.0, -1.0])
            for row in speedmap:
                _ensure_capacity(row, target_index, lambda: -1.0)
            for row in parallelogram_count_map:
                _ensure_capacity(row, target_index, list)
            for row in parallelogram_id_map:
                _ensure_capacity(row, target_index, list)

    track_map, check_map = data_process(xymap, parallelogram_id_map, opt.input_video_path, height, edge, total_frames)
    write_to_file(txtoutputpath, xymap, total_frames, i, names, classmap, speedmap, parallelogram_count_map, parallelogram_id_map, track_map)

    # 额外导出每个疑似篡改轨迹的原始消失起止帧（来自 check_map）
    try:
        segments_output_path = os.path.join(rootdir, 'segments.json')
        segments_data = {}
        for vehicle_id, frame_range in check_map.items():
            try:
                start_index, end_index = int(frame_range[0]), int(frame_range[1])
            except Exception:
                continue
            segments_data[str(vehicle_id)] = {
                "start": start_index,
                "end": end_index,
            }
        with open(segments_output_path, 'w', encoding='utf-8') as f:
            json.dump(segments_data, f, ensure_ascii=False)
        print(f"Saved segment metadata to {segments_output_path}")
    except Exception as e:
        print(f"Warning: failed to write segments.json: {e}")
    # 额外导出每帧每车的检测框，供后续截图时高亮使用
    try:
        bbox_output_path = os.path.join(rootdir, 'bboxes.json')
        bbox_data = {}
        for frame_idx, frame_boxes in enumerate(xymap):
            frame_dict = {}
            for track_id, box in enumerate(frame_boxes):
                if box == [-1.0, -1.0, -1.0, -1.0]:
                    continue
                try:
                    x1, y1, x2, y2 = [int(round(v)) for v in box]
                except Exception:
                    continue
                frame_dict[str(track_id)] = [x1, y1, x2, y2]
            if frame_dict:
                bbox_data[str(frame_idx)] = frame_dict
        with open(bbox_output_path, 'w', encoding='utf-8') as f:
            json.dump(bbox_data, f, ensure_ascii=False)
        print(f"Saved bbox metadata to {bbox_output_path}")
    except Exception as e:
        print(f"Warning: failed to write bboxes.json: {e}")

   # 将处理的结果保存为视频
    try:
        iio.imwrite(opt.output_video_path, processed_frames, plugin="FFMPEG", fps=fps)
        print("Video saved successfully.")
    except Exception as e:
        print(f"Failed to save video: {e}")

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()