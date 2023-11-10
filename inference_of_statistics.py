# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time   : 2023/7/19 上午11:22
@Author : xiaoshijia
@Email  : xiaoshijia@cpgroup.cn
@File   : inference_of_statistics.py
@IDE    : PyCharm
@Description : 说明
"""
import traceback
import cv2
import gc
import logging
import time
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from c_0216.utils.yaml_config import YAMLConfig
from c_0216.utils.pigcount_mid_status import status_list, status_update
from c_0216.utils.result_synchronization import range_statistics, AIOT_CN_ENUM
from c_0216.utils.aiplogging import md_logger
from c_0216.utils.sql_status import SqliteHelper
from c_0216.utils.image_utils import image_numpy_to_base64, get_stream_frame
from conf.base_conf import PUSH_MODE, PIC_FILE_PATH

md_logger("single_pig_statistics_interval_inference")


def judge_result(count, pic_path, pig_dict):
    """
    当前检测结果，与众数差值=>4，返回当前结果，反之返回历史
    :param count: 当前检测结果数量
    :param pic_path: 当前检测画面
    :param pig_dict: 历史结果
    :return:
    """
    res_pic_path = None
    res_number_pig = None
    max_count = max(pig_dict.values(), key=lambda v: v[2])[2]  # find the maximum count
    res = {k: v for k, v in pig_dict.items() if v[2] == max_count}
    for _, v in res.items():
        res_number_pig = v[0]
        res_pic_path = v[1]
    abs_count = abs(count - res_number_pig)
    if abs_count >= 4:
        logging.warning(f"\n警告,猪只统计值变化较大: {abs_count}\n")
        return count, pic_path
    else:
        return res_number_pig, res_pic_path


c = YAMLConfig()
print(f"YAML DICT\n{c.metadata}\n")
minute_dict, second_dict = c.get_camera_configs()
camera_configs = minute_dict
model_configs = c.get_model_configs()
print(f"Model Configs:\n{model_configs}\n")
print(f"Camera Configs:\n{camera_configs}\n")
sql = SqliteHelper()


def run(items, yolo_detector_dict, trigger_type="manual"):
    camera_stream_list = []
    body_list = []
    pig_dict = {}
    body = None
    for item in items:
        camera_stream_list.append(
            (item["stream_address"], cv2.VideoCapture(item["stream_address"]), item["algorithms"]))
    frame_count = 0
    ret = None
    stop_tag = False
    if PUSH_MODE == "HTTP":
        executor = ThreadPoolExecutor(max_workers=50)
    else:
        executor = None
    pts = np.array([(535, 494), (532, 777), (573, 1005), (662, 1272),
                     (721, 1392), (826, 1553), (1005, 1539), (1164, 1498), (1288, 1443), (1364, 1400),
                     (1373, 1226), (1377, 1011), (1333, 806), (1297, 707), (1219, 544), (1169, 461), (1091, 364),
                     (802, 395)], np.int32)
    start_timestamp = time.time()
    start_time = time.strftime("%Y%m%d%H%M%S", time.localtime(start_timestamp))
    day_time = time.strftime("%Y%m%d", time.localtime(start_timestamp))
    hour_time = time.strftime("%H", time.localtime(start_timestamp))
    while True:
        triggering_time = int(time.time())
        triggering_time = triggering_time - triggering_time % 60
        triggering_time = int(triggering_time * 1000)
        # triggering_time = 1684656000000  # 指定时间
        stream_index = 0
        for camera_info in camera_stream_list:
            stream_address = camera_info[0]
            cap = camera_info[1]
            algorithms = camera_info[2]
            if frame_count == 0 or frame_count % 20 != 0:  # 分钟级的处理逻辑
                ret = cap.grab()
                continue
            if ret:
                try:
                    ret, frame = get_stream_frame(cap=cap, item=stream_address)
                except Exception as e:
                    logging.error(e)
                    break
                # ret, frame = cv2.VideoCapture("/home/xiaoshijia/Desktop/ds.mp4").read()  # 指定文件检测
                for algorithm in algorithms:
                    model_name = model_configs[algorithm][0]
                    if model_name.endswith('pt'):
                        # noinspection PyBroadException
                        try:
                            mask = np.zeros_like(frame[:, :, 0])
                            timeline_area = frame[99: 181, :]
                            cv2.fillPoly(mask, [pts], 255)
                            masked_img = cv2.bitwise_and(frame, frame, mask=mask)
                            results = yolo_detector_dict[algorithm](masked_img)
                            img = results[0].plot()
                            count = len(results[0].boxes)
                            combined_img = np.vstack((timeline_area, img))
                            dir_path = os.path.join(PIC_FILE_PATH, day_time, algorithm,
                                                    stream_address.split("/")[-1])
                            if not os.path.exists(dir_path):
                                os.makedirs(dir_path)
                            pic_path = f"{dir_path}/{start_time}-{frame_count}.jpg"
                            # cv2.imwrite(pic_path, img)  # 算法检测图片
                            cv2.imwrite(pic_path, combined_img)  # 还原时间轴图片
                            pig_dict = status_list(stream_address=stream_address, algorithm=algorithm)
                            logging.debug(f"\nRedis查询结果:\n\t{pig_dict}\n")
                            key = f"{start_time}-{count}"
                            logging.debug(f"Key:\n\t{key}")
                            if key in pig_dict:
                                pig_dict[key][0] = count
                                pig_dict[key][1] = pic_path
                                pig_dict[key][2] += 1  # increment the count
                            else:
                                pig_dict[key] = [count, pic_path, 1]  # initialize the count to 1
                            count, pic_path = judge_result(count, pic_path, pig_dict)
                            if trigger_type == "auto":  # 只有自动盘点需要入库
                                status_update(stream_address=stream_address, algorithm=algorithm, data=pig_dict)
                            status = f"COUNT: {count}"
                        except Exception:
                            logging.error(traceback.format_exc())
                            continue
                    else:
                        continue
                    # if trigger_type == "auto":
                    # 自动盘点：持续1分钟 做 众数逻辑
                    # 手动盘点：持续20帧 做 视频流缓冲
                    if trigger_type == "auto" and time.time() - start_timestamp >= 5:
                        stop_tag = True
                        logging.warning("Time over\n")
                        if hour_time in ["16"] and pig_dict:
                            tmp_dict = {}
                            for value in pig_dict.values():
                                # The pigcount is the first item in the list
                                pigcount = value[0]
                                pigcount_appear = value[2]
                                # Add the count for each pigcount
                                if pigcount in tmp_dict:
                                    tmp_dict[pigcount] += pigcount_appear
                                else:
                                    tmp_dict[pigcount] = pigcount_appear
                            sorted_number = sorted(tmp_dict.items(), key=lambda item: item[1], reverse=True)
                            most_common_number = sorted_number[0][0]
                            res = {k: v for k, v in pig_dict.items() if v[0] == most_common_number}
                            target_time = "1700"
                            res_pic_path = None
                            res_number_pig = None
                            closest_key = max(res, key=lambda k: abs(int(k.split('-')[0]) - int(target_time)))
                            for tmp_key, v in res.items():
                                if tmp_key == closest_key: 
                                    res_number_pig = v[0]
                                    res_pic_path = v[1]
                                    
                            sql.update_ai_status(algorithm=algorithm, camera_path=stream_address, ai_status=status)
                            if res_pic_path:
                                body = (
                                    AIOT_CN_ENUM[algorithm],
                                    stream_address,
                                    res_number_pig,
                                    res_pic_path,
                                    triggering_time,
                                    "path",
                                    trigger_type
                                )
                                body_list.append(body)
                            else:
                                logging.warning("当前没有检测结果，算法检测异常")
                    elif trigger_type == "manual" and frame_count != 0 and frame_count % 20 == 0:
                        stop_tag = True
                        sql.update_ai_status(algorithm=algorithm, camera_path=stream_address, ai_status=status)
                        body = (
                            AIOT_CN_ENUM[algorithm],
                            stream_address,
                            count,
                            pic_path,
                            triggering_time,
                            "path",
                            trigger_type
                        )
                        body_list.append(body)
                    if executor and stop_tag and body_list:
                        executor.submit(
                            range_statistics, body
                        )
                    logging.debug("Frame over\n")
            else:
                cap.open(stream_address)
            stream_index += 1
        if stop_tag:
            break
        frame_count += 1
    for camera_info in camera_stream_list:
        cap = camera_info[1]
        cap.release()
    del executor
    gc.collect()
    print(f"body_list: {body_list}")
    return body_list
