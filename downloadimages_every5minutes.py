# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time   : 2023/10/11 上午11:03
@Author : xiaoshijia
@Email  : xiaoshijia@cpgroup.cn
@File   : pull_stream.py
@IDE    : PyCharm
@Description : 拉流方法
"""
import cv2
import time
import traceback
from ultralytics import YOLO
import cv2
import numpy as np
from tqdm import tqdm
import shutil
import os
from datetime import datetime
  
def pigcount(img, area):
    
    #model = YOLO('/home/deepl/ultralytics/runs/segment/train7/weights/best.pt')  # load a custom model
    #model = YOLO('/home/deepl/ultralytics/checkpoint/train0002/weights/best.pt')
    model = YOLO('/home/deepl/ultralytics_926revise/checkpoint/82/weights/best.pt')
    #model = YOLO('/home/deepl/ultralytics/1009.pt')
    #mask = np.zeros_like(img[:, :, 0])
    new_mask = np.zeros_like(img[:, :, 0])
    cv2.fillPoly(new_mask, [area], 255)
    
    masked_img = cv2.bitwise_and(img, img, mask=new_mask)
    #  model = YOLO('yolov8n-seg.yaml').load('yolov8n.pt')
    #  Predict with the model
    results = model(masked_img)  # predict on an image
    # res = model(img)
    res_plotted = results[0].plot()
    timeline_area = img[99: 181, :]
    combined_img = np.vstack((timeline_area, res_plotted))
    
    return combined_img, len(results[0].boxes), masked_img


def get_stream_frame(cap, item):
    ret, frame = None, None
    # noinspection PyBroadException
    try:
        ret, frame = cap.read()
    except Exception:
        print(traceback.format_exc())
    if ret:
        return ret, frame
    else:
        cap.release()
        cap.open(item)
        cap.grab()
        raise RuntimeError("ret none, retrying!")


if __name__ == '__main__':
    stream_address = "rtmp://qvs-live-rtmp.cpgroup.cn:2045/2xenzw32d1rf9/31011500991180017737_34020000001310000025"
    c = cv2.VideoCapture(stream_address)
    pts = np.array([(518.4, 415.2), (535.1999999999999, 1012.8), (662.4, 1392.0), (780.0, 1579.1999999999998), 
                    (1123.2, 1538.3999999999999), (1404.0, 1396.8), (1365.6, 813.5999999999999), (1118.3999999999999, 290.4), (518.4, 415.2)], np.int32)
    #today = datetime.date.today()  
    current_datetime = datetime.now()
    today = current_datetime.date()
    # 创建文件夹名称  
    folder_name = today.strftime("%Y-%m-%d")
    if not os.path.exists(folder_name):
        # 创建文件夹  
        os.mkdir(folder_name)
    count = 0 
    while True:
        r, f = get_stream_frame(cap=c, item=stream_address)
        #img, pig_count, masked_img = pigcount(f, pts)
        now = datetime.now()   
        # 省略年月日，只保留时间部分  
        time_only = now.time()  
        # 将时间格式化为字符串，并省略前导零和不必要的小数部分  
        formatted_time = time_only.strftime("%H:%M:%S") 
        #cv2.imwrite("{}/{}_result.{}".format(folder_name, formatted_time, "jpg"), img)
        cv2.imwrite("{}/{}.{}".format(folder_name, formatted_time, "jpg"), f)
        break
        #cv2.imwrite("output/{}".format(i), masked_img)
        #shutil.copy(full_path, "{}/{}".format(prefix, i))
        
