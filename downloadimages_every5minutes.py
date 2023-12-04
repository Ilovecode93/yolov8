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
from datetime import datetime
import numpy as np
import os

def get_stream_frame(cap, item):
    """
    Attempts to read a frame from the stream. If unsuccessful, tries to reopen the stream.
    """
    try:
        ret, frame = cap.read()
        if ret:
            return ret, frame
        else:
            cap.release()
            cap.open(item)
            cap.grab()
            raise RuntimeError("Frame retrieval failed, retrying!")
    except Exception:
        print(traceback.format_exc())
        return None, None

if __name__ == '__main__':
    stream_address = "rtmp://qvs-live-rtmp.cpgroup.cn:2045/2xenzw32d1rf9/31011500991180017737_34020000001310000025"
    c = cv2.VideoCapture(stream_address)

    pts = np.array([(518.4, 415.2), (535.2, 1012.8), (662.4, 1392.0), (780.0, 1579.2), 
                    (1123.2, 1538.4), (1404.0, 1396.8), (1365.6, 813.6), (1118.4, 290.4), (518.4, 415.2)], np.int32)

    today = datetime.now().date()
    folder_name = today.strftime("%Y-%m-%d")

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    while True:
        r, f = get_stream_frame(cap=c, item=stream_address)
        if f is not None:
            now = datetime.now()
            formatted_time = now.strftime("%H-%M-%S")
            cv2.imwrite(f"{folder_name}/{formatted_time}.jpg", f)
            break  # Remove or modify this line if you want the loop to continue
        else:
            print("Failed to capture frame, retrying...")
            time.sleep(1)  # Adding a short delay before retrying can be helpful
        
