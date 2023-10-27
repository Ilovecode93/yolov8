#!/bin/bash

# 获取当前的小时和分钟
current_hour=$(date +"%H")
current_minute=$(date +"%M")

# 检查时间是否超过 16:40
if [ "$current_hour" -lt "17" ] || ([ "$current_hour" -eq "17" ] && [ "$current_minute" -le "40" ]); then
    cd /home/deepl/ultralytics/ 
    /home/deepl/miniconda3/envs/yolotrt/bin/python downloadimages_every5minutes.py
fi

