import cv2
import os
import argparse
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import random    
import os
import glob
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.gridspec as gridspec

# def pigcount(model, img, area):
#     mask = np.zeros_like(img[:, :, 0])
#     cv2.fillPoly(mask, [area], 255)
#     masked_img = cv2.bitwise_and(img, img, mask=mask)
#     results = model(masked_img)  # predict on an image
#     res_plotted = results[0].plot()
#     return res_plotted, len(results[0].boxes), masked_img
def pigcount(img, area):
    
    #model = YOLO('/home/deepl/ultralytics/runs/segment/train7/weights/best.pt')  # load a custom model
    #model = YOLO('/home/deepl/ultralytics/checkpoint/train0002/weights/best.pt')
    #model = YOLO('/home/deepl/ultralytics/smallpig_weights/1017/weights/best.pt')
    #model = YOLO('/home/deepl/ultralytics_926revise/checkpoint/82/weights/best.pt')
    #model = YOLO('/home/deepl/ultralytics/1011weights/weights/best.pt')
    model = YOLO('/home/deepl/ultralytics/smallpig_weights/1018_6:30/weights/best.pt')
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
    
    font = cv2.FONT_HERSHEY_SIMPLEX  # 选择字体
    font_scale = 3  # 字体大小
    font_color = (255, 255, 255)  # 字体颜色，例如白色
    font_thickness = 2  # 字体厚度
    text_position = (10, 300)  # 文本位置（距离左上角的像素）
    text = f'Pig count: {len(results[0].boxes)}'  # 创建文本字符串
    
    cv2.putText(combined_img, text, text_position, font, font_scale, font_color, font_thickness)
    return combined_img, len(results[0].boxes), masked_img

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='Process a video.')
    parser.add_argument('--video_dir', help='Path to the directory containing video files.')
    pts = np.array([(535, 494), (532, 777), (573, 1005), (662, 1272), 
                    (721, 1392), (826, 1553), (1005, 1539), (1164, 1498), (1288, 1443),(1364, 1400),
                    (1373, 1226),(1377, 1011),(1333,806),(1297, 707),(1219,544),(1169,461),(1091,364),(802,395)], np.int32)    
    # pts = np.array([(518.4, 415.2), (535.2, 1012.8), (662.4, 1392.0), (780.0, 1579.2), 
    #                 (1123.2, 1538.4), (1404.0, 1396.8), (1365.6, 813.6), (1118.4, 290.4), (518.4, 415.2)], np.int32)
    args = parser.parse_args()
    # model = YOLO('/home/deepl/ultralytics/1011weights/weights/best.pt')
    video_dir = args.video_dir
    sub_dirs = [os.path.join(video_dir, o) for o in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir,o))]
    video_files = glob.glob(os.path.join(video_dir, '*.flv'))
    pig_counts = []
    # for sub_dir in sub_dirs:
    #     print("sub_dir: ", sub_dir)
    #     video_files = glob.glob(os.path.join(sub_dir, '*.flv'))
    #     max_pigs_per_video = {}

    result_path = os.path.join(video_dir.replace(video_dir.split("/")[-1], ""), video_dir.split("/")[-1] + "_images")
    print("result_path: ", result_path)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    for video_path in video_files:
        print("video_path: ", video_path)
        prefix = os.path.splitext(os.path.basename(video_path))[0]
        cap = cv2.VideoCapture(video_path)
        relative_name = video_path.split("/")[-1].split(".")[0]
        if not cap.isOpened():
            print(f"Error: Cannot open video file {video_path}")
            continue
        else:        
            fps = cap.get(cv2.CAP_PROP_FPS)
            print("FPS: ", fps)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # 计算视频中间时间点的帧索引
            middle_frame_index = total_frames // 2
            # 设置视频读取的当前帧索引
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
            ret, frame = cap.read()
            
            if ret:
                # 在这里，你可以选择保存该帧为图像文件，或者进行其他你想要的处理
                output_image_path = os.path.join(result_path, f"{relative_name}_frame_{middle_frame_index}.png")
                #output_image_path = result_path + f"{relative_name}_frame_{middle_frame_index}.png"
                result_image_path = os.path.join(result_path, f"{relative_name}_result_{middle_frame_index}.png")
                img, pig_count, masked_img = pigcount(frame, pts)
                pig_counts.append(pig_count)
                cv2.imwrite(output_image_path, frame)
                cv2.imwrite(result_image_path, img)
                print(f"Saved frame {middle_frame_index} of video {video_path} to {output_image_path}")
            else:
                print(f"Error: Failed to read frame {middle_frame_index} of video {video_path}")
                continue
            cap.release()
    
    counts, bins, patches = plt.hist(pig_counts, bins=range(min(pig_counts), max(pig_counts) + 1), edgecolor='black', align='left')
    plt.xlabel('Pig Count')
    plt.ylabel('Frequency')
    plt.title('Distribution of Pig Count')
    
    # 设置横坐标的刻度和标签
    plt.xticks(bins[:-1], bins[:-1])
    plt.show()  # 显示条形图
            # frame_count = 0
            # img_count = 0
            # pig_count_dict = defaultdict(int)
            # while True:
            #     ret, frame = cap.read()
            #     if ret:
            #         if frame_count % int(fps*1) == 0:
            #             result_pic, number_pig, masked_img = pigcount(model, frame, pts)
            #             pig_count_dict[number_pig] += 1
            #             img_count += 1
            #         frame_count += 1
            #     else:
            #         break

            # max_count = max(pig_count_dict.values())
            # max_pig = [k for k, v in pig_count_dict.items() if v == max_count][0]
            # total_counts = sum(pig_count_dict.values())
            # max_proportion = max_count / total_counts * 100
            # max_pigs_per_video[prefix] = (max_pig, max_proportion)
            # print(f"Most common pig count: {max_pig}, Proportion: {max_proportion}%" , "Video: ", video_path)
            # cap.release()

        # sorted_max_pigs_per_video = {
        #     k: v
        #     for k, v in sorted(
        #         max_pigs_per_video.items(),
        #         key=lambda item: (item[0].split('_')[0], int(item[0].split('_')[1][:-2]), item[0].split('_')[1][-2:])
        #     )
        # }

        # plt.figure(figsize=(10, 5))
        # bars = plt.bar([dt for dt in sorted_max_pigs_per_video.keys()], [v[0] for v in sorted_max_pigs_per_video.values()])
        # plt.xlabel('Video Name')
        # plt.ylabel('Max Pig Count')
        # plt.title('Max Possible Pig Count per Video')
        # plt.xticks(rotation=45)
        # for bar, key in zip(bars, sorted_max_pigs_per_video.keys()):
        #     yval = bar.get_height()
        #     plt.text(bar.get_x() + bar.get_width()/2, yval - 5, f'{sorted_max_pigs_per_video[list(sorted_max_pigs_per_video.keys())[bars.index(bar)]][1]:.2f}%', ha='center', va='bottom')
        #     plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f'{sorted_max_pigs_per_video[list(sorted_max_pigs_per_video.keys())[bars.index(bar)]][0]}', ha='center', va='bottom')
        # # for bar in bars:
        # #     yval = bar.get_height()
        # #     plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f'{sorted_max_pigs_per_video[list(sorted_max_pigs_per_video.keys())[bars.index(bar)]][1]:.2f}%', ha='center', va='bottom')
        # plt.tight_layout()
        # plt.savefig(f'{os.path.basename(sub_dir)}.png')
        # plt.close()
        

