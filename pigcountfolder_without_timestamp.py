from ultralytics import YOLO
import cv2
import numpy as np
from tqdm import tqdm
import shutil

def concatenate_images(img, res_img):
    h, w, _ = img.shape
    result = np.zeros((h, w*2, 3), dtype=np.uint8)
    result[:, :w] = img
    result[:, w:] = res_img
    return result

def pigcount(img, area):
    
    #model = YOLO('/home/deepl/ultralytics/runs/segment/train7/weights/best.pt')  # load a custom model
    #model = YOLO('/home/deepl/ultralytics/checkpoint/train0002/weights/best.pt')
    #model = YOLO('/home/deepl/ultralytics/521_checkpoint/train0006/weights/best.pt')
    #model = YOLO('/home/deepl/ultralytics_926revise/checkpoint/82/weights/best.pt')
    model = YOLO('/home/deepl/ultralytics/smallpig_weights/1018_6:30/weights/best.pt')
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

    font = cv2.FONT_HERSHEY_SIMPLEX  # 选择字体
    font_scale = 3  # 字体大小
    font_color = (255, 255, 255)  # 字体颜色，例如白色
    font_thickness = 2  # 字体厚度
    text_position = (10, 300)  # 文本位置（距离左上角的像素）
    text = f'Pig Count: {len(results[0].boxes)}'  # 创建文本字符串
    cv2.putText(res_plotted, text, text_position, font, font_scale, font_color, font_thickness)
    return res_plotted, len(results[0].boxes), masked_img
    #return masked_img, len(results[0].boxes)

if __name__ == "__main__":
    import os
    import argparse
    parser = argparse.ArgumentParser(description='Process a folder.')
    parser.add_argument('--input_folder', help='Path to the video file.')
    args = parser.parse_args()
    prefix = os.path.splitext(os.path.basename(args.input_folder))[0]+"_imageoutput"
    print(prefix)
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    pts = np.array([(535, 494), (532, 777), (573, 1005), (662, 1272), 
                    (721, 1392), (826, 1553), (1005, 1539), (1164, 1498), (1288, 1443),(1364, 1400),
                    (1373, 1226),(1377, 1011),(1333,806),(1297, 707),(1219,544),(1169,461),(1091,364),(802,395)], np.int32)
    print(args.input_folder)
    for i in tqdm(os.listdir(args.input_folder)):
        full_path = os.path.join(args.input_folder, i)
        tmp_frame = cv2.imread(full_path)
        img, pig_count, masked_img = pigcount(tmp_frame, pts)    
        print("pig_count: ", pig_count)
        if pig_count != 0:
            combined_img = concatenate_images(tmp_frame, img)
            cv2.imwrite("{}/{}_result.{}".format(prefix, i.split(".")[0], i.split(".")[1]), combined_img)
            #cv2.imwrite("output/{}".format(i), masked_img)
            #shutil.copy(full_path, "{}/{}".format(prefix, i))
            
            

