import cv2

# 定义鼠标点击事件的回调函数
def get_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'Coordinates: ({x}, {y})')

# 读取图像
image_path = 'smallpig2/frame_0104.png'
image = cv2.imread(image_path)

# 创建一个名为 'image_window' 的窗口
cv2.namedWindow('image_window')

# 将鼠标回调函数绑定到 'image_window' 窗口
cv2.setMouseCallback('image_window', get_coordinates)

# 显示图像，直到用户按下任意键关闭窗口
cv2.imshow('image_window', image)
cv2.waitKey(0)

# 关闭所有 OpenCV 窗口
cv2.destroyAllWindows()