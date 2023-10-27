import numpy as np
import os
import glob

folder_path = '1009hard/*'

# 使用glob()函数遍历文件夹中的所有文件
count = 1
for filename in glob.glob(folder_path):
    # 获取文件的基本名和扩展名
    base_name = os.path.basename(filename)
    file_extension = os.path.splitext(base_name)[1]
    front_name = os.path.splitext(base_name)[0]
    print(front_name)
    
    new_base_name = base_name.replace(front_name, "1009hard" + "_" + str(count))
    new_filename = os.path.join(os.path.dirname(filename), new_base_name)

    # 重命名文件
    os.rename(filename, new_filename)
    print(f'{filename} has been renamed to {new_filename}')
    count += 1