from PIL import Image
from fnmatch import fnmatch
from queue import Queue
import cv2
import time
import os
import numpy as np
import pathlib
from pathlib import Path

def process_image(img:np.ndarray,img_name:str):
    """初步处理图像，将图像转化为二值图，进行膨胀、腐蚀，然后寻找边界，进行分割。返回分割的图像列表

    Args:
        img (np.ndarray): 传入的图像
    Returns:
        list: 图像列表
        list: 每个字符名字
    """
    # 将RGB图像转换为灰度图像
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 处理上下四个像素行，防止后面提取到空白轮廓
    gray[:4,:]=255
    gray[-4:,:]=255
    # 对灰度图像进行高斯滤波
    blur = cv2.GaussianBlur(gray,(5,5),0)
    # 将图像转换为二值图
    ret, thresh = cv2.threshold(blur,0,255,cv2.THRESH_OTSU)
    kernel = np.ones((4,4),np.uint8)
    # 对图像进行膨胀操作
    dilation = cv2.dilate(thresh,kernel,iterations = 1)
    # 对图像进行腐蚀操作
    erosion = cv2.erode(dilation,kernel,iterations = 1)
    # 寻找图像中的边界
    contours,hierarchy = cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # 根据contours分割图像
    chars = []
    bound_list = []
    for cnt in contours:
        #计算矩形边框
        x,y,w,h = cv2.boundingRect(cnt)
        if h >= 30 and w <= 100: 
            flag = True
            for x_,y_,w_,h_ in bound_list:
                if(y in range(y_,y_+h_) and 
                x in range(x_,x_+w_) and 
                y+h in range(y_,y_+h_) and 
                x+w in range(x_,x_+w_)):
                    flag = False
            if flag:
                bound_list.append([x,y,w,h])
    # 根据x对图像排个序
    bound_list = sorted(bound_list,key=lambda x:x[0])
    for x,y,w,h in bound_list:
        if h >= 30 and w <= 100:      
            # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        #提取每个字母区域
            char_img = img[y:y+h, x:x+w]
            chars.append(char_img)
    char_name_list = [i for i in img_name]
    return chars,char_name_list



def get_all_chars(doc_path,save_path):
    D = Path(doc_path)
    img_path_list = [] # 验证码地址
    img_name_list = [] # 整个验证码名称
    char_name_all = [] # 分割后的字符的值
    chars_all = [] # 分割后的字符
    for i in D.iterdir():
        # 将图片地址和图片名称对应
        img_path_list.append(str(i))
        img_name_list.append(i.name.split('.')[0])
    for index,img_path in enumerate(img_path_list):
        img = cv2.imread(img_path)
        img_name = img_name_list[index]
        chars,char_name_list = process_image(img,img_name)
        if len(chars) == 4: # 如果分割出四个字符
            chars_all += chars
            char_name_all += char_name_list
    # 打开保存单个字符的文档
    D_save = Path(save_path)
    # for index,char_name in enumerate(name_all):
    #     file_name = str(index)+'.jpg'
    #     file_path = str(D_save/file_name)
    #     cv2.imwrite(file_path,res[index])
    for index,img in enumerate(chars_all):
        file_name = str(index)+'_'+char_name_all[index]+'.jpg'
        file_path = str(D_save/file_name)
        cv2.imwrite(file_path,img)
    
if __name__ == '__main__':
    get_all_chars('VertifyCode/data1/','data/code_split/images')
