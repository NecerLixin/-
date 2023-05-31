from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
from pathlib import Path
from torchvision.transforms import transforms
index = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 
         "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", 
         "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", 
         "u", "v", "w", "x", "y", "z", "A", "B", "C", "D", 
         "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", 
         "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", 
         "Y", "Z"]
root = Tk()
root.title('Image Viewer')
win_width = root.winfo_screenwidth()  
win_height = root.winfo_screenheight()
width = 500
height = 500
root.resizable(width, height)
# 居中
root.geometry('%dx%d+%d+%d'%(width,height,(win_width-width)/2,(win_height-height)/2))
image = 0

def image_split():
    """初步处理图像，将图像转化为二值图，进行膨胀、腐蚀，然后寻找边界，进行分割。返回分割的图像列表

    Args:
        img (np.ndarray): 传入的图像
    Returns:
        list: 图像列表
        list: 每个字符名字
    """
    global image
    img = np.array(image)
    # img_name = image_path.split('/')[-1].split('.')[0]
    # 将RGB图像转换为灰度图像
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # 处理上下四个像素行，防止后面提取到空白轮廓
    gray[:4,:]=255
    gray[-4:,:]=255
    # 对灰度图像进行高斯滤波
    blur = cv2.GaussianBlur(gray,(7,7),0)
    # 将图像转换为二值图
    ret, thresh = cv2.threshold(blur,240,255,cv2.THRESH_OTSU)
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
    # char_name_list = [i for i in img_name]
    # chars = [Image.fromarray(i) for i in chars]
    global bbb
    return chars
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.pool2 = nn.MaxPool2d(2, 2) 
        self.fc1 = nn.Linear(576, 300)  
        self.fc2 = nn.Linear(300, 120)  
        self.fc3 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, 62)
        self.flatten = nn.Flatten()
        self.bc1 = nn.BatchNorm2d(6)
        self.bc2 = nn.BatchNorm2d(16)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.bc1(x)
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.bc2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x)) 
        x = F.dropout(x,0.3)
        x = F.relu(self.fc2(x))
        x = F.dropout(x,0.3)
        x = self.fc3(x)
        x = F.relu(x)
        x = F.dropout(x,0.3)
        x = self.fc4(x)
        x = F.log_softmax(x)
        return x
def open_image(): 
    global image
    input_image = filedialog.askopenfilename(title='选择图片')
    image = Image.open(input_image)
    image = image.resize((160, 60))
    img = ImageTk.PhotoImage(image)
    image_label.config(image=img)
    image_label.image = img 
def split_image():
    temp_file = 'temp/'
    chars = image_split()
    messagebox.showinfo("提示", "字符分割完成")
    for i,char in enumerate(chars):
        cv2.imwrite(temp_file+str(i+1)+'.jpg',char)

def split_image_show():
    global frame
    img1 = ImageTk.PhotoImage(Image.open("temp/1.jpg"))  
    img2 = ImageTk.PhotoImage(Image.open("temp/2.jpg"))  
    img3 = ImageTk.PhotoImage(Image.open("temp/3.jpg"))  
    img4 = ImageTk.PhotoImage(Image.open("temp/4.jpg"))  
    label1.config(image=img1)
    label2.config(image=img2)
    label3.config(image=img3)
    label4.config(image=img4)
    label1.image = img1
    label2.image = img2
    label3.image = img3
    label4.image = img4

def pred():
    model = torch.load('model_final.pth',map_location='cpu')
    model.eval()
    file_path = Path('temp/')
    trans = transforms.Compose([transforms.Resize([30,30]),
                                transforms.ToTensor()])
    pred_res = []
    image_path_list = []
    for i in file_path.iterdir():
        img_path = str(file_path/i.name)
        image_path_list.append(img_path)
        image_path_list = sorted(image_path_list,key=lambda x:int(x.split('/')[-1].split('.')[0]))
    for image_path in image_path_list:
        image = Image.open(image_path)
        X = trans(image)
        X = X.unsqueeze(0)
        output = model(X)
        _,y_pred = torch.max(output.data,1)
        pred_res.append(y_pred.item())
    pred_res = [index[i] for i in pred_res]
    res = " ".join(pred_res)
    messagebox.showinfo('识别结果',res)
    
image_label = Label(root)
image_label.pack()

frame = Frame(root)  
frame.pack()
label1 = Label(frame) 
label2 = Label(frame)  
label3 = Label(frame)  
label4 = Label(frame)
label1.grid(row=0, column=0)    
label2.grid(row=0, column=1)    
label3.grid(row=1, column=0)
label4.grid(row=1, column=1)
# 打开图片按钮
open_button = Button(root, text='选择验证码', command=open_image)
open_button.pack()
# 字符分割按钮
split_button = Button(root,text='字符分割',command=split_image)
split_button.pack()
# 分割字符显示
chars_show_button = Button(root,text='显示分割字符',command=split_image_show)
chars_show_button.pack()
# 字符识别按钮
char_recognize_button = Button(root,text='字符识别',command=pred)
char_recognize_button.pack()

root.mainloop()