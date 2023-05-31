import torch
import torchvision
import json
import numpy as np
from torch.utils import data
from PIL import Image
import cv2
from pathlib import Path
from torchvision.transforms import transforms
from torch import nn
from torch.nn import functional as F
bbb = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
def get_images(file_path:str):
    """获得文件夹中的所有图片

    Args:
        file_path (str): 图像所在的文件夹

    Returns:
        list: 图像列表
        list: 内容列表
    """
    D = Path(file_path)
    res_image = []
    res_name = []
    for i in D.iterdir():
        name = i.name
        img_path = str(D/name)
        temp_name = name.split('.')[0]
        res_name.append(temp_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        res_image.append(image)
    return res_image,res_name

def image_split(img:np.ndarray,img_name:str):
    """初步处理图像，将图像转化为二值图，进行膨胀、腐蚀，然后寻找边界，进行分割。返回分割的图像列表

    Args:
        img (np.ndarray): 传入的图像
    Returns:
        list: 图像列表
        list: 每个字符名字
    """
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
    char_name_list = [i for i in img_name]
    chars = [Image.fromarray(i) for i in chars]
    global bbb
    if len(chars)==4:
        bbb += 1
    return chars,char_name_list

def image_transformer(file_path:str,index):
    """将文件中验证码的每个字符转换成tensor，四个字符和各自的内容放到一个字典中

    Args:
        file_path (str): 文件位置

    Returns:
        list:返回一个列表，列表的元素是一个字典，字典中chars是完成trans的四个字符，labels中是各个字符的内容
    """
    list_image,list_name = get_images(file_path)
    transform = transforms.Compose([transforms.Resize([30,30]),
                                   transforms.ToTensor()])
    res = []
    for j,image in enumerate(list_image):
        temp_dict = dict()
        chars,char_name_list = image_split(image,list_name[j])
        chars = [transform(i) for i in chars]
        char_name_list = [index[i] for i in char_name_list]
        temp_dict['chars'] = chars
        temp_dict['labels'] = char_name_list
        res.append(temp_dict)
    return res

def get_index(index_path):
    with open(index_path,'r') as f:
        index = json.load(f)
    return index
def model_pred(model:nn.Module,chars):
    model.eval()
    res = []
    for X in chars:
        X = X.unsqueeze(0)
        output = model(X)
        _,y_pred = torch.max(output.data,1)
        res.append(y_pred)
    return res
def isRight(pred,labels):
    for index,p in enumerate(pred):
        if p.item() != labels[index]:
            return False
    else:
        return True
def test():
    model = torch.load('pretrained_model/model_final.pth',map_location='cpu')
    model.to(device)
    model.eval()
    index = get_index('test_data/index/index.json')
    data_set = image_transformer('test_data/image/',index)
    all_size = len(data_set)
    right = 0
    c = 0
    for sample in data_set:
        print(c)
        c += 1
        chars = sample['chars']
        labels = sample['labels']
        pred = model_pred(model,chars)
        if isRight(pred,labels):
            right += 1
    print('accuracy:',right/all_size)


if __name__ == '__main__':
    test()
    print(bbb)
    
        
        