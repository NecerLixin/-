import numpy as np
import random
import sys
from PIL import Image
from captcha.image import ImageCaptcha

num = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
       'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
       'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

#生成验证码的随机数字列表，验证码图片中有4个数字
def random_text(charset=num, size=4):
    caplist = []
    for i in range(size):
        captcha_text = random.choice(charset)
        caplist.append(captcha_text)
    return caplist

#生成验证码
def create_captext(file_path):
    caplist = random_text()
    caplist = ''.join(caplist)#将验证码列表转为字符串
    image = ImageCaptcha()
    captcha = image.generate(caplist)#生成图片
    image.write(caplist,file_path+caplist+'.jpg' )#保存图片，图片名即为里面的数字

#手动实现进度条   
# for i in range(100):
#     create_captext()
#     sys.stdout.write("\r start %d/%d" %(i+1, 100))
#     sys.stdout.flush()
# sys.stdout.write("\n")
# sys.stdout.flush()

#自动进度条
from tqdm import tqdm
import time
for i in tqdm(range(1000)):
    # create_captext('VertifyCode/data1/')
    create_captext('/Users/lijinliang/Desktop/人工智能/数字图像处理/大作业final/final_test/test_data/image/')
    time.sleep(0.01)