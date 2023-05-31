import torch
import torchvision
import json
from pathlib import Path
from torch.utils import data
import cv2
from torch import nn
import torch.nn.functional as F
from torch import optim
from torchvision import transforms
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_counter = []
train_loss = []
test_loss = []

def get_images_and_labels_path(pic_path:str,label_path:str,index_path:str):
    # 读取所有图片的文件名
    D = Path(pic_path)
    name_path_list = []
    with open(index_path,'r') as f:
        index = json.load(f)
    for c in D.iterdir():
        name_path_list.append(c.name)
        name_path_list = sorted(name_path_list,key=lambda x:int(x.split('.')[0]))
    img_path_list = [str(D/name) for name in name_path_list]
    # 获得图片名和label的映射
    with open(label_path,'r') as f:
        labels:dict = json.load(f)
        labels = list(labels.values())
    labels = [float(index[char]) for char in labels]
    return img_path_list,labels
def get_images_and_labels(imgs_path:str,index_path:str):
    D = Path(imgs_path)
    with open(index_path,'r') as f:
        index = json.load(f)
    imgs_list = []
    label_list = []
    for i in D.iterdir():
        imgs_list.append(str(D / i.name))
        val = i.name.split('.')[0].split('_')[-1]
        numb = index[val]
        label_list.append(numb)
    return imgs_list,label_list
    
    
class MyDataSet(data.Dataset):
    def __init__(self,pic_path,index_path) -> None:
        self.pic_path = pic_path
        self.transformer = transforms.Compose([
    transforms.Resize((30, 30)),
    transforms.ToTensor()
])
        self.img_path_list, self.lable_list = get_images_and_labels(pic_path,index_path)

        
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        name = img_path.split('/')[-1]
        # img = cv2.imread(img_path)
        img = Image.open(img_path)
        label = int(self.lable_list[index])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # ret, img = cv2.threshold(img,0,255,cv2.THRESH_OTSU)
        img = self.transformer(img)
        # img = torch.from_numpy(img)
        # label = F.one_hot(torch.tensor(label))
        label = torch.tensor(label)
        sample = dict()
        sample['image'] = img
        sample['label'] = label
        sample['image'].to(device)
        sample['label'].to(device)
        return sample
    def __len__(self):
        return len(self.img_path_list)

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

class NetWithT(nn.Module):
    def __init__(self,teacher_model:nn.Module) -> None:
        super(NetWithT,self).__init__()
        self.teacher_model = teacher_model
        self.conv1 = nn.Conv2d(4, 6, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.pool2 = nn.MaxPool2d(2, 2) 
        self.fc1 = nn.Linear(576, 120)  
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 62)
        self.flatten = nn.Flatten()
        self.bc1 = nn.BatchNorm2d(6)
        self.bc2 = nn.BatchNorm2d(16)
        self.supervise_layer = torch.ones([30,30])
        self.supervise_layer.to(device)
        
    def forward(self,x):
        # [30,30] -> [1,30,30]
        layer = self.supervise_layer.unsqueeze(0).to(device)
        # [1,30,30] -> [batch,30,30]
        layer = torch.repeat_interleave(layer,x.shape[0],0)
        # [batch,1] -> [batch,1,1]
        t = self.teacher_model(x).to(device)
        t = t.unsqueeze(-1)
        layer*=t
        # [batch,30,30] -> [batch,1,30,30]
        layer = layer.unsqueeze(1)
        x = torch.cat([x,layer],dim=1)
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
        x = F.log_softmax(x)
        return x
class teacherNet(nn.Module):
    def __init__(self):
        super(teacherNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.pool2 = nn.MaxPool2d(2, 2) 
        self.fc1 = nn.Linear(576, 120)  
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
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
        # x = F.log_softmax(x)
        x = F.sigmoid(x)
        return x.to(device)
        
def train(model:Net,criteon,optimizer,epochs,train_set,test_set):
    model.train()
    c = 1
    for epoch in range(epochs):
        print(f"epoch:{epoch}------------------------")
        acc_list = []
        loss_list = []
        for batch,sample in enumerate(train_set):
            X = sample['image'].to(device)
            y = sample['label'].to(device)
            # print(f'y:\n{y}')
            optimizer.zero_grad()
            output = model(X)
            loss = criteon(output,y)
            loss.backward()
            optimizer.step()
            acc = acc_score(output,y)
            acc_list.append(acc)
            loss_list.append(loss.item())
            # if batch % 100 == 0:
            #     print('train_acc:',np.array(acc_list).mean())
        train_loss.append(np.array(loss_list).mean())
        train_counter.append(c)
        c+=1
        test_eval(model,test_set,criteon)
        print('loss:',loss.item())
        print('train_acc:',np.array(acc_list).mean())
            
def acc_score(output:torch.Tensor,y:torch.Tensor):
    _,y_pred = torch.max(output.data,1)
    correct = (y_pred == y).sum()
    acc = correct/y.size(0)
    return acc.item()



def test_eval(model,test_data,criteon):
    model.eval()
    acc_list = []
    loss_list = []
    for batch,sample in enumerate(test_data):
        x = sample['image'].to(device)
        y = sample['label'].to(device)
        output = model(x)
        loss = criteon(output,y)
        loss_list.append(loss.item())
        acc = acc_score(output,y)
        acc_list.append(acc)
    res_acc = np.array(acc_list).mean()
    eval_loss = np.array(loss_list).mean()
    test_loss.append(eval_loss)
    print("eval_acc:",res_acc)
    print('eval_loss:',eval_loss)
    
    

if __name__ == '__main__':
    batch_size = 80
    Mydata = MyDataSet('data/code_split/images','data/code_split/index/index.json')
    train_set,test_set = data.random_split(Mydata,[0.8,0.2])
    img_data_set = data.DataLoader(Mydata,batch_size=1,shuffle=True,drop_last=False)
    train_data  = data.DataLoader(train_set,batch_size=batch_size,shuffle=True)
    test_data = data.DataLoader(test_set,batch_size=batch_size,shuffle=True)
    model = Net()
    # teacherModel = torch.load('model/teacher_model_sigmoid_95.pth')
    # teacherModel.to(device)
    # model = NetWithT(teacher_model=teacherModel)
    model.to(device)
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    # criteon = nn.NLLLoss()
    criteon = nn.NLLLoss()
    # criteon = nn.MSELoss()
    train(model,criteon,optimizer,50,train_data,test_data)
    torch.save(model,'model/model_final.pth')
    # for batch,sample in enumerate(train_data):
    #     data = sample['image']
    #     label = sample['label']
    #     name = sample['name']
    #     print(name,label)
    #     if batch == 20:
    #         break
    fig = plt.figure(dpi=200)
    plt.plot(train_counter, train_loss, color='blue',label='trainning loss')
    plt.plot(train_counter, test_loss, color='red',label='test loss')
    plt.legend()
    plt.savefig('temp_final.jpg')
    # for batch,sample in enumerate(img_data_set):
    #     print(sample['image'].shape,sample['label'])
    #     break


    
    
