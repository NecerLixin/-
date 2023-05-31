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
        if label < 10:
            label = 0
        else:
            label = 1
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

class MyCNN(nn.Module):
    def __init__(self) -> None:
        super(MyCNN,self).__init__()
        # [batch,3,100,100] -> [batch,10,90,90]
        # [batch,10,80,80] -> [batch,10,40,40]
        # [batch,10,40,40] -> [batch,5,30,30]
        # [batch,5,30,30] -> [batch,5,15,15]
        self.net = nn.Sequential(nn.Conv2d(1,16,11,stride=1),nn.ReLU(),
                                 nn.MaxPool2d(2),nn.ReLU(),
                                 nn.Conv2d(16,32,11),nn.ReLU(),
                                 nn.MaxPool2d(2),nn.ReLU(),
                                 nn.Flatten(),
                                 nn.Linear(2688,1000),nn.Sigmoid(),nn.Dropout(0.3),
                                #  nn.Linear(1000,400),nn.Sigmoid(),nn.Dropout(0.3),
                                 nn.Linear(1000,1),nn.Sigmoid())
        self.log_softmax = nn.LogSoftmax()
    def forward(self,X):
        res = self.net(X)
        return res
class MyCNN2(nn.Module):
    def __init__(self) -> None:
        super(MyCNN2,self).__init__()
        # [batch,3,100,100] -> [batch,10,90,90]
        # [batch,10,80,80] -> [batch,10,40,40]
        # [batch,10,40,40] -> [batch,5,30,30]
        # [batch,5,30,30] -> [batch,5,15,15]
        self.net = nn.Sequential(nn.Conv2d(1,16,7,stride=1),nn.ReLU(),
                                 nn.MaxPool2d(2),nn.ReLU(),
                                 nn.Conv2d(16,32,7),nn.ReLU(),
                                 nn.MaxPool2d(2),nn.ReLU(),
                                 nn.Conv2d(32,32,5),nn.ReLU(),
                                 nn.MaxPool2d(2),nn.ReLU(),
                                 nn.Flatten(),
                                 nn.Linear(480,100),nn.Sigmoid(),nn.Dropout(0.3),
                                #  nn.Linear(1000,400),nn.Sigmoid(),nn.Dropout(0.3),
                                 nn.Linear(100,1),nn.Sigmoid())
        self.log_softmax = nn.LogSoftmax()
    def forward(self,X):
        res = self.net(X)
        return res
class teacherCNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(teacherCNN,self).__init__()
        self.conv1 = nn.Conv2d(1,10,11,stride=1)
        self.conv2 = nn.Conv2d(10,5,11)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(420,100)
        self.fc2 = nn.Linear(100,1)
        self.bn1 = nn.BatchNorm2d(10)
        self.bn2 = nn.BatchNorm2d(5)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.flatten(x)
        # x = F.dropout(x,0.3)
        x = F.dropout(self.fc1(x),0.3)
        x = F.relu(x)
        x = self.fc2(x)
        res = F.sigmoid(x)
        return res   
class teacherCNN2(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(teacherCNN2,self).__init__()
        self.net = nn.Sequential(nn.Conv2d(1,4,11),nn.ReLU(),
                                 nn.MaxPool2d(2),nn.BatchNorm2d(4),nn.Dropout(0.3),
                                 nn.Conv2d(4,8,7),nn.ReLU(),
                                 nn.BatchNorm2d(8),
                                 nn.MaxPool2d(2),
                                #  nn.Dropout(0.3),
                                #  nn.Conv2d(8,16,3),nn.ReLU(),
                                #  nn.MaxPool2d(3),
                                 nn.Flatten(),
                                #  nn.Linear(384,100),nn.ReLU(),nn.Dropout(0.3),
                                 nn.Linear(1008,50),nn.ReLU(),
                                #  nn.Dropout(0.3),
                                 nn.Linear(50,1),nn.Sigmoid()
                                 )
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(1,4,11)
        self.conv2 = nn.Conv2d(4,8,7)
        self.fc1 = nn.Linear(1008,50)
        self.fc2 = nn.Linear(50,1)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)
    def forward(self,x):
        # res = self.net(x)
        # return res
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2)
        x = F.dropout(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.sigmoid(x)
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
        return x
def train(model,criteon,optimizer,epochs,train_set,test_set):
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
            y = y.unsqueeze(-1).float()
            loss = criteon(output,y)
            loss.backward()
            optimizer.step()
            acc = acc_score2(output,y)
            acc_list.append(acc)
            loss_list.append(loss.item())
            # if batch % 100 == 0:
            #     print('train_acc:',np.array(acc_list).mean())
        train_loss.append(np.array(loss_list).mean())
        train_counter.append(c)
        c+=1
        print('loss:',loss.item())
        print('train_acc:',np.array(acc_list).mean())
        test_eval(model,test_set,criteon)
        
            
def acc_score(output:torch.Tensor,y:torch.Tensor):
    _,y_pred = torch.max(output.data,1)
    correct = (y_pred == y).sum()
    acc = correct/y.size(0)
    return acc.item()
def acc_score2(output:torch.tensor,y:torch.Tensor):
    y_pred = torch.round(output)
    acc = (y_pred==y).sum()/y_pred.shape[0]
    return acc.item()
def test_eval(model,test_data,criteon):
    model.eval()
    acc_list = []
    loss_list = []
    for batch,sample in enumerate(test_data):
        x = sample['image'].to(device)
        y = sample['label'].to(device)
        y = y.unsqueeze(-1).float()
        output = model(x)
        loss = criteon(output,y)
        loss_list.append(loss.item())
        acc = acc_score2(output,y)
        acc_list.append(acc)
    res_acc = np.array(acc_list).mean()
    eval_loss = np.array(loss_list).mean()
    test_loss.append(eval_loss)
    print('eval_loss:',eval_loss)
    print("eval_acc:",res_acc)
    
    
    

if __name__ == '__main__':
    Mydata = MyDataSet('data/code_split/images','data/code_split/index/index.json')
    train_set,test_set = data.random_split(Mydata,[0.8,0.2])
    img_data_set = data.DataLoader(Mydata,batch_size=1,shuffle=True,drop_last=False)
    train_data  = data.DataLoader(train_set,batch_size=80,shuffle=True)
    test_data = data.DataLoader(test_set,batch_size=80,shuffle=True)
    # model = MyCNN()
    model = teacherNet()
    model.to(device)
    optimizer = optim.Adam(model.parameters(),lr=0.0001)
    # optimizer = optim.SGD(model.parameters(),lr=0.001)
    # criteon = nn.NLLLoss()
    criteon = nn.BCELoss()
    # criteon = nn.MSELoss()
    train(model,criteon,optimizer,50,train_data,test_data)
    torch.save(model,'model/teacher_model.pth')
    # for batch,sample in enumerate(train_data):
    #     data = sample['image']
    #     label = sample['label']
    #     name = sample['name']
    #     print(name,label)
    #     if batch == 20:
    #         break
    fig = plt.figure()
    plt.plot(train_counter, train_loss, color='blue')
    plt.plot(train_counter, test_loss, color='red')
    plt.savefig('teacherModel2.jpg')

    
    
