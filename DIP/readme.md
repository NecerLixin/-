文件结构：
.
|____train_test_loss.jpg #训练和测试的损失，看是否过拟合
|____model
| |____model_final.pth #保存训练的模型
|____src
| |____create_code.py # 生成验证码
| |____code_split.py # 验证码字符分割
| |____main.py # 模型训练
| |____teacherModel.py # 一个尝试，先分类数字和字母，然后将output作为第四个通道，事实证明没有任何作用