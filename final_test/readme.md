文件结构
.
|____pretrained_model
| |____model_final.pth # 训练好的识别模型
|____src
| |____test.py  # 对整张验证码进行分割和识别，计算精度
| |____create_code.py #生成验证码数据集