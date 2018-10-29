1、mnist_inference.py 
定义前向传播相关子函数

2、mnist_train.py
定义训练及相关优化方法
可直接运行训练模型并保存到同级文件夹model/下

3、NumReco.py
定义查找本地模型，识别数字的子函数

4、SimpeleUI.py 
界面程序。可在界面调用NumReco.py中的识别函数
可直接运行（在mnist_train.py训练出模型以后）