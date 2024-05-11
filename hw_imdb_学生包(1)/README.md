# imdb lstm 分类实验

注，所有代码在如下版本中测试通过

```
Python=3.9
Pytorch=1.13.1+cu117
numpy=1.26.4
```

理论上其他的pytorch版本也能通过

## 作业① LSTM RNN GRU 对比试验

### 基础代码
example_imdb_lstm_torch.py

在这个代码当中，

Line 14-30 初始化训练设备

Line 42-57 定义训练集

Line 60-77 定义网络结构

Line80-89 定义超参，实例化训练集和网络

### 执行命令：

```
python  example_imdb_lstm_torch.py
```
训练5 epoch, 耗时约1min，训练分类精度为 0.931

### LSTM的参考结果如下：
```log
vocab_size:  20001
ImdbNet(
  (embedding): Embedding(20001, 64)
  (lstm): LSTM(64, 64)
  (linear1): Linear(in_features=64, out_features=64, bias=True)
  (act1): ReLU()
  (linear2): Linear(in_features=64, out_features=2, bias=True)
)
Train Epoch: 1 Loss: 0.592848    Acc: 0.665735
Test set: Average loss: 0.4720, Accuracy: 0.7789
Train Epoch: 2 Loss: 0.390458    Acc: 0.827177
Test set: Average loss: 0.3778, Accuracy: 0.8319
Train Epoch: 3 Loss: 0.297707    Acc: 0.877496
Test set: Average loss: 0.3528, Accuracy: 0.8449
Train Epoch: 4 Loss: 0.237297    Acc: 0.908047
Test set: Average loss: 0.3485, Accuracy: 0.8558
Train Epoch: 5 Loss: 0.185850    Acc: 0.931410
Test set: Average loss: 0.3699, Accuracy: 0.8523
```

### 你的任务

阅读Pytorch内置的[Recurrent Layers](https://pytorch.org/docs/stable/nn.html#recurrent-layers)的官方文档（包括LSTM，RNN，GRU），了解不同的Recurrent Layers的输入和输出结构，以及初始化参数的含义。请在实验报告当中任意挑选一种，简单介绍它的输入输出的格式、以及初始化参数的含义。（1分）

修改$ImdbNet$​ 中的self.lstm为上述三种内置的Layer（LSTM，RNN，GRU；其中原始代码中已经填充了LSTM），运行代码并在实验报告当中汇报结果，结果格式请参考上面的“**LSTM的参考结果**”（2分）

## 作业② 手写LSTM实验

该作业参考时间：

GPU 1分钟1个epoch

CPU 5分钟1个epoch

### 基础代码

lstm_manual_template.py

Line 56-66 是你需要实现的手写LSTM内容，包括LSTM类所属的\_\__init\_\__函数和_forward_函数

Line 69-92 是你需要实现网络推理和训练内容，仅需要完善_forward_函数

Line 97-136 训练代码，需要修改超参以完成实验内容

### 你的任务

在不使用nn.LSTM的情况下，从原理上实现LSTM。

你可以参考PPT或者Pytorch官方文档[LSTM — PyTorch 2.2 documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM)来完成这个任务。

训练后测试集准确率要求不低于80% ，你需要在实验报告当中汇报结果，结果的格式请参考上面的“**LSTM的参考结果**”（2分）

我们会检查代码实现的正确性（2分）

调整网络结构（例如网络隐藏层维度，1分）、损失函数（其它的损失函数可以参考Pytorch的官方文档[torch.nn#loss-functions](https://pytorch.org/docs/stable/nn.html#loss-functions)，1分）、训练流程（例如训练的超参数，epoch、batchsize等，1分），观察他们对训练效果的影响。

**注：80%的准确率仅要求最优结果。在调整网络结构、损失函数、训练流程当中，不要求达到80%准确率。**



## 最后，你需要提交的内容

### 一份实验报告：

内容包括

1、作业① 中，任选一种Recurrent Layers的简介（1分）

2、作业① 中，LSTM RNN GRU 对比实验的实验结果（2分）

3、作业② 中，超过80%实验结果的截图（2分）

4、作业②中，调整三个不同内容的结果截图（3分）



### 代码文件

内容包括整个作业包，其中必须包括手写LSTM的代码（正确实现，2分）。





