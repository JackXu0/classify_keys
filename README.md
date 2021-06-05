# 钥匙分类器

#### 准备

##### 安装依赖 (Python3)

```bash
pip install torch
pip install torchvision
pip install Pillow
pip install opencv-python
```

##### 更改配置

- 在config.py文件中修改钥匙的类别数

![image-20210604144148252](/Users/zhuocheng/Library/Application Support/typora-user-images/image-20210604144148252.png)

#### 训练

- 将视频放到dataset/training_videos目录下。视频名 = “类别ID.mp4”. 类别ID从0开始，每次递增1.
- 运行train.py
- 训练结束，模型参数被存储在checkpoints文件夹下

#### 测试

- 将视频放到dataset/testing_videos目录下。视频名 = “类别ID.mp4”. 类别ID从0开始，每次递增1.

- 在test.py中，更改checkpoint路径, 一般取最新的checkpoint

  - ```python
    model.load_state_dict(torch.load("checkpoints/8.pth.tar"))
    ```

- 运行test.py

#### 预测

- 将视频放到dataset/prediction_videos目录下。视频名随意。

- 在prediction.py中，修改视频路径

  - ```python
    video_url = "dataset/prediction_videos/random.mp4"
    ```

- 在prediction.py中，更改checkpoint路径, 一般取最新的checkpoint

  - ```python
    model.load_state_dict(torch.load("checkpoints/8.pth.tar"))
    ```

- 运行prediction.py。