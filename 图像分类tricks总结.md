计算机视觉主要问题有图像分类、目标检测和图像分割等。针对图像分类任务，提升准确率的方法路线有两条，一个是模型的修改，另一个是各种数据处理和训练的技巧(tricks)。本文在精读论文的基础上，总结了图像分类任务的各种tricks。

# Warm up
学习率是神经网络训练中最重要的超参数之一，针对学习率的技巧有很多。Warm up是在ResNet论文[1]中提到的一种学习率预热的方法。由于刚开始训练时模型的权重(weights)是随机初始化的(全部置为0是一个坑，原因见[3])，此时选择一个较大的学习率，可能会带来模型的不稳定。学习率预热就是在刚开始训练的时候先使用一个较小的学习率，训练一些epoches或iterations，等模型稳定时再修改为预先设置的学习率进行训练。论文[1]中使用一个110层的ResNet在cifar10上训练时，先用0.01的学习率训练直到训练误差低于80%(大概训练了400个iterations)，然后使用0.1的学习率进行训练。

上述的方法是constant warmup，18年Facebook又针对上面的warmup进行了改进[2]，因为从一个很小的学习率一下变为比较大的学习率可能会导致训练误差突然增大。论文[2]提出了gradual warmup来解决这个问题，即从最开始的小学习率开始，每个iteration增大一点，直到最初设置的比较大的学习率。

# Label smoothing

# Random image cropping and patching (RICAP)

RICAP方法随机裁剪四个图片的中部分，然后把它们拼接为一个图片，同时混合这四个图片的标签。

RICAP在caifar10上达到了2.19%的错误率。
![](http://ww1.sinaimg.cn/large/e323d644ly1g0oqid20bgj20ee0aotcq.jpg)

如下图所示，I_x, I_y是原始图片的宽和高。w和h称为boundary position，它决定了四个裁剪得到的小图片的尺寸。w和h从beta分布Beta(β, β)中随机生成，β也是RICAP的超参数。最终拼接的图片尺寸和原图片尺寸保持一致。

![](http://ww1.sinaimg.cn/large/e323d644ly1g0oqom5lj0j20ec0a9tc2.jpg)

![](http://ww1.sinaimg.cn/large/e323d644ly1g0oqvuqfidj20nr0jmq7i.jpg)


RICAP的代码如下：

```python
beta = 0.3 # hyperparameter
for (images, targets) in loader:
    
    # get the image size
    I_x, I_y = images.size()[2:]

    # draw a boundry position (w, h)
    w = int(np.round(I_x * np.random.beta(beta, beta)))
    h = int(np.round(I_y * np.random.beta(beta, beta)))
    w_ = [w, I_x - w, w, I_x - w]
    h_ = [h, h, I_y - h, I_y - h]

    # select and crop four images
    cropped_images = {}
    c_ = {}
    W_ = {}
    for k in range(4):
        index = torch.randperm(images.size(0))
        x_k = np.random.randint(0, I_x - w_[k] + 1)
        y_k = np.random.randint(0, I_y - h_[k] + 1)
        cropped_images[k] = images[index][:, :, x_k:x_k + w_[k], y_k:y_k + h_[k]]
        c_[k] = target[index].cuda()
        W_[k] = w_[k] * h_[k] / (I_x * I_y)
    
    # patch cropped images
    patched_images = torch.cat(
                (torch.cat((cropped_images[0], cropped_images[1]), 2),
                torch.cat((cropped_images[2], cropped_images[3]), 2)),
            3)
    #patched_images = patched_images.cuda()
    
    # get output
    output = model(patched_images)
            
    # calculate loss and accuracy
    loss = sum([W_[k] * criterion(output, c_[k]) for k in range(4)])
    acc = sum([W_[k] * accuracy(output, c_[k])[0] for k in range(4)])
```
# 知识蒸馏

# Random erasing

# Cutout
Cutout是一种新的正则化方法。原理是在训练时随机把图片的一部分减掉，这样能提高模型的鲁棒性。这是一种很像data augmentation和dropout的方法，它的来源是计算机视觉任务中经常遇到的物体遮挡问题。通过cutout生成一些类似被遮挡的物体，不仅可以让模型在遇到遮挡问题时表现更好，还能让模型在做决定时更多地考虑环境(context)。

代码如下：
```python
import torch
import numpy as np

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
```

效果如下图，每个图片的一小部分被cutout了。

![](http://ww1.sinaimg.cn/large/e323d644ly1g0ol7f29gpj207m09iaev.jpg)

# Cosine learning rate decay
# Mixup training

Mixup training，就是每次取出2张图片，然后将它们线性组合，得到新的图片，以此来作为新的训练样本，进行网络的训练，如下公式，其中x代表图像数据，y代表标签，则得到的新的x_hat, y_hat。

![](http://ww1.sinaimg.cn/large/e323d644ly1g0osd8xdjzj208p00kjr5.jpg)

mixup方法主要增强了训练样本之间的线性表达，增强网络的泛化能力，并且使用mixup方法需要较长的时间收敛。

mixup代码如下：

```python
l = np.random.beta(mixup_lambda, mixup_lambda)

index = torch.randperm(input.size(0))
input_a, input_b = input, input[index]
target_a, target_b = target, target[index]

mixed_input = l * input_a + (1 - l) * input_b

output = model(mixed_input)
loss = l * criterion(output, target_a) + (1 - l) * criterion(output, target_b)
acc = l * accuracy(output, target_a)[0] + (1 - l) * accuracy(output, target_b)[0]
```

# AdaBound
# 其他经典的tricks
## 常用的正则化方法为
- dropout
- L1/L2正则
- Batch Normalization
- shakeshake
- shake drop
- 


## 传统的数据增强方法
- 随机裁剪
- 水平翻转
- 添加PCA噪声

其他：
- mixup
- auto data augomentation


# 参考
[1] Deep Residual Learning for Image Recognition(https://arxiv.org/pdf/1512.03385.pdf)

[2] New warm up(https://arxiv.org/pdf/1706.02677v2.pdf)

[3] http://cs231n.github.io/neural-networks-2/

[3]Bag of Tricks for Image Classification with Convolutional Neural Networks(https://arxiv.org/pdf/1812.01187.pdf)
