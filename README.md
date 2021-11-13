# 2020-cs231n

借此平台保留参考代码，希望以后有机会能跟别人分享讨论！

**`NOTE`** 考虑到这类仓库对github生态&大家学习的弊大于利，过段时间仓库应该会404啦！希望仓库内容曾对大家有积极帮助！

<br />

<br />

## 1. `./cs231n` （Stanford深度学习与计算机视觉基础课程）

- 前言：

  - 主要包含自己本地环境下做的2020年作业（也就是没用上GPU...orz），PyTorch和Tensorflow可选作业中，我选了据说更复杂的Tensorflow（仅出于学习目的）；

  - 基本完成3个assignment（剩下assignment2的`TensorFlow.ipynb`的Part-V）。每个notebook中应该都附带了本地的运行结果，如果发现任何缺漏、疑问或错误希望能指出（可以发issue！），非常感谢！

  - 第一次完成时，代码和注释贪图了**dirty and quick**（orz），后期考虑美化；

- **关于`assignment1`：**

  - 由于我对numpy不太熟悉，对向量化求导过程不太熟练，里边的代码虽然跑通了，但优雅程度（包括注释的可读性）肯定不好，待美化；

- **关于`assignment2`：**

  - 我觉得主要的**亮点**在自己的**卷积速度**以及**卷积梯度反向传导**。**卷积速度**媲美`fast_layer.py`中的实现（还没看代码，估计在GPU上就不行了orz）。**卷积梯度反向传导**综合考虑了步长`stride >= 1 `的情况，应该相比一些网上资料甚至一些教材中说的“对边缘进行0填充，旋转180°，作卷积”要更具体严谨；
  - 但卷积梯度反向传导和池化层的速度依旧比`fast_layer.py`中的实现要逊色，而且我的池化层未实现向量化。待学习；

  - `im2col`相关拓展的编译在Windows上可能有坑，我也在notebook相应的cell中给出了解决方案：主要是要安装一个4G+的VS Build Tools （作业对Windows开发环境似乎不太友好，很多获取数据的脚本都是`*.sh`）；

- **关于`assignment3`：**

  - 之前让我闻之丧胆的RNN和LSTM，实现起来感觉居然是最简单的。

    因为里面的计算主要是element-wise的乘积与简单的线性函数，所以BP过程比较简单（感觉相比Batch Normalization的BP友好不少）。但LSTM的Computation Graph有点意思（可以在其中瞥到ResNet的Gradient High-way），**详情参见`notes`**；

  - 授课人Justin Johnson（那位帅哥）还有一篇关于Real-time Style Transfer的论文，我感觉这想法挺有好玩：https://arxiv.org/pdf/1603.08155.pdf

  - `Generative_Adversarial_Networks_TF.ipynb`中计算目标损失有个**小技巧**：将生成器和判别器的输出视作得分的logits值，免去了耗时的softmax()计算。**我在相应的cell中也阐述了自己的看法，主要说明为什么这样做是等价的**，也在`gan_tf.py`的`generator_loss()`中给出**三种（等价的）由生成器输出计算损失的方法**；

- **关于`notes`**：

  - 主要包含三个作业中设计的一些比较有意思的Back Propagation的推导。通过整理也让我进一步感受到了Computation Graph的及Local Gradient的强大；

  - emmm，字确实是丑（憋喷了
  - （身边没有七彩笔，而且每次都要拍照扫描整成pdf上传有点麻烦……突然很想攒个pad）

- 最后：

  - 作业算是基本完工了，大概耗时15天。最重要的是，也算瞥到了深度学习与计算机视觉的门（感觉不算入门，毕竟还没实战应用orz），它们对我不再像以前可以掉包的“黑箱”一样神秘，应该可以尝试逐步跟进前沿技术和算法了。
  - 我也学到了不少向量化编程和模块化编程（比如model跟solver分开，甚至model中各个layer也分开，需要使用时就像做三明治一样好玩了！）的技巧，让我逐渐有了安排好代码结构及注释结构的意识（关于注释结构，之前一直不是很注意！要小心）。

- 



<br />

<br />

----



<div align="center">by Divsigma@github.com</div>

