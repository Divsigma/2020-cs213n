# Courses
主要记录一些公开课/网课的笔记及作业，借此平台保存一下，希望以后有机会能跟别人分享讨论！



- `cs231` （Stanford深度学习与计算机视觉课程）

  - 主要包含自己本地环境下做的2020年作业（也就是没用上GPU...orz）
  - 现阶段基本完成3个assignment（assignment2剩下`TensorFlow.ipynb`的Part-V，assignment3剩下GAN部分）。每个notebook中应该都附带了本地的运行结果，如果发现任何缺漏、疑问或错误希望能指出（可以发issue！），非常感谢！

  - 第一次完成是，代码和注释贪图了**dirty and quick**（orz），以后在整理computation graph、求导式子及向量化过程时会同步美化；

  - 关于`assignment1`：

    - 由于我对numpy使用不太熟悉，对向量化求导过程不太熟练，里边的代码虽然跑通了，但优雅程度（包括注释的可读性）肯定不好，待美化；

  - 关于`assignment2`：

    - 我觉得主要的**亮点**在自己的**卷积速度**以及**卷积梯度反向传导**。**卷积速度**媲美`fast_layer.py`中的实现（还没看代码，估计在GPU上就不行了orz）。**卷积梯度反向传导**综合考虑了步长`stride >= 1 `的情况，应该相比一些网上资料甚至一些教材中说的“对边缘进行0填充，旋转180°，作卷积”要更具体严谨；
    - 但卷积梯度反向传导和池化层的速度依旧比`fast_layer.py`中的实现要逊色，而且我的池化层未实现向量化。待学习；

    - `im2col`相关拓展的编译在Windows上可能有坑，我也在notebook相应的cell中给出了解决方案：主要是要安装一个4G+的VS Build Tools （作业对Windows开发环境似乎不太友好，很多获取数据的脚本都是`*.sh`）；

  - 关于`assignment3`：

    - 之前让我闻之丧胆的RNN和LSTM，实现起来感觉居然是最简单的（里面的计算主要是element-wise的乘积与简单的线性函数，BP过程就比较简单了），但LSTM的computation graph有点意思（可以在其中瞥到ResNet的gradient high-way），待整理附上；

  - （手头没有pad做电子笔记，拍照放大又会糊，目前想到扫描上传工序又比较麻烦，所以还不知怎么把手写的东西push上来orz...）

- 



<br />

<br />

----



<div align="center">by Divsigma@github.com</div>

