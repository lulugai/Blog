# [12]Transfer Learning

<img src="D:\Course\Lihungyi-spr2021\LeeHungyi-Note\2021-12-10.png" alt="2021-12-10" style="zoom: 67%;" />

- Domain shift: Training and testing data have different distributions. (**HW11**)

  <img src="D:\Course\Lihungyi-spr2021\LeeHungyi-Note\2021-12-10-160731.png" alt="2021-12-10-160731" style="zoom: 33%;" />

- DaNN: 我们采用DaNN来实现迁移学习。DaNN的主要核心是：模型分成两部分，上半部分是一个特征提取器，下半部分是一个分类器，让source data和target data经过特征提取器后是同分布的。这种操作能够有效降低后续分类器的错误率。



# [14]Life Long Learning

Life Long Learning的意思是：机器首先学习了任务一，然后学习了任务二，此时机器同时掌握了任务一和任务二，如果机器在之后的时间中继续学习别的任务，机器就能够拥有更多的技能，理想状态下，机器可以无所不能。

要想实现Life Long Learning，需要解决以下几个问题：（1）如何在学习新知识时对旧知识进行保留；（2）在训练不同的任务时，如何进行知识的迁移（3）如何进行有效的模型扩张使模型更加符合当前实际情况而不浪费计算资源。实际过程中，Life Long Learning容易碰到灾难性遗忘的现象，目前对于灾难性遗忘的解决方法有以下几个常见的做法：

- Dynamic Expansion：直接搞一批新的参数来学习新任务，单这样模型的参数会越来越多，往往需要搭配一些模型压缩的操作。

- Rehearsal：如果让新任务上的梯度能尽可能接近旧任务上的梯度，那就可以保留很大一部分旧知识。

- Regularization：加一些正则项来避免跟旧任务关联比较大的参数的更新幅度过大。这是因为大部分神经网络都是大规模参数中有部分参数对模型并无决定性的作用，因此正则化的方法是有用武之地的。

  作业中需要用到的EWC和MAS实际上都是基于Regularization的方法。在非Life Long Learning的问题上，模型在任务A上训练完之后，直接拿去任务B上进行微调，而这种训练出来的模型并不能完成任务A了（因为出现了灾难性以往），但是当我们添加一个正则项（L2）之后，使任务B上训练完的参数不能离任务A上训练完的结果太远，这就是Regularization的基本思想。

  Regularization中，直接加入L2正则项并没有考虑不同的参数对于任务的重要性，会使任务B的学习陷入瓶颈，所以在进行基于Regularization的方法时，需要计算每个参数$\theta_i$对任务A的重要性$\Omega_i$，然后添加了正则项的损失函数就变成了
  $$
  L(\theta) = L_B(\theta) + \frac{\lambda}{2}\sum_i\Omega_i(\theta_i - \theta^*_{A,i})^2
  $$



# [10]Anomaly Detection

**Anomaly Detection**：异常侦测。如果此时的输入x和训练数据很像，经过异常侦测后，我们判断它为“正常”数据，反之则判断为“异常”数据。

**Anomaly Detection方法：**

- 用classifier的信心分数。【虽然简单，但效果不错】
- 用AutoEncoder的方式。训练一个autoencoder，当图片是异常图片时，decoder得到的图片和原图片相差越大。
- 用K-means的方式。正常数据与所在类的中心的距离比异常数据与所在类中心的距离要小。
- 用PCA的方式。计算训练数据的主成分，将测试数据投影在这些成分上，再将这些投影重建，对重建的图片和原图进行平方差的计算，正常数据的平方差结果比异常数据的平方差结果要小。

# [11]GAN

**GAN**:Generative adversarial network生成对抗网络。

GAN框架让一个深度学习模型学习训练数据分布，从而生成具有同分布的类似数据。

GAN由两个不同的模型组成，一个是生成模型G(Generator)，一个是鉴别模型D(Discriminator)。其中，G的作用是产生fake图像使其的分布与训练图像相似; D的作用是来判断这个fake图像与真正的图像是否相同。

训练过程中，G通过产生越来越好的fake图像，来不断试图去打败D；同时D也是如此。这个训练在当生成器生成看起来像是直接来自训练数据的完美赝品时，判别器总是猜测生成器输出为真或假的概率为50%时达到平衡。

此次实验采用DCGAN作为模型架构。DCGAN是将CNN与GAN的一种结合，将GAN的G和D换成了两个CNN。

# [8]Sequence to Sequence

常见的seq2seq模型都是encoder-decoder模型，主要由Encoder和Decoder两部分组成，这两部分大多数情况下均由RNN来实现，作用是解决输入和输出的长度不一致的问题。Encoder是将一连串的输入编码为单个向量，Decoder是将Encoder输出的单个向量逐步解码，一次输出一个结果，每次的输出会影响到下一次的输出，一般会在Decoder的开头加入“<BOS>”来表示开始解码，在Decoder的结尾加入“<EOS>”来表示输出结束

# [7]Network Compression

李宏毅老师上课介绍了四种network compression的方式（知识蒸馏，网路剪枝，网络重构，参数量化).

## 知识蒸馏

主要思想：通过使用一个较大的已经训练好的网络去训练一个较小的网络，使得小网络可以尝试复制出大网络的输出。

### 理论依据

知识蒸馏使用Teacher-Student模型。知识蒸馏的过程分2个阶段：

1. 原始模型训练。即训练Teacher模型，简称Net-T。
2. 小模型训练。即训练Student模型，简称Net-S。

知识蒸馏一般面向于分类问题，即模型的最后有softmax层的问题。使用Net-T来训练Net-S时，可以让Net-S去学习Net-T的泛化能力。

原本的softmax函数为：
$$
S_i= \frac{e^{V_i}}{\sum_je^{V_j}}
$$
如果直接采用softmax层的输出值用来训练小网络，当softmax输出概率分布熵相对较小时，负标签的值会接近0，这对损失函数的贡献非常小，所以在进行知识蒸馏的过程中，引入了“温度”的概念。

<img src="D:\Course\Lihungyi-spr2021\LeeHungyi-Note\162211.png" style="zoom:50%;" />

更新后的softmax函数为：
$$
S_i=\frac{e^{V_i}/T}{\sum_j e^{V_j}/T}
$$

### 知识蒸馏的具体方法

知识蒸馏分为两步：原始模型训练+小模型训练。原始模型训练就是普通训练方法，因此只需要讲述小模型训练过程即可。

小模型训练过程中的损失函数L LL由softmax的差异loss和分类误差loss加权得到。
$$
L = \alpha L_{soft} + \beta L_{hard}
$$

## 网路剪枝

网路剪枝分为神经元剪枝和权重剪枝，神经元剪枝实际上就是让一个已经学完的模型中的神经元进行删减，使得整个网路变得更瘦。权重剪枝实际上就是让一个已经学完的模型中的连接线进行删减，使得参数量更少。本文重点介绍神经元剪枝。

如果要进行神经元剪枝，就必须要先衡量神经元的重要性，衡量完所有的神经元后，就可以把比较不重要的神经元删减掉。

如何衡量神经元重要性呢？可以看batchnorm层的$\gamma$因子来决定神经元的重要性。

# Adversarial Attack



# Explainable AI

**Why we need Explainable ML?**

1. Loan issuers are required by law to explain their models. 
2.  Medical diagnosis model is responsible for human life. Can it be a black box?
3. If a model is used at the court, we must make sure the model behaves in a nondiscriminatory manner.
4. If a self-driving car suddenly acts abnormally, we need to explain why. 
5. **Make people (your customers, your boss, yourself) comfortable.**

- Sailency map
- filter explanation
- Lime

