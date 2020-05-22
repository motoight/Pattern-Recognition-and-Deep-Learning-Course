### how to run
------
- 运行之前需要打开py文件修改相应的filepath
- 所有运行日志均在runs-5文件夹中
- author： motoight
-------


### some point to say
- 实验五主要实现GAN，WGAN，WGAN-GP三种网络（实际网络差别不大，主要是使用了不同的loss函数），实验背景是拟合高斯分布，比较不同GAN网络的收敛速度。
- 下面是实现的效果，使用gan网络，具体参数可以参照实验报告。
- 实验报告中重点比较了深度学习中不同优化器的原理，优劣，适用场景等。

### experiment result
- creatgif的函数再utils.py中，但是自己生成的gif动图文件太大，可以达到28M。为了方便上传，我使用了[https://www.tuhaokuai.com/gif](https://www.tuhaokuai.com/gif)，压
压缩后文件之后300多k，但是效果还是很清晰。果然cv一途，任重道远。
![img](https://github.com/motoight/Pattern-Recognition-and-Deep-Learning-Course/blob/master/dl_lab/dl_lab5_gan/nets/fig/fitting_distribution%20(1).gif)
