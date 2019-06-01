<style type="text/css">
.md-toc {
    font-size: 25px;
}
</style>

[TOC]

# 最大似然估计（MLE）



## 什么是最大似然估计？

最大似然估计是一种参数估计的方法。其核心思想是：找到参数$\theta$的一个估计值，使得当前样本出现的可能性最大。

设总体$X$属于离散型，其分布律$P\{X=x\}=p(x;\theta), \theta \in \Theta$的形式为已知，$\theta$为待估参数，$\Theta$是$\theta$可能取值的范围。设$X_1,X_2,\cdots,X_n$是来自$X$的样本，则$X_1,X_2,\cdots,X_n$的联合分布律为：
$$
\prod_{i=1}^{n}p(x_i;\theta)
$$
样本$X_1,X_2,\cdots,X_n$取到观察值$x_1,x_2,\cdots,x_n$的概率，即事件$\{X_1=x_1,X_2=x_2,\cdots,X_n=x_n\}$发生的概率为：
$$
L(\theta)=L(x_1,x_2,\cdots,x_n;\theta)=\prod_{i=1}^{n}p(x_i;\theta),\theta\in\Theta
$$
这一概率随$\theta$的取值而变化，它是$\theta$的函数，$L(\theta)$称为样本的**似然函数**。

最大似然估计法，就是固定样本观察值$x_1,x_2,\cdots,x_n$，在参数$\theta$取值的可能范围$\Theta$内挑选出一个$\hat\theta$，使得似然函数$L(x_1,x_2,\cdots,x_n;\theta)$达到最大。即：
$$
\hat\theta=\mathop{\arg\max}_{\theta\in\Theta}L(x_1,x_2,\cdots,x_n;\theta)
$$
这样得到的$\hat\theta$与样本值$x_1,x_2,\cdots,x_n$有关，常记为$\hat\theta(x_1,x_2,\cdots,x_n)$，称为参数$\theta$的**最大似然估计值**，而相应的统计量$\hat\theta(X_1,X_2,\cdots,X_n)$称为参数$\theta$的**最大似然估计量**。

## 为什么要有参数估计？

当模型已定，参数未知时。

例如，假设我们知道全国人民的身高服从正态分布，但不知道均值和方差。这时可以通过采样，观察其结果，然后再用样本数据的结果推出正态分布的均值与方差的最大概率值，这样就可以知道全国人民的身高分布的函数。

## 举例

1. 抛硬币。现有一个正反面不是很均匀的硬币，如果正面朝上记为H，反面朝上记为T，抛10次的结果如下：

   T, T, T, H, T, T, T, H, T, T

   求这个硬币正面朝上的概率有多大？

   

   很显然，这个概率是0.2。现在用MLE的思想来求解它。

   

   设$x_1,x_2,\cdots,x_n$是相应于样本$X_1,X_2,\cdots,X_n$的一个样本值。

   不妨用$x_i=1$表示正面朝上，$x_i=0$表示反面朝上

   

   设正面朝上的概率为$\theta$，抛硬币服从二项分布$X \sim b(1,\theta)$，$X$的分布律为：
   $$
   P\{X=x\}=p(x;\theta)=\theta^x (1-\theta)^{1-x},    x=0,1
   $$
   似然函数为：
   $$
   L(\theta)=\prod_{i=1}^{n}p(x_i;\theta)=\prod_{i=1}^{n}\theta^{x_i}(1-\theta)^{1-x_i}
   $$
   取对数后，为
   $$
   \begin{align*}
   ln L(\theta)&=ln\prod_{i=1}^{n}\theta^{x_i}(1-\theta)^{1-x_i}\\
   &=\sum_{i=1}^{n}ln\{\theta^{x_i}(1-\theta)^{1-x_i}\}\\
   &=\sum_{i=1}^{n}[ln\theta^{x_i}+ln(1-\theta)^{1-x_i}]\\
   &=\sum_{i=1}^{n}[x_iln\theta+(1-x_i)ln(1-\theta)]\\
   &=\sum_{i=1}^{n}x_iln\theta+(n-\sum_{i=1}^{n}x_i)ln(1-\theta)
   \end{align*}
   $$

   求导：
   $$
   \frac{\partial ln L(\theta)}{\partial \theta}=\frac{\sum_{i=1}^{n} x_i}{\theta}-\frac{n-\sum_{i=1}^{n} x_i}{1-\theta}
   $$

   令$\frac{\partial ln L(\theta)}{\partial \theta}=0$，可得：
   $$
   \hat \theta = \frac{\sum_{i=1}^{n} x_i}{n}
   $$
   可知概率$\hat \theta=0.2$

   

2. 设$X \sim N(\mu, \sigma^2)$, $\mu, \sigma^2$为未知参数，$x_1,x_2,\cdots,x_n$是来自$X$的一个样本值。求$\mu, \sigma^2$的最大似然估计量。

   解：$X$的概率密度为：
   $$
   f(x;\mu,\sigma^2)=\frac{1}{\sqrt{2\pi}\sigma}exp[-\frac{1}{2\sigma^2}(x-\mu)^2]
   $$
   似然函数为：
   $$
   \begin{align*}
   L(\mu,\sigma^2)&=\prod_{i=1}^n \frac{1}{\sqrt{2\pi}\sigma}exp[-\frac{1}{2\sigma^2}(x_i-\mu)^2]\\
   &=(\frac{1}{\sqrt{2\pi}\sigma})^n exp[-\frac{1}{2\sigma^2}\sum_{i=1}^n (x_i-\mu)^2]\\
   &=(\frac{1}{2\pi \sigma^2})^{\frac{n}{2}} exp[-\frac{1}{2\sigma^2}\sum_{i=1}^n (x_i-\mu)^2]\\
   &=(2\pi)^{-\frac{n}{2}}(\sigma^2)^{-\frac{n}{2}}exp[-\frac{1}{2\sigma^2}\sum_{i=1}^n (x_i-\mu)^2]
   \end{align*}
   $$
   它的对数：
   $$
   ln L(\mu,\sigma^2)=-\frac{n}{2}ln(2\pi)-\frac{n}{2}ln(\sigma^2)-\frac{1}{2\sigma^2}\sum_{i=1}^n (x_i-\mu)^2
   $$
   令
   $$
   \left\{
   	\begin{array}{l}
   		\frac{\partial ln L(\mu,\sigma^2)}{\partial \mu}=\frac{1}{\sigma^2}\Sigma_{i=1}^n(x_i-\mu)=0\\
   
   		\frac{\partial ln L(\mu,\sigma^2)}{\partial \sigma^2}= -\frac{n}{2\sigma^2} + \frac{1}{2 \sigma^4}\sum_{i=1}^n (x_i-\mu)^2=0
   	\end{array}
   \right.
   $$
   联合求解，得到参数$\mu$和$\sigma^2$的最大似然估计值分别为：
   $$
   \left\{
   	\begin{array}{l}
   		\hat \mu = \overline x = \frac{1}{n} \sum_{i=1}^n x_i\\
   
   		\hat \sigma^2 = \frac{1}{n} \sum_{i=1}^n (x_i-\overline x)^2
   	\end{array}
   \right.
$$
   相应的最大似然估计量分别为： 
   
    $\hat \mu=\bar X$,  $\hat \sigma^2=\frac{1}{n}\sum_{i=1}^n (X_i- \overline X)$










