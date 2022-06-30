## 基本概念 ##
1. <b>tensor</b>:an object that is <b>invariant</b> under a change of coordinates,and has components that change in a <b>special,predictable </b>way under a change of coordinates<br><b>tensor</b>:a collection of <b>vectors</b> and <b>convectors</b> combined together using the <b>tensor produnct</b>
tensor是使用张量积组合在一起的的向量（基）和余向量（变换）集合
即向量本身是不变的，但在坐标系表示是不同的，tensor可以写成array的形式，但不就是array
[tensor理解视频连接](https://www.bilibili.com/video/BV1Ty4y1z74k?p=2&vd_source=ae580808c513257258006ec36f2bd35f)P2 3min30s
2. 一范数：需要区分0与非0小值的情形
3. 线性映射：线性映射是一种抽象定义，矩阵是一种表现形式
4. max范数
$$\Vert \boldsymbol{x} \Vert_\infty = \mathop{max}\limits_{i}\lvert x_{i} \rvert$$
向量中具有最大幅度的元素的绝对值
5. Frobenius范数，衡量矩阵大小$$\Vert \boldsymbol{A} \Vert_F=\sqrt{\sum_{i,j}A^2_{i,j}}$$
6. 特征分解：假设$\boldsymbol{A}$有n个线性无关的特征向量{$\boldsymbol{v}^{(1)},...,\boldsymbol{v}^{(n)}$},对应的特征值{$\lambda_1,...,\lambda_n$}。我们将特征向量连接成一个矩阵$\boldsymbol{V} = [ \boldsymbol{v}^{(1)},..., \boldsymbol{v}^{(n)} ]$,$ \boldsymbol{\lambda} = [\lambda_{1},...,\lambda_{n}]^{T} $，$ \boldsymbol{A} $的<b>特征分解</b>记作$$ \boldsymbol{A} =  \boldsymbol{V}diag( \boldsymbol{\lambda}) \boldsymbol{V}^{-1}$$
7. 特征向量:指与$\boldsymbol{A}$相乘后相当于对该向量放缩的非零向量$\boldsymbol{v}$:$$\boldsymbol{Av} = \lambda\boldsymbol{v}$$
8. 标量$\lambda$被称为该特征向量对应的特征值
9. 奇异矩阵：矩阵是奇异的当且仅当含有零特征值
10. 实对称矩阵的特征分解可用于优化二次方程$f( \boldsymbol{x}) =  \boldsymbol{x}^T \boldsymbol{Ax}$,其中$\Vert \boldsymbol{x} \Vert_2=1$。当$ \boldsymbol{x} $等于$ \boldsymbol{A} $的某个特征向量时$f$讲返回对应的特征值。在限制条件下，函数$f$的最大值是最大特征值，最小值是最小特征值。