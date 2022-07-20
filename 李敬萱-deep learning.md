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
6. 特征分解：假设$\boldsymbol{A}$有n个线性无关的特征向量{$\boldsymbol{v}^{(1)},...,\boldsymbol{v}^{(n)}$},对应的特征值为{$\lambda_{1},...,\lambda_{n}$}。我们将特征向量连接成一个矩阵$\boldsymbol{V} = [ \boldsymbol{v}^{(1)},..., \boldsymbol{v}^{(n)} ]$,$ \boldsymbol{\lambda} = [\lambda_{1},...,\lambda_{n}]^{T} $，$ \boldsymbol{A} $的<b>特征分解</b>记作$$ \boldsymbol{A} =  \boldsymbol{V}diag( \boldsymbol{\lambda}) \boldsymbol{V}^{-1}$$
7. 特征向量:指与$\boldsymbol{A}$相乘后相当于对该向量放缩的非零向量$\boldsymbol{v}$:$$\boldsymbol{Av} = \lambda\boldsymbol{v}$$
8. 标量$\lambda$被称为该特征向量对应的特征值
9. 奇异矩阵：矩阵是奇异的当且仅当含有零特征值
10. 实对称矩阵的特征分解可用于优化二次方程$f( \boldsymbol{x}) =  \boldsymbol{x}^T \boldsymbol{Ax}$,其中$\Vert \boldsymbol{x} \Vert_2=1$。当$ \boldsymbol{x} $等于$ \boldsymbol{A} $的某个特征向量时$f$讲返回对应的特征值。（a）在限制条件下，函数$f$的最大值是最大特征值，最小值是最小特征值。（b）  
（a）的证明：因为$ \boldsymbol{Ax} =  \lambda \boldsymbol{x}$,所以当$ \boldsymbol{x}为特征值时 $ $ f( \boldsymbol{x}) = \lambda \boldsymbol{x}^T \boldsymbol{x}$，因为$\Vert \boldsymbol{x} \Vert_2=1$，所以（b）得证。
（b）的证明：[参考论文连接](http://www.doc88.com/p-8028028861219.html)
$$\boldsymbol{f(x)} =  \boldsymbol{x}^T\boldsymbol{Ax} ， \boldsymbol{x} \in \boldsymbol{R}^n$$ $$ \boldsymbol{x}^T \boldsymbol{Ax}=\boldsymbol{x}^T  \boldsymbol{P}^T\Lambda\boldsymbol{Px}= \boldsymbol{(Px)}^T\Lambda \boldsymbol{( \boldsymbol{Px})}，其中\boldsymbol{P} \boldsymbol{P}^T=E,\Lambda=diag( \boldsymbol{\lambda})$$$$ \boldsymbol{x}^T \boldsymbol{Ax}=\sum^n_{i=1}\lambda_i\Vert \boldsymbol{Px}\Vert^2， \boldsymbol{x}\in \boldsymbol{R}^n$$$$ \lambda_{min}\Vert \boldsymbol{Px}\Vert^2\leq\boldsymbol{x}^T \boldsymbol{Ax}\leq\lambda_{max}\Vert \boldsymbol{Px}\Vert^2， \boldsymbol{x}\in \boldsymbol{R}^n$$$$ \lambda_{min}\Vert \boldsymbol{x}\Vert^2\leq\boldsymbol{x}^T \boldsymbol{Ax}\leq\lambda_{max}\Vert \boldsymbol{x}\Vert^2， \boldsymbol{x}\in \boldsymbol{R}^n$$$$ \lambda_{min}\leq\frac{\boldsymbol{x}^T }{\Vert \boldsymbol{x}\Vert}\boldsymbol{A} \frac{\boldsymbol{x}}{\Vert \boldsymbol{x}\Vert}\leq\lambda_{max}， \boldsymbol{x}\in \boldsymbol{R}^n，证毕$$
11. 奇异值分解（SVD）：任何一个实矩阵均可奇异值分解
[奇异值分解的理解视频](https://www.bilibili.com/video/BV1MT4y1y75x?spm_id_from=333.880.my_history.page.click&vd_source=ae580808c513257258006ec36f2bd35f)
首先线性代数的核心是线性变换，线性变换$ \boldsymbol{L} $的一种表现形式为矩阵$ \boldsymbol{A} $，SVD需要找到线性变换$ \boldsymbol{L} $：$\boldsymbol{V}\rightarrow\boldsymbol{W}$
任何一个线性变换都可以由旋转、伸缩、旋转三步变换，对应$$\boldsymbol{A=UDV}^T$$矩阵中的旋转表达形式：$$M(\theta)=\begin{bmatrix}
    cos\theta&-sin\theta\\
    sin\theta&cos\theta
\end{bmatrix}$$显然为正交矩阵，所以$\boldsymbol{U}$与$\boldsymbol{V}^T$均为正交矩阵，$\boldsymbol{D}$可以表示为$\begin{bmatrix}
    diag(\boldsymbol{\lambda})&0\\
    0&0
\end{bmatrix}$的形式。
12. Moore-Penrose伪逆$$\boldsymbol{A}^{+} = \boldsymbol{VD}^+\boldsymbol{U}^T$$$\boldsymbol{D}^+$的伪逆是非零元素取倒数再转置得到的
13. 关于降维：[参考视频](https://www.bilibili.com/video/BV1MT4y1y75x?spm_id_from=333.880.my_history.page.click&vd_source=ae580808c513257258006ec36f2bd35f)17min55s
14. 主成分分析：数据降维,类似数分中$\boldsymbol{f(x) = g(x)}+\epsilon,\epsilon\rightarrow0$,将无穷小舍去的思想
15. 信息论
    1. 1979年 ID3基于信息增益 分类决策树
    1993年  C4.5基于增益率 分类决策树
    1984年 CART 基于基尼指数 回归决策树
    信息熵：量化信息 一条信息的信息量大小=和它的不确定性有直接关系，不确定性大，信息熵大 使用概率刻画不确定性
    2. 数学定义
       假定随机变量x服从分布：$P($x $=$ x$_i)=p_i,，i=1,...m$，则x的信息熵定义$H($x$)=-\sum^m_{i=1}{p_ilog_2p_i}$
       是随机变量x的不确定性的度量，是描述x包含的所有可能发生事件所包含的信息量，熵只依赖于概率分布，与随机变量取值无关，且均匀分布信息量最大，此时单位为bit
16. 贝叶斯公式：
$$P(H|E) =\frac{P(E,H)}{P(E)} = \frac{P(E|H)P(H)}{P(E)}$$
17. 贝叶斯公式应用实例：
给定一封邮件，判断其是否属于垃圾邮件，$E$表示邮件，$E$是由$n$个单词组成的，记为$E = {\{e_1,e_2,...e_n\}}$,H表示该邮件是垃圾邮件$$P(H|E) =  \frac{P(E|H)P(H)}{P(E)}=\frac{P(e_1,e_2,...e_n|H)P(H)}{P(e_1,e_2,...e_n)}\ (1)$$在$(1)$式中表达的是给定的邮件与已知的垃圾邮件单词完全一致，显然不合理$$扩展为\frac{\prod \limits_{i=1}^{n} P(e_i|H)P(H)}{P(e_1,e_2,...e_n)} \ (2)$$(1)$\rightarrow$(2)是朴素贝叶斯的假设，各单词出现相互独立，$P(e_i)$可以用频率值替代，同理可求出不是垃圾邮件的概率，因为分母相同，所以可以仅比较分子即可实现分类
18. 机器学习基础任务
    1. 分类(Classification)$$f:R^n\rightarrow\{1,2,...,k\}$$
    2. 输入缺失分类(Classification with input missing)
       输入的度量无法被保证，要学习一组函数，而非一个函数
    3. 回归(Regression)$$f:R^n\rightarrow R$$
    4. 转录
    5. 机器翻译
    6. 结构化输出
    7. 异常检测
    8. 合成和采样
    9. 缺失值填补
    10. 去噪
    11. 密度估计或概率分布律函数估计
19. 经验
    1. 无监督学习
    2. 监督学习
20. 正则化
21. SVM　[参考链接](https://zhuanlan.zhihu.com/p/31652569?ivk_sa=1024320u)
    1.  间隔：$$\begin{equation*}\gamma:= \min_i \frac{2\vert  \boldsymbol{w^Tx_i+}b \vert}{\Vert \boldsymbol{w} \Vert}\end{equation*}$$
    2. 线性支持向量机:希望找到使间隔最大的参数$\boldsymbol{w},b$
    $$\begin{equation*}
    \begin{split}
    &\max_{\boldsymbol{w},b} \min_{i} \frac{2\vert  \boldsymbol{w^Tx_i+}b \vert}{\Vert \boldsymbol{w} \Vert}\\
    &s.t.\ \begin{array}{lc}
    y_i(\boldsymbol{w^Tx_i+}b)>0，\forall i \in [1,n]
    \end{array}
    \end{split}
    \end{equation*}
    $$
    若 $(\boldsymbol{w^*},b^*)是原问题的解，那么 (r\boldsymbol{w^*},rb^*)$ 也是原问题的解，不妨令$$\min_{i}\vert \boldsymbol{w^Tx+}b\vert = 1$$
    原问题转换为
    $$\begin{equation*}
    \begin{split}
    &\min_{\boldsymbol{w,b}} \frac{\Vert \boldsymbol{w} \Vert}{2}=\frac{1}{2} {\boldsymbol{w}^T\boldsymbol{w}}\\
    &s.t.\ \begin{array}{lc}
    y_i(\boldsymbol{w^Tx_i+}b)=1，\forall i \in [1,n]
    \end{array}
    \end{split}
    \end{equation*}
    $$
    3. kernel method
    核心思想：将低维数据映射到高维空间
    4.  kernel trick
    因为需要的是内积，所以找到kernel function $k(\boldsymbol{u,v})=\Phi(\boldsymbol{u})^T\Phi(\boldsymbol{v})$，简化运算