# openmm-chargeflux

# 包含构象决定的电荷输运的静电相互作用

分子内的电荷输运对于dipole moment的模拟有着至关重要的作用，分子运动的细节受这一性质影响极大。本文在此将构象决定的电荷输运加入经典的静电相互作用中，并对周期性体系使用Ewald Summation进行处理。

对于本文描述的体系，假定其有$N$个粒子，对于每个粒子$i$，其坐标记为$p_i$，电荷记为$q_i$。体系总能量用$E$表示。两粒子距离记为$r_{ij}$。

## 体系的静电势能

体系静电势能遵循库伦定律:

$$E = \sum^{N}_{i=1}\sum^{N}_{j=i+1}k_e \frac{q_i q_j}{r^2_{ij}}$$

这一描述可以直接运用于非周期体系。对于周期性体系，一般使用Ewald Summation处理。Ewald Summation将点电荷使用Gaussian函数展开，并扣除相反的Gaussian函数。点电荷+反Gaussian函数在实空间快速收敛，而展开的Gaussian函数可以在reciprocal space下准确求和。计算中将能量分为三部分：倒易空间部分$E_{rec}$，短程部分$E_{dir}$，以及自相关校正$E_{self}$。其表达式为：

$$E_{rec}=\frac{1}{2\pi V} \sum_{\boldsymbol k \neq 0} \frac{exp(-( \frac{\pi\boldsymbol{k}}{\alpha})^2)}{\boldsymbol{k}^2} \sum^{N}_{i=1} \sum^{N}_{j=1} q_i q_j exp\{2\pi i \boldsymbol{k}(\boldsymbol{p}_i - \boldsymbol{p}_j)\}$$

$$E_{dir}=\frac{1}{2}\sum^{N}_{i=1} \sum^{N}_{j=i+1}q_i q_j \frac{erfc(\alpha r_{ij})}{r_{ij}}$$

$$E_{self}=-\frac{\alpha}{\sqrt{\pi}}\sum^{N}_{i}q^2_i$$

其中，倒易空间能量$E_{rec}$可以化简为

$$E_{rec}=\frac{1}{2\pi V} \sum_{\boldsymbol k \neq 0} \frac{exp(-( \frac{\pi\boldsymbol{k}}{\alpha})^2)}{\boldsymbol{k}^2} \sum^{N}_{i=1} \sum^{N}_{j=1} q_i q_j exp\{2\pi i \boldsymbol{k}(\boldsymbol{p}_i - \boldsymbol{p}_j)\}$$

$$E_{rec}=\frac{1}{2\pi V} \sum_{\boldsymbol k \neq 0} \frac{exp(-( \frac{\pi\boldsymbol{k}}{\alpha})^2)}{\boldsymbol{k}^2} \sum^{N}_{i=1} \sum^{N}_{j=1} q_i q_j cos\{2\pi \boldsymbol{k}\boldsymbol{p}_i - 2\pi \boldsymbol{k}\boldsymbol{p}_j\}$$

$$E_{rec}=\frac{1}{2\pi V} \sum_{\boldsymbol k \neq 0} \frac{exp(-( \frac{\pi\boldsymbol{k}}{\alpha})^2)}{\boldsymbol{k}^2} \sum^{N}_{i=1} \sum^{N}_{j=1} q_i q_j \{cos(2\pi \boldsymbol{k}\boldsymbol{p}_i)cos(2\pi \boldsymbol{k}\boldsymbol{p}_j)+sin(2\pi \boldsymbol{k}\boldsymbol{p}_i)sin(2\pi \boldsymbol{k}\boldsymbol{p}_j)\}$$

$$E_{rec}=\frac{1}{2\pi V} \sum_{\boldsymbol k \neq 0} \frac{exp(-( \frac{\pi\boldsymbol{k}}{\alpha})^2)}{\boldsymbol{k}^2} \{ (\sum^N_{i=1}q_i cos(2\pi \boldsymbol{k}\boldsymbol{p_i}))^2 + (\sum^N_{i=1}q_i sin(2\pi \boldsymbol{k}\boldsymbol{p_i}))^2 \}$$

这一形式对点电荷系统普遍有效。对于加入电荷输运的体系，原子点电荷是其他原子坐标的函数，即为

$$q_i = q_i(p_1, p_2, ..., p_N)$$

这一关系式并不影响能量的求解，但对于导数的计算，这一依赖关系影响很大。

## 体系静电势能的导数

对于固定点电荷的Ewald Summation来说，其势能导数如下：

$$ \frac{\partial E_{self}}{\partial p_{i,l}} = 0 $$

$$ \frac{\partial E_{dir}}{\partial p_{i,l}} = \frac{1}{2} \sum^{N}_{j=i+1}q_i q_j \frac{\partial}{\partial p_{i,l}} \frac{erfc(\alpha r_{ij})}{r_{ij} } $$

$$\frac{\partial E_{rec}}{\partial p_{i,l}}=\frac{1}{2\pi V} \sum_{\boldsymbol k \neq 0} \frac{exp(-( \frac{\pi\boldsymbol{k}}{\alpha})^2)}{\boldsymbol{k}^2} \frac{\partial}{\partial p_{i,l}} \{ (\sum^N_{j=1}q_j cos(2\pi \boldsymbol{k}\boldsymbol{p_j}))^2 + (\sum^N_{j=1}q_j sin(2\pi \boldsymbol{k}\boldsymbol{p_j}))^2 \}$$

其中倒易空间能量导数可以进一步整理如下：

$$\frac{\partial E_{rec}}{\partial p_{i,l}}=\frac{1}{2\pi V} \sum_{\boldsymbol k \neq 0} \frac{exp(-( \frac{\pi\boldsymbol{k}}{\alpha})^2)}{\boldsymbol{k}^2} \left\{ \frac{\partial \left[ \sum^N_j q_j cos\left( 2\pi \boldsymbol{k} \boldsymbol{p_j} \right) \right]^2 }{\partial p_{i,l}} + \frac{\partial \left[ \sum^N_j q_j sin\left( 2\pi \boldsymbol{k} \boldsymbol{p_j} \right) \right]^2 }{\partial p_{i,l}} \right\}$$

$$\frac{\partial \left[ \sum^N_j q_j cos\left( 2\pi \boldsymbol{k} \boldsymbol{p_j} \right) \right]^2 }{\partial p_{i,l}} = -2\left[ \sum^N_j q_j cos\left( 2\pi \boldsymbol{k} \boldsymbol{p_j} \right) \right] 2 \pi \boldsymbol{k}_l q_i sin \left( 2\pi \boldsymbol{k} \boldsymbol{p}_i \right)$$

$$\frac{\partial \left[ \sum^N_j q_j sin\left( 2\pi \boldsymbol{k} \boldsymbol{p_j} \right) \right]^2 }{\partial p_{i,l}} = 2\left[ \sum^N_j q_j sin\left( 2\pi \boldsymbol{k} \boldsymbol{p_j} \right) \right] 2 \pi \boldsymbol{k}_l q_i cos \left( 2\pi \boldsymbol{k} \boldsymbol{p}_i \right)$$

对于电荷输运模型，其电荷同时也是坐标的导数。于是有

$$\frac{\partial E(p_1,..,p_N,q_1,...,q_N)}{\partial p_{i,l}} = \sum^{N}_{j=1}(\frac{\partial E(p)}{\partial p_j} \frac{\partial p_{j}}{\partial p_{i,l}} + \frac{\partial E(q)}{\partial q_j} \frac{\partial q_{j}}{\partial p_{i,l}})$$

考虑到$\frac{\partial E(p)}{\partial p_j} \frac{\partial p_{j}}{\partial p_{i,l}}$于$i\neq j$时为0，有

$$\frac{\partial E(p_1,..,p_N,q_1,...,q_N)}{\partial p_{i,l}} = \frac{\partial E(p)}{\partial p_{i,l}} + \sum^{N}_{j=1} \frac{\partial E(q)}{\partial q_j} \frac{\partial q_{j}}{\partial p_{i,l}}$$

其中$\frac{\partial E(p)}{\partial p_{i,l}}$一项与固定电荷下形式相同。对于电荷求导的部分，$\frac{\partial q_{j}}{\partial p_{i,l}}$于定义电荷输运过程中求得，可以被看作已经知晓，因此求取$\frac{\partial E}{\partial q}$在这一过程中至关重要。

对于倒易空间，有

$$\frac{\partial E_{rec}}{\partial q_i} = \frac{1}{2\pi V} \sum_{\boldsymbol k \neq 0} \frac{exp(-( \frac{\pi\boldsymbol{k}}{\alpha})^2)}{\boldsymbol{k}^2} \left\{ \frac{\partial \left[ \sum^{N}_{j} q_j cos\left( 2\pi \boldsymbol{k} \boldsymbol{p_j} \right) \right]^2 }{\partial q_i} + \frac{\partial \left[ \sum^N_j q_j sin\left( 2\pi \boldsymbol{k} \boldsymbol{p_j} \right) \right]^2 }{\partial q_i} \right\}$$

其中

$$\frac{\partial \left[ \sum^N_j q_j cos\left( 2\pi \boldsymbol{k} \boldsymbol{p_j} \right) \right]^2 }{\partial q_i} = 2 \left[ \sum^N_j q_j cos\left( 2\pi \boldsymbol{k} \boldsymbol{p_j} \right) \right] cos \left( 2\pi \boldsymbol{k}\boldsymbol{p}_i \right)$$

$$\frac{\partial \left[ \sum^N_j q_j sin\left( 2\pi \boldsymbol{k} \boldsymbol{p_j} \right) \right]^2 }{\partial q_i} = 2 \left[ \sum^N_j q_j sin\left( 2\pi \boldsymbol{k} \boldsymbol{p_j} \right) \right] sin \left( 2\pi \boldsymbol{k}\boldsymbol{p}_i \right)$$

对于实空间部分，我们有

$$ \frac{\partial E_{dir}}{\partial q_i} = \frac{1}{2} \sum^{N}_{j=1}\sum^{N}_{k=j+1}\frac{erfc(\alpha r_{jk})}{r_{jk} } \frac{\partial}{\partial q_i} (q_j q_k) $$

去除与$q_i$无关部分，余下

$$ \frac{\partial E_{dir}}{\partial q_i} = \frac{1}{2} \sum^{N}_{j \neq i} q_j \frac{erfc(\alpha r_{ij})}{r_{jk} }$$

自相关校正项在固定电荷下为常数，对总能量变化并无贡献。然而在允许电荷输运的前提下，自相关校正项亦为体系坐标的函数。导数为：

$$ \frac{\partial E_{self}}{\partial q_i} = -\frac{2\alpha}{\sqrt{\pi}} q_i $$
