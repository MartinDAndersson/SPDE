I'll convert the LaTeX document to markdown format:

# Non-parametric estimation of the diffusion function

Let us now introduce more precisely the setting of the present paper.
Let us fix $T>0$.
We consider the following $d$-dimensional stochastic partial differential equation driven by a white-coloured noise $W$ with spatial covariance $\Gamma$ (see Section prl for a precise definition) and with bounded regular initial condition.

$$
\begin{cases}
  -Lu(t,x)+\sigma(u(t,x))\dot W(t,x)=0\\
  u(\cdot,0)=u_0,\\
\end{cases}       
$$

for $x\in\mathbb{R}^d$, $t\in [0,T]$. Here, $u_0$ belongs to the space of bounded Lipschitz functions, $\sigma:\mathbb{R}\rightarrow\mathbb{R}$ is an (unknown) measurable function which satisfies the following condition:

**Assumption (A)**: $\sigma$ is non negative and $M$-Lipschitz continuous for some $M>0$.

Let us now introduce the Riesz potential. Let $0<\beta<d$ and $\mu$ be a positive measure on $\mathbb{R}^d$. The Riesz potential $I_{d-\beta}$ is defined for $x\in\mathbb{R}^d$

$$
\left(I_{d-\beta}\mu\right)(x)=\int_{\mathbb{R}^d}\|x-y\|^{-\beta}d\mu(y)=(K_{d-\beta}*\mu)(x),
$$

with $K_{d-\beta}(y)=\|y\|^{-\beta}$, where $\|.\|$ denotes the $L^2$ norm.

**Assumption (B)** The spatial covariance $\Gamma$ of the noise $W$ can be written as 

$$
\Gamma=(I_{d-\beta}\delta),
$$

with $\beta\leq d$ and $\beta<2$.

On the other hand, we assume that the differential operator $L$ satisfies the following assumptions:

**Assumption (C)**

$$
L:=\partial_t-\sum_{i,j=1}^da_{i,j}(t,x)\partial^2_{x_ix_j}-\sum_{i=1}^db_{i}(t,x)\partial_{x_i},
$$

where $a,b$ are Lipschitz, with the matrix $(a_{ij})$ being uniformly elliptic. *[double check that this is enough to get the estimates we need instead of extra assumption, i.e. add more regularity to coefficient functions if needed]*

**Remark**
The non negativity assumption in **(A)** is necessary for identifiability purpose since we are actually going to estimate the quantity $\sigma^2$.

**Remark** The assumption **(C1)** ensure that zero-order derivative $G(t,\cdot)$ satisfies the growth condition $(i)$ in Assumption **(C2)**, see Section prl.

A noteworthy particular case of the system is the *stochastic heat equation*, given by

$$
\frac{\partial u(t,x)}{\partial t}=\frac{1}{2}\Delta u(t,x)+\sigma(u(t,x))\dot W(t,x),
$$

where $\Delta$ is the Laplace operator. We will use the abbreviation SHE in what follows.

The goal of this paper is to build a consistent non parametric estimator of the process $(\sigma(u(t,x)))_{(x,t)\in \mathbb{R}^d\times [0,T]}$ for the $L^1$ norm. Let us first give a heuristic description of the method we will adopt:

*[Here we can provide an intuitive explanation, maybe by looking at the simplified case where $W$ is smooth.]*

Since the noise $W$ is not a proper function and $u$ is not differentiable, we will have to introduce a discretisation of the operator $L$ and perform a regularisation procedure over a small ball in order to apply this method.
Let $(x_0,t_0)\in \mathbb{R}\times [0,T]$ and consider two discretisation parameters $h,\epsilon>0$. Let
$\mathcal{B}_{x_0,\epsilon}$ be the euclidian ball in $\mathbb{R}^d$ centered in $x_0$ and with radius $\epsilon$. Let $V(\epsilon,d)$ denotes its volume, that is

$$
V(\epsilon,d)=\frac{\pi^\frac{d}{2}}{\Gamma\left(\frac{d}{2}+1\right)}\epsilon^d.
$$

Consider the estimator

$$
\begin{aligned}
&\Sigma_{\epsilon, h}(x_0,t_0)
:=\mathbb{E}_{u(t_0,x_0)}\left[\left(\frac{1}{m(\epsilon)}\int_{\mathcal{B}_{x_0,\epsilon}}\int_{t_0}^{t_0+\epsilon}\right.\right.\\
&\times\left.\left.\left(L^hu(s,y)-\int_{\mathbb{R}^d}L^hG(s,y-z)u_0(z)dz\right)dy,ds\right)^2\right],
\end{aligned}
$$

where $\mathbb{E}_{u(t_0,x_0)}$ is the conditional expectation given $u(x_0,t_0)$, $G$ is the fundamental solution of the associated PDE, $m$ is given by

$$
m(\epsilon)=\sqrt{\epsilon\int_{\mathcal{B}_{0,\epsilon}}\int_{\mathcal{B}_{0,\epsilon}}\Gamma(x-y)dxdy}=\sqrt{\epsilon}m_1(\epsilon)
$$

and $L^h$ is the finite difference approximation of the operator $L$. More precisely, for every function $f:[0,T]\times\mathbb{R}^d\mapsto\mathbb{R}$,

$$
\begin{aligned}
L^h:f(t,x)&\rightarrow (T_{h^2}f)(t,x)-\sum_{i,j=1}^da_{ij}(t,x)(\mathcal D^{2,h}_{ij}f)(t,x)\\
&+\sum_{i=1}^db_i(t,x)(\mathcal D^{1,h}_i f)(t,x),\\
&=(T_{h^2}f)(t,x)-(S^hf)(t,x)
\end{aligned}
$$

where

$$
\begin{aligned}
(T^{h^2}f)(t,x)&=\frac{f(t+h^2,x)-f(t,x)}{h^2}\\
(\mathcal D^{2,h}_{ij} f)(t,x)&=\frac{f(t,x+h(e_i+e_j))+f(x,t)-f(t,x+he_i)-f(t,x+he_j)}{h^2}\\
(\mathcal D^{1,h}_{i} f)(t,x)&=\frac{f(t,x+he_i)-f(t,x)}{h}.
\end{aligned}
$$

Notice that when $\gamma=\delta$, $m_1(\epsilon)=\sqrt{V(\epsilon,d)}$.

The main result of the paper is as follows.

**Theorem**
Let $h>0$, $p\in\mathbb{N}^*$. Let $\rho\in (0,1)$ and let $\epsilon=h^\rho$. Then, assuming that **(A)-(B)-(C1)-(C2)** are verified, we have for all $(x_0,t_0)\in\mathbb{R}^d\times [0,T]$, for all $\gamma\in (\rho,1)$ and for all $(\nu,\kappa)\in \left(1-\frac{\beta}{2},\frac{1}{2}-\frac{\beta}{4}\right)$,

$$
\begin{aligned}
&\left\|\Sigma_{\epsilon, h}(x_0,t_0)-\sigma^2(u(x_0,t_0))\right\|_{L^p(\Omega)}\\
&\leq K(|x_0|^2+|t_0|^2)\left(M(h^{\rho\kappa}+h^{\rho\nu})\right.\\
&\left.+h^{(1-\rho)(d+1-\beta)}+h^{2-2\gamma}+h^{(2-\beta)(\gamma-\rho)}\left(1+log(h)\mathbb{I}_{\beta=1}\right)\right),
\end{aligned}
$$

where $K$ is a constant which depends on $\gamma,M,p,\nu,\kappa,\beta,d,G$ (where $G$ is the fundamental solution of the homogeneous PDE associated to the SPDE)

*[somewhere heuristic derivation and link/comparison to existing literature and SDE case. Also explain the bias/variance tradeoff as with many non-parametric estimators]*

**Remark**
It is actually possible to optimize the rate in the previous result assuming a specific value for $\beta$. Indeed we have to solve the optimization problem

$$
\left\{
  \begin{array}{cll}
    &&\textit{Maximize } \min\left(\rho\nu,\rho\kappa,2-2\gamma,(1-\rho)(d+1-\beta),(2-\beta)(\gamma-\rho)\right)\\
    &&\textit{Given } 1>\gamma>\rho>0
  \end{array}
  \right.
$$

The solution to this problem is obtained for

$$
\begin{aligned}
\rho_0&=\frac{4-2\beta}{4-2\beta+min(\kappa,\nu)(4-\beta)}\\
\gamma_0&=\frac{4+(2-\beta)\rho_0}{4-\beta},
\end{aligned}
$$

yielding an overall rate of convergence

$$h^\frac{(-2+\beta)\min(\kappa,\nu)}{4-2\beta+\min(\kappa,\nu)(4-\beta)}.$$

Taking $\kappa$ arbitrarily close to $\frac{2-\beta}{4}$, we can then see that the rate can be arbitrarily close to

$$h^{\frac{(-2+\beta)}{12-\beta}}$$

(of course, this will yield an exploding constant $K$ as $\kappa\rightarrow\frac{2-\beta}{4}$).

**Remark**
Notice that when $M\neq 0$ (that is, $\sigma$ is not constant) and when $\beta=d=1$ and taking $\kappa,\nu$ arbitrary close to $\frac{1}{4}$ and $\frac{1}{2}$ respectively, we can see by the previous remark that the rate of convergence is arbitrarily close to $\frac{1}{11}$. The low Hölder regularity of the solution to the SPDE is mainly to blame for this surprisingly slow rate. Indeed, if we consider the much simpler problem of estimating a constant volatility $\sigma^2\in\mathbb{R}_+$, then since $M=0$, we can choose $\rho=0$ and the optimal rate is then obtained for $\gamma_0$ such that $2-2\gamma_0=(2-\beta)\gamma_0$ (when $\beta=1$ for example, we have the much faster optimal speed of convergence $log(h)h^{\frac{2}{3}}$, obtained for $\gamma=\frac{2}{3}$).

*[discussion about discretisation of integral, what error it produces, and continue with discussion about regression then how conditional expectation can be computed in practice. That is, for a given point $u(t_0,x_0)$ we observe $u$ on a ball around it but not for all points, so discuss approximation better (by assuming dense observations). That is, $u$ is always measured for $t_k$ and $x_k$ around $(t_0,x_0)$. Make another theorem out of this with the idea that each measurement consist of measures of $u$ on $(t_k,x_j)$, and then one can make a regression problem out of it]*