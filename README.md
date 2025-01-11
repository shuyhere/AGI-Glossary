# AGI-Glossary

***generalization*** The ability to make accurate predictions on previously unseen inputs is a key goal in machine learning and is known as generalization.

***linear models & polynomial regression***
When we fit some data using a polynomial function of the form:
$$y(x, \boldsymbol{w}) = w_0 + w_1x + w_2x^2 + ... + w_Mx^M = \sum_{j=0}^M w_jx^j$$

, where $M$ is the *order* of the polynomial, and $x_j$ denotes $x$ raised to the power of $j$.
Although the polynomial function $y(x, \boldsymbol{w})$ is a nonlinear function of $x$, it is a linear function of the coefficients $\boldsymbol{w}$.

Functions, such as this polynomial, that are linear in the unknown parameters have important properties, as well as significant limitations, and are called linear models.

***error function***
The values of the coefficients $\boldsymbol{w}$ will be determined by fitting $y(x, \boldsymbol{w})$ to the training data. This can be done by *minimizing* an error function that measures the misfit between the fuction $y$, for any given value of $w$, and the training set data points.

***model complexity***
There remains the problem of choosing the order $M$ of the polynomial $y(x, \boldsymbol{w})$, and as we will see this will turn out to be an example of an important concept called *model comparison* or *model selection*.
Our goal is to achieve good *generalization* by making accurate predictions for *new data*. We can obtain some quantitative insight into the dependence of the generalization performance on $M$ by considering a separate set of data known as a *test set*.

***epistemic uncertainty*** or ***model uncertainty*** We will not be able to perfectly predict the exact output given the input, due to lack of knowledge of the input-output mapping.

***aleatoric uncertainty*** or ***data uncertainty*** we will not be able to perfectly predict the exact output given the input, due to intrinsic (irreducible) stochasticity in the mapping.

***logistic regression***
When $f$ is an *affine function* of the form
$$y(\boldsymbol{x};\boldsymbol{\theta})=b+\boldsymbol{w}^\mathsf{T}\boldsymbol{x}=b+w_1x_1+w_2x_2+\cdots+w_Dx_D$$
where $\boldsymbol{\theta} = (b, \boldsymbol{w})$ are the parameters of the model, this model is called logistic regression.
In statistics, the $\boldsymbol{w}$ parameters are usually called regression coefficients (and are typically denoted by $\beta$ ) and $b$ is called the intercept. In ML, the parameters $\boldsymbol{w}$ are called the *weights* and $b$ is called the *bias*.

***Maximum likelihood estimation***
The intuition is that a good model (with low loss) is one that assigns a high probability to the true output $y$ for each corresponding input $\boldsymbol{x}$. The average negative log probability of the training set is given by

$$\text{NLL}(\boldsymbol{\theta}) = -\frac{1}{N}\sum_{i=1}^N\log p(y_i|\boldsymbol{x}_i;\boldsymbol{\theta})$$
.

This is called the *negative log likelihood*. If we minimize this, we can compute the *maximum likelihood estimate* or *MLE*:

$$\hat{\boldsymbol{\theta}} = \arg\min_{\boldsymbol{\theta}}\text{NLL}(\boldsymbol{\theta})$$

This is a very common way to fit models to data.

***Regression***
Now suppose that we want to predict a real-valued quantity $y\in\mathbb{R}$ instead of a class label $y \in \{1, \ldots, C\}$; this is known as *regression*.


***Deep neural networks***
We can create much more powerful models by learning to do nonlinear *feature extraction* (for example, $\boldsymbol{\phi}(\boldsymbol{x})=[1,x_1,x_2,x_1^2,x_2^2,\ldots]$ ) automatically.

If we let $\boldsymbol{\phi}(\boldsymbol{x})$ have its own set of parameters, say $\mathbf{V}$, then the overall model has the form

$$y(\boldsymbol{x};\boldsymbol{w},\mathbf{V})=\boldsymbol{w}^\mathsf{T}\boldsymbol{\phi}(\boldsymbol{x};\mathbf{V})$$

We can *recursively decompose* the feature extractor $\boldsymbol{\phi}(\boldsymbol{x};\mathbf{V})$ into a composition of simpler functions.
The resulting model then becomes a stack of $L$ *nested functions*:

$$y(x;\theta)=y_L(y_{L-1}(\cdots (y_1(\boldsymbol{x})) \cdots ))$$

where $y_\ell(\boldsymbol{x})=y(\boldsymbol{x};\boldsymbol{\theta}_\ell)$ is the function at layer $\ell$.

This is the key idea behind deep neural networks or DNNs, which includes common variants such as convolutional neural networks (CNNs) for images, and recurrent neural networks (RNNs) for sequences.

# Reference
C. M. Bishop, H. Bishop, Deep Learning, https://doi.org/10.1007/978-3-031-45468-4_1

Probabilistic Machine Learning: An Introduction”. Online version. November 23, 2024