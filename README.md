# AGI-Glossary

***generalization*** The ability to make accurate predictions on previously unseen inputs is a key goal in machine learning and is known as generalization.

***linear models***
When we fit some data using a polynomial function of the form:
$$y(x, \mathbf{w}) = w_0 + w_1x + w_2x^2 + ... + w_Mx^M = \sum_{j=0}^M w_jx^j$$

, where $M$ is the *order* of the polynomial, and $x_j$ denotes $x$ raised to the power of $j$.
Although the polynomial function $y(x, \mathbf{w})$ is a nonlinear function of $x$, it is a linear function of the coefficients $\mathbf{w}$.

Functions, such as this polynomial, that are linear in the unknown parameters have important properties, as well as significant limitations, and are called linear models.

***error function***
The values of the coefficients $\mathbf{w}$ will be determined by fitting $y(x, \mathbf{w})$ to the training data. This can be done by *minimizing* an error function that measures the misfit between the fuction $y$, for any given value of $w$, and the training set data points.

***model complexity***
There remains the problem of choosing the order $M$ of the polynomial $y(x, \mathbf{w})$, and as we will see this will turn out to be an example of an important concept called *model comparison* or *model selection*.
Our goal is to achieve good *generalization* by making accurate predictions for *new data*. We can obtain some quantitative insight into the dependence of the generalization performance on $M$ by considering a separate set of data known as a *test set*.

# Reference
C. M. Bishop, H. Bishop, Deep Learning, https://doi.org/10.1007/978-3-031-45468-4_1