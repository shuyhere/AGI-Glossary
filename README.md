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

***maximum likelihood estimation***
The intuition is that a good model (with low loss) is one that assigns a high probability to the true output $y$ for each corresponding input $\boldsymbol{x}$. The average negative log probability of the training set is given by

$$\text{NLL}(\boldsymbol{\theta}) = -\frac{1}{N}\sum_{i=1}^N\log p(y_i|\boldsymbol{x}_i;\boldsymbol{\theta})$$
.

This is called the *negative log likelihood*. If we minimize this, we can compute the *maximum likelihood estimate* or *MLE*:

$$\hat{\boldsymbol{\theta}} = \arg\min_{\boldsymbol{\theta}}\text{NLL}(\boldsymbol{\theta})$$

This is a very common way to fit models to data.

***regression***
Now suppose that we want to predict a real-valued quantity $y\in\mathbb{R}$ instead of a class label $y \in \{1, \ldots, C\}$; this is known as *regression*.


***deep neural networks***
We can create much more powerful models by learning to do nonlinear *feature extraction* (for example, $\boldsymbol{\phi}(\boldsymbol{x})=[1,x_1,x_2,x_1^2,x_2^2,\ldots]$ ) automatically.

If we let $\boldsymbol{\phi}(\boldsymbol{x})$ have its own set of parameters, say $\mathbf{V}$, then the overall model has the form

$$y(\boldsymbol{x};\boldsymbol{w},\mathbf{V})=\boldsymbol{w}^\mathsf{T}\boldsymbol{\phi}(\boldsymbol{x};\mathbf{V})$$

We can *recursively decompose* the feature extractor $\boldsymbol{\phi}(\boldsymbol{x};\mathbf{V})$ into a composition of simpler functions.
The resulting model then becomes a stack of $L$ *nested functions*:

$$y(x;\theta)=y_L(y_{L-1}(\cdots (y_1(\boldsymbol{x})) \cdots ))$$

where $y_\ell(\boldsymbol{x})=y(\boldsymbol{x};\boldsymbol{\theta}_\ell)$ is the function at layer $\ell$.

This is the key idea behind deep neural networks or DNNs, which includes common variants such as convolutional neural networks (CNNs) for images, and recurrent neural networks (RNNs) for sequences.

***no free lunch theorem***
There is no single best model that works optimally for all kinds of problems. The reason is that a set of assumptions (also called inductive bias) that works well in one domain may work poorly in another.

***unsupervised learning***
From a probabilistic perspective, we can view the task of unsupervised learning as fitting an unconditional model of the form $p(\boldsymbol{x})$, which can generate new data $\boldsymbol{x}$, whereas supervised learning involves fitting a conditional model, $p(\boldsymbol{y}|\boldsymbol{x})$, which specifies (a distribution over) outputs given inputs.

Unsupervised learning forces the model to “explain” the *high-dimensional* inputs, rather than just the low-dimensional outputs. This allows us to learn richer models of “how the world works”.

***clustering***
A simple example of unsupervised learning is the problem of finding clusters in data. The goal is to partition the input into regions that contain “similar” points.

***self-supervised learning***
A recently popular approach to unsupervised learning is known as self-supervised learning. In this approach, we create *proxy supervised* tasks from unlabeled data. This avoids the hard problem of trying to infer the “true latent factors” $\mathbb{z}$ behind the observed data.

***evaluating unsupervised learning***
It is very hard to evaluate the quality of the output of an unsupervised learning method, because there is *no ground truth* to compare to. A common method for evaluating unsupervised models is to measure the probability assigned by the model to unseen test examples.
We can do this by computing the *(unconditional) negative log likelihood* of the data:
$$\mathcal{L}(\boldsymbol{\theta};\mathcal{D})=-\frac{1}{|\mathcal{D}|}\sum_{\boldsymbol{x}\in\mathcal{D}}\log p(\boldsymbol{x}|\boldsymbol{\theta})$$

This treats the problem of unsupervised learning as one of *density estimation*. The idea is that a good model will not be “surprised” by actual data samples (i.e., will assign them high probability).
Thus the model has learned to capture the *typical patterns* in the data. This can be used inside of a *data compression* algorithm.

An alternative evaluation metric is to use the learned unsupervised representation as features or input to a downstream supervised learning method.

We can increase the sample efficiency of learning (i.e., reduce the number of labeled examples needed to get good performance) by first learning a good representation.

Increased sample efficiency is a useful evaluation metric, but in many applications, especially in science, the goal of unsupervised learning is to gain understanding, not to improve performance on some prediction task. This requires the use of models that are **interpretable**, but which can also generate or “explain” most of the observed patterns in the data. To paraphrase Plato, the goal is to discover how to “carve nature at its joints”. Of course, evaluating whether we have successfully discovered the true underlying structure behind some dataset often requires performing experiments and thus interacting with the world.

**reinforcement learning (RL)**
The system or agent has to learn how to interact with its environment.
This can be encoded by means of a *policy* $\boldsymbol{a}=\pi(\boldsymbol{x})$, which specifies which action to take in response to each possible input $\boldsymbol{x}$ (derived from the environment state).

**one-hot encoding or dummy encoding** 
If a variable $x$ has $K$ possible values, we will denote its dummy encoding as follows: 
$$\text{one-hot}(x) = [\mathbb{I}(x = 1), \ldots, \mathbb{I}(x = K)]$$

**bag of words model**
A simple approach to dealing with variable-length text documents is to interpret them as *a bag of words*, in which we ignore word order. To convert this to a vector from a fixed input space, we first *map each word to a token from some vocabulary*.
* **stop word removal** dropping punctuation, converting all words to lower case; dropping common but uninformative words, such as “and” and “the”
* **stemming** replacing words with their base form, such as replacing “running” and “runs” with “run”

**TF-IDF**
One problem with representing documents as word count vectors is that frequent words may have undue influence, just *because the magnitude of their word count is higher, even if they do not carry much semantic content.*

To reduce the impact of words that occur many times in general (across all documents), we compute a quantity called the inverse document frequency, or **IDF**:
$\mathrm{IDF}_i\triangleq\log\frac{N}{1+\mathrm{DF}_i}$, where $\mathrm{DF}_i$ is the number of documents that contain word $i$ and $N$ is the total number of documents.

We can combine these transformations to compute the TF-IDF matrix as follows:
$$\mathrm{TFIDF}_{ij}=\mathrm{TF}_{ij}\times\mathrm{IDF}_i$$
, where $\mathrm{TF}_{ij}$ is the frequency of word $i$ in document $j$.

**word embeddings**
TFIDF does not solve the fundamental problem with the one-hot encoding (from which count vectors are derived), which is that that semantically similar words, such as “man” and “woman”, may be not be any closer (in vector space) than semantically dissimilar words, such as “man” and “banana”.
The standard way to solve this problem is to use word embeddings, in which we map each sparse one-hot vector, $x_{nt} \in \{0,1\}^{V}$ ,to a lower-dimensional dense vector $\mathbf{e}_{nt} \in \mathbb{R}^K$, using $\mathbf{e}_{nt} = \mathbf{E}x_{nt}$, where $\mathbf{E} \in \mathbb{R}^{V \times K}$ is learned such that *semantically similar words are placed close by.*

**out of vocabulary or OOV**
At test time, the model may encounter a completely novel word that it has not seen before.
It is better to leverage the fact that words have substructure, and then to take as input **subword units** or **wordpieces**; these are often created using a method called **byte-pair encoding**.
# Reference
C. M. Bishop, H. Bishop, Deep Learning, https://doi.org/10.1007/978-3-031-45468-4_1

Probabilistic Machine Learning: An Introduction”. Online version. November 23, 2024

