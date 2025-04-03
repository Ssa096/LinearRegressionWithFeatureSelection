# The problem of linear regression with feature selection, solved by first order methods.

We consider linear regression model:

$$Y = X\beta + \epsilon$$

, where:

* $Y \in R^n$ - vector of response variable.
* $X \in R ^ {n \times p}$ - model of observations of exogenous variables.
* $\beta \in R^p$ - vector of model coefficients.
* $\epsilon \in R^n$ - vector of errors.

We will assume that exogenous variables have been standarized to have zero means and unit Euclidean norm. The goal of
this project is to find $\beta$ vector that minimize below function:

$$\frac{1}{2}||y-X\beta||^2_2$$

subject to:

$$||\beta||_0 \leq k$$

,where $||\beta||_0$ denotes number of elements in vector $\beta$ not equal to 0. Therefore, the presented problem can
be understood as fitting the best possible linear regression number with number of feature selected being not higher
than $k$. This problem will be solved with discrete first-order algorithms.

Let's suppose that the problem is finidng $\beta$ that minimize the function below:

$$g(\beta)$$

subject to:
$$||\beta||_0 \leq k$$

where $g(\beta) \geq 0$ is a convex function which is Lipschitz continuous (there is a real constant $l > 0$ such
that):

$$|g(x_1) - g(x_2)| \leq l|x_1 - x_2|$$

for all real values of $x_1$ and $x_2$.

When $g(\beta) = ||\beta - c||_2^2$ for a given $c$, then beta can be computed as follows: $\beta_i = c_i$ if $c_i$ is
in $k$ highest(in terms of absolute value) elements of $c$, otherwise $\beta_i = 0$. Let's assume $H_k(c)$ - set of
solutions for this problem and $L > l$ - parameter of the optimizer. From that the following algorithm can be proposed:

### Algorithm 1

Input: $g(\beta)$, $L$ parameter and $\epsilon$ - convergance tolerance ($10e^{-4}$ is a good value to start with).

Output: First-order stationary solution $\beta^*$.

1. Initialize with $\beta_1 \in R^p$ such that $||\beta_1||_0 \leq k$.
2. For $m \geq 1$:
   $$\beta_{m+1} \in H_k(\beta_m - \frac{1}{L} \nabla g(\beta_m))$$
3. Repeat step 2 until $g(\beta_m) - g(\beta_{m+1}) \leq \epsilon$.

What is worth mentioning is that this algorithm needs just a few iterations to detect the final set of exogenous
features, the following iterations results only in fitting the appropriate values of coefficients for these features.
Therefore, stricter early stopping followed by more classical approach for selected features (like least squares) can
be considered.

### Algorithm 2

The alternative for algorithm 1. Input parameters, output and first step are the same, second step is quite different:

$$\beta_{m+1} = \lambda_m \eta_m + (1 - \lambda_m)\beta_m$$

where:

$$\eta_m \in H_k(\beta_m - \frac{1}{L} \nabla g(\beta_m))$$

and:

$$\lambda_m = arg min_\lambda g(\lambda \eta_m + (1 - \lambda)\beta_m)$$

Stopping criteria differs too - it is either $|g(\eta_{m+1}) - g(\eta_m)| \leq \epsilon$ or total number of iterations
is set at the beginning and after them $\eta_m$ which gave the lowest value of $g$ is returned. In addition, after
finishing algorithm 2, algorithm 1 with $\eta_m$ as starting point can be run - it usually takes a few iterations to
polish the result.

As far as linear regression is concerned:

* $g(\beta) = \frac{1}{2}||y-X\beta||^2_2$
* $\nabla g(\beta) = -X^{'}(y-X\beta)$
* $l$ - the highest eigenvalue of $X^{'}X$

### Experimental data

For the sake of experiments, both synthetic and real data will be used.

### Synthetic data

* $x_i \sim \mathcal{N}(0, \Sigma)$ be the i-th column of $X$ matrix (being standarized to have a unit Euclidean
  norm). All columns are independent.
* $\Sigma := (\sigma_{ij})$ - covariance matrix.
* $\epsilon $ - vector of i.i.d. errors such that $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$.
* $k_0$ - number of nonzeros in $\beta_0$.
* SNR (Signal-to-Noise Ratio): SNR $= \frac{var(X^{'}\beta_0)}{\sigma^2}$

The following examples are considered:

1. $\sigma_{ij} = \rho^{|i-j|}$, $k_0 \in \{5, 10\}$, $\beta^0_i = 1$ for $k_0$ equispaced values (rounding to nearest
   integer if necessary). $\rho$ is another parameter to explore ($\rho \in \{0.5, 0.8, 0.9\}$ is a good place to
   start).
2. $\Sigma = I, k_0 = 5, \beta_0 = (1^{'}_{5 \times 1}, 0^{'}_{p-5 \times 1}) \in R^p$
3. $\Sigma = I, k_0=10, \beta_i^0 = \frac{1}{2} + (10 - \frac{1}{2})\frac{i-1}{k_0}, i=1,...,10, \beta_i^0 = 0, i > 10 $
4. $\Sigma = I, k_0=6, \beta^0 = (-10, -6, -2, 2, 6, 10, 0^{'}_{p-6})$

### Real data

* Diabetes data[1] - avaible at sklearn. Orignally it consists of 10 response variables, but it will be transformed to
  64[2] - 9 of them will be squared, already existing, features and the remaining 45 will be interactions between the
  original set of features. Then 350 random samples will be chosen and transform so that every feature would be of zeros
  means and having Euclidean Norm equal to 1.
* Leukemia data[3] - the preprocessing process here is a little bit more complex:
    * All variables are standarized (zero means and unit Euclidean norm).
    * 1000 features with highest (in terms of absolute value) correlation with response variable are selected.
    * Semisynthetic y is generated: $y = X\beta^0 + \epsilon$, where:
        * $\beta^0_i = 1$ if $i \leq 5$ else 0
        * $\epsilon \sim \mathcal{N}(0, \sigma^2)$ - sigma is chosen so SNR=7.

### References

1. Diabetes data import
   documentation - https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html
2. Bradley Efron. Trevor Hastie. Iain Johnstone. Robert Tibshirani. "Least angle regression." Ann. Statist. 32 (2) 407 -
   499, April 2004. https://doi.org/10.1214/009053604000000067
3. Hameed, Shilan S.; Hassan, Rohayanti; Hassan, Wan Haslina; Muhammadsharif, Fahmi F.; Latiff, Liza Abdul (2021). The
   microarray dataset of leukemia cancer in csv format.. PLOS ONE.
   Dataset. https://doi.org/10.1371/journal.pone.0246039.s001
