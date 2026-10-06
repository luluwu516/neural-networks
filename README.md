# Neural Network

The notebook trains two classifiers on the MNIST handwritten-digit dataset:

1. Softmax regression: a single linear layer followed by softmax

2. Two-layer neural network: a hidden layer of 128 sigmoid neurons, then a softmax output layer

I implemented both models from scratch in NumPy to strengthen my understanding of neural networks.

## Activation Functions

A network made **only** of linear layers has the same representational power as a single linear layer, stripping it of its ability to learn complex representations. 

$$ w_1^T(w_2^T(w_3^Tx)) = w_4^Tx$$

Non-linear activation functions are essential in neural networks. They let compositions of layers represent genuinely complex functions. 

Here are some classical **non-Linear activation functions**: 

### Sigmoid Function

  $$ \sigma(x)=\frac{1}{1+e^{-x}} $$
  $$ \frac{d}{dx} \sigma = \sigma'(x)=\sigma(x) (1 - \sigma(x)) $$

  * Range between $0$ and $1$
  * **The Vanishing Gradient Problem**: For large positive or small negative inputs, the sigmoid curve flattens out into saturated regimes, and the local derivative is virtually zero. During backpropagation, the upstream gradient is multiplied by the near-zero local gradient, causing the downstream gradient vanish. The effect kills the gradients, stopping learning
  * **Not Zero-Centered**: Output are all positive, forcing the weight gradients to be either all positive or all negative for a single example, which leads to inefficient, zig-zagging updates during gradient descent
  * **High Computational Cost**: Calculating the exponential function is relatively expensive

---

### Tanh Function

$$ \tanh(x) = \frac{\sinh(x)}{\cosh(x)} = \frac{e^x - e^{-x}}{e^x + e^{-x}} $$
$$ \frac{d}{dx} \tanh(x) = \tanh'(x) = \text{sech}^2(x) = 1 - \tanh^2(x) $$

  * Range between $-1$ and $1$
  * **Zero-Centered Advantage**: Tanh outputs are zero-centered, which improves training optimization dynamics
  * **The Vanishing Gradient Problem**: Tanh still flattens out at extreme negative or positive input values. As a result, it suffers from the same saturating flat regimes that cause vanishing gradients during backpropagation

---

### Rectified Linear Unit (ReLU) Function

$$ \text{ReLU}(x) = \max(0,x) $$
$$ \frac{d}{d x}\text{ReLU}(x) = \text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x < 0 \end{cases} $$

  * Range between $0$ and $\infty$
  * **No Saturation** (in the positive regime): It does not saturate for positive values, it prevents local gradients from vanishing
  * Output always positive
  * **Efficiency**: It requires only a simple threshold operation, making it extremely fast to compute
  * **Fast Converge**: converges much faster than sigmoid/tanh in practice 
  * **The "Dead ReLU" Problem**: For negative inputs, the gradient is identically zero. If a neuron's weights get updated such that it ouputs negative values across the entire training dataset, its gradient remains zero permanently, rendering it a "dead ReLU" that never learns again

---

### Leaky ReLU

$$ \text{Leaky ReLU}(x) = \max(0.01x, x) $$
$$ \frac{d}{d x}\text{Leaky ReLU}(x) = \text{Leaky ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0.01 & \text{if } x < 0 \end{cases} $$

  * Introduces a small positive slope (e.g., 0.01) in the negative regime
  * Range between $-\infty$ and $\infty$
  * **No Saturation** (on both ends): This ensures local gradients never reach zero, preventing neurons from dying

---

### Parametric ReLU (PReLU)

$$ \text{PReLU}(x) = \max(\alpha x, x)$$
$$ \frac{d}{d x}\text{PReLU}(x) = \text{PReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ \alpha & \text{if } x < 0 \end{cases} $$

  * Makes the negative slope a learnable parameter($\alpha$) that updates automatically via backpropagation during training.
  * Range between -∞ and ∞
  * Learnable parameter
  * **No saturation**
  * **No dead neuron**
  * Still cheap to compute

---

### Exponential Linear Unit (ELU)

$$ \text{ELU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha (e^x - 1) & \text{if } x \le 0 \end{cases} $$
$$ \frac{d}{d x}\text{ELU}(x) = \text{ELU}'(x) = \begin{cases} 1 & \text{if } x > 0 \alpha e^x & \text{if } x \le 0  \end{cases} $$

  * Uses an exponential curve in the negative regime that smoothly asymptotes to negative value
  * Range between $-\alpha$ and $\infty$
  * Soft/asymptotic saturation on the negative side only (bounded, smoothly approaches $-\alpha$); no hard zero-gradient region, so neurons don't die, but gradients do shrink for very negative inputs
  * **No Dead Neuron**
  * **Better Zero-Centered**
  * **Higher Computational Cost**: Calculating the exponential function is relatively expensive

---

### Scaled ELU (SELU)

$$ \text{SELU}(x) = \lambda \begin{cases} x & \text{if } x > 0 \\ \alpha (e^x - 1) & \text{if } x \le 0  \end{cases}$$
$$ \frac{d}{d x}\text{SELU}(x) =\text{SELU}'(x) = \lambda \begin{cases} 1 & \text{if } x > 0 \\ \alpha e^x & \text{if } x < 0 \end{cases} $$

  * Range between $-1.758$ and $\infty$
  * **Self-Normalization** (under LeCun-normal weight initialization and a plain feedforward architecture): Activations automatically drive toward a mean of zero and a standard deviation of one across deep layers, removing the need for extra batch normalization layers
  * Soft/asymptotic saturation on the negative side only (bounded, smoothly approaches $-\alpha$); no hard zero-gradient region, so neurons don't die, but gradients do shrink for very negative inputs
  * **No Dead Neuron**

---

### Gaussian Error Linear Unit (GELU)
  
$$ \text{GELU}(x) = 0.5x \left(1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right) $$
$$ \frac{d}{dx}\text{GELU}(x) = \text{GELU}'(x) = \Phi(x) + x\phi(x) $$

  * Range between $-0.17$ and $\infty$
  * **Probabilistic gating**: Instead of gating inputs by their sign, GELU weights an input by its value, scaling it probabilistically based on its magnitude
  * **Non-monotonic Function**: It has a smooth, non-monotonic curve that lets small negative inputs pass through with a small negative value instead of zeroing them out
  * **No Dead Neuron**
  * **Transformer Standard**: Used in Transformers

---

### Sigmoid Linear Unit (SiLU) / Swish
  
$$ \text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}} $$
$$ \text{SiLU}'(x) = \text{SiLU}(x) + \sigma(x)(1 - \text{SiLU}(x)) $$

  * Range between $-0.27846$ and $\infty$
  * **Non-monotonic Function**: SiLU is smooth, continuously differentiable everywhere, and features a small negative dip near zero before rising monotonically for positive values
  * **Self-Gated**: It multiplies the input by a sigmoid gate bounded between 0 and 1, modulating the magnitude of the information passing through
  * **No Dead Neuron**

---

### Activation Functions Summary Table

| Function | Range | Zero-centered? | Saturates? | Gradient issue | Cost |
|---|---|---|---|---|---|
| **Sigmoid** | (0, 1) | No | Both ends (hard) | Vanishes at both ends; gradient always positive | Expensive (exponential) |
| **Tanh** | (−1, 1) | Yes | Both ends (hard) | Vanishes at both ends | Still fairly expensive |
| **ReLU** | [0, ∞) | No | Negative side (hard) | Gradient is exactly 0 for $x\le 0$ ("dead ReLU"); constant (non-vanishing) for $x>0$ | Very cheap (a max op) |
| **Leaky ReLU** | (−∞, ∞) | Roughly | No | No dead neurons; gradient never vanishes | Cheap |
| **PReLU** | (−∞, ∞) | Roughly | No | No dead neurons; gradient never vanishes | Cheap |
| **ELU** | ($-\alpha$, ∞) | Roughly | Negative side (soft/asymptotic)† | Gradient → 0 as $x\to-\infty$, but never exactly 0; no dead neuron | Expensive (exponential) |
| **SELU** | (−1.758, ∞) | Roughly | Negative side (soft/asymptotic)† | Gradient → 0 as $x\to-\infty$, but never exactly 0; no dead neuron | Moderately expensive |
| **GELU** | [−0.17, ∞) | Roughly | Negative side (soft/asymptotic)† | Gradient → 0 as $x\to-\infty$, but never exactly 0; no dead neuron | Moderately expensive |
| **SiLU** | [−0.278, ∞) | Roughly | Negative side (soft/asymptotic)† | Gradient → 0 as $x\to-\infty$, but never exactly 0; no dead neuron | Moderately expensive |

†No hard zero-gradient region (so no permanently dead neuron), but gradients do shrink asymptotically for very negative inputs.

> ### Takeaway:
>  - **Avoid Sigmoid and Tanh** in hidden layers of deep networks because their saturated regimes lead to severe vanishing gradient issues.
>  - **Default to ReLU**: It is simple, computationally lightweight, and work well across most standard architectures.
>  - **Try variants for that last 1% of accuracy**. If dead neurons or minor performance bottlenecks occur, switching to Leaky ReLU, ELU, or GELU can offer small incremental gains.


## Initialization

Initialization determines the statistics of activations at the start of training, which in turn determine how well gradients flow and whether the model can use its full capacity. Good initialization gives faster, better convergence, whereas poor initialization causes vanishing or exploding activations/gradients before learning even begins.

**Goal**: Keep the variance of activations (forward pass) and gradients (backward) roughly constant across layers. 

> **Notation**: $\mathcal{N}(0,\sigma^2)$ denotes a normal distribution with variance $\sigma^2$ (std $\sigma$). $n_j$ = fan-in, $n_{j+1}$ = fan-out. Biases are typically initialized to zero.

---

### Constant Initialization
  
$$ w_i = c\ \forall i$$

- **Degenerate**: Every node in a layer receives the same input and computes the same gradients, so all weights update in the same way, preventing the network from breaking symmetry

- All zeros is the special case $c=0$; It is even worse than a nonzero constant: backprop multiplies by the (zero) downstream weights, so earlier layers receive zero gradient, and with ReLU/tanh the hidden activations are also zero, so even the last layer's weights get zero gradient (only biases update)

---

### Small Gaussian / Normal Initialization (Used in the project)

$$ W \sim \mathcal{N}(0,\sigma^2),\quad \sigma = 0.01 $$

- **Small random values** (e.g. $\sigma = 0.01$): No feature has prior importance, and the network remains in the near-linear region of activations

- Fine for shallow networks, but in deep networks the std of activations shrinks layer by layer, giving vanishing updates

- Larger values (e.g. $\sigma=0.05$) can saturate tanh/sigmoid immediately or explode activations in deep ReLU stacks. Whether a given $\sigma$ is "too large" depends on width and depth: pre-activation std is roughly $\sigma\sqrt{n_j}$.

---

### Xavier/Glorot Initialization

$$ W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_j+n_{j+1}}},\ +\sqrt{\frac{6}{n_j+n_{j+1}}}\right) \quad\Longleftrightarrow\quad \text{Var}(W)=\frac{2}{n_j+n_{j+1}} $$

- Chooses the weight scale so that the variance of activations is preserved across layers
- **Assumptions:** zero-mean, independent inputs and weights, and an activation that is approximately linear near zero (i.e. tanh-like). Under these, $\text{Var}(Y)=\text{Var}(X)$ requires $\text{Var}(W)=1/n_j$
- Preserving the *backward* gradient variance instead requires $\text{Var}(W)=1/n_{j+1}$. Xavier compromises between the two, which is why both fan-in and fan-out appear
- For CNNs: $\text{fan}_{in}=C_{in}\times k^2$ and $\text{fan}_{out}=C_{out}\times k^2$ (square kernels)
- Not suited to ReLU: it shrinks activation variance by roughly half per layer

---

### LeCun-normal / Simpler Xavier Initialization

$$ W \sim \mathcal{N}\left(0,\ \sigma^2=\frac{1}{n_j}\right) $$

- Fan-in-only version that preserves forward variance; performs comparably to Xavier in practice

- Rrequired for SELU's self-normalizing property

---

### MSRA / Kaiming / He / Xavier2 Initialization

$$ W \sim \mathcal{N}\left(0,\ \sigma^2=\frac{2}{n_j}\right) $$

- Same variance-preservation idea, re-derived for **ReLU**: ReLU zeroes roughly half the pre-activations, halving the second moment, so the weight variance is doubled ($\text{Var}(W)=2/n_j$, i.e. std multiplied by $\sqrt{2}$) to compensate

- **Caveat for ResNets:** a residual block computes $x+F(x)$, so with MSRA init each block roughly doubles the activation variance, giving exponential growth ($\sim 2^L$) with depth. Standard fix: initialize the inner layers of each block with MSRA and set the final layer of the block to zero (with BatchNorm, zero the last BN $\gamma$), so each block starts as an identity mapping

---

## Normalization and Data Preprocessing

### Normalization

### Principle Component Analysis (PCA)

## Optimizers


## Regularization


## Data Augmentation

