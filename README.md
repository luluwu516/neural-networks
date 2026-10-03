# Neural Network




## Activation Functions

A network made **only** of linear layers has the same representational power as a single linear layer, stripping it of its ability to learn complex representations. 

$$ w_1^T(w_2^T(w_3^Tx)) = w_4^Tx$$

Non-linear activation functions are essential in neural networks. They let compositions of layers represent genuinely complex functions. 

Classical Non-Linear Activation Functions and the Vanishing Gradient Problem

* ### Sigmoid Function

  $$ \sigma(x)=\frac{1}{1+e^{-x}} $$
  $$ \frac{d}{dx} \sigma = \sigma'(x)=\sigma(x) (1 - \sigma(x)) $$

  * Range between $0$ and $1$
  * **The Vanishing Gradient Problem**: For large positive or small negative inputs, the sigmoid curve flattens out into saturated regimes, and the local derivative is virtually zero. During backpropagation, the upstream gradient is multiplied by the near-zero local gradient, causing the downstream gradient vanish. The effect kills the gradients, stopping learning.
  * **Not Zero-Centered**: Output are all positive, forcing the weight gradients to be either all positive or all negative for a single example, which leads to inefficient, zig-zagging updates during gradient descent.
  * **High Computational Cost**: Calculating the exponential function is relatively expensive.

* ### Tanh Function

  $$\tanh(x) = \frac{\sinh(x)}{\cosh(x)} = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$
  $$\frac{d}{dx} \tanh(x) = \tanh'(x) = \operatorname{sech}^2(x) = 1 - \tanh^2(x) $$

  * Range between $-1$ and $1$
  * **Zero-Centered Advantage**: Tanh outputs are zero-centered, which improves training optimization dynamics.
  * **The Vanishing Gradient Problem**: Tanh still flattens out at extreme negative or positive input values. As a result, it suffers from the same saturating flat regimes that cause vanishing gradients during backpropagation.

* ### Rectified Linear Unit (ReLU) Function

  $$\text{ReLU}(x) = \max(0,x)$$
  $$ \frac{d}{d x}\text{ReLU}(x) = \text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x < 0 \end{cases} $$

  * Range between $0$ and $\infty$
  * **No Saturation** (in the positive regime): It does not saturate for positive values, it prevents local gradients from vanishing.
  * Output always positive
  * **Efficiency**: It requires only a simple threshold operation, making it extremely fast to compute.
  * **Fast Converge**: converges much faster than sigmoid/tanh in practice. 
  * **The "Dead ReLU" Problem**: For negative inputs, the gradient is identically zero. If a neuron's weights get updated such that it ouputs negative values across the entire training dataset, its gradient remains zero permanently, rendering it a "dead ReLU" that never learns again.

* ### Leaky ReLU

  $$\text{Leaky ReLU}(x) = \max(0.01x, x)$$
  $$ \frac{d}{d x}\text{Leaky ReLU}(x) = \text{Leaky ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0.01 & \text{if } x < 0 \end{cases} $$

  * Introduces a small positive slope (e.g., 0.01) in the negative regime.
  * Range between $-\infty$ and $\infty$
  * **No Saturation** (on both ends): This ensures local gradients never reach zero, preventing neurons from dying.

* ### Parametric ReLU (PReLU)

  $$ \text{PReLU}(x) = \max(\alpha x, x)$$
  $$ \frac{d}{d x}\text{PReLU}(x) = \text{PReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ \alpha & \text{if } x < 0 \end{cases} $$

  * Makes the negative slope a learnable parameter($\alpha$) that updates automatically via backpropagation during training.
  * Range between -∞ and ∞
  * Learnable parameter
  * **No saturation**
  * **No dead neuron**
  * Still cheap to compute

* ### Exponential Linear Unit (ELU)

  $$ \text{ELU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha (e^x - 1) & \text{if } x \le 0 \end{cases} $$
  $$ \frac{d}{d x}\text{ELU}(x) = \text{ELU}'(x) = \begin{cases} 
      1 & \text{if } x > 0 \\
      \alpha e^x & \text{if } x \le 0 
   \end{cases} 
  $$

  * Uses an exponential curve in the negative regime that smoothly asymptotes to negative value.
  * Range between $-\alpha$ and $\infty$
  * Soft/asymptotic saturation on the negative side only (bounded, smoothly approaches $-\alpha$); no hard zero-gradient region, so neurons don't die, but gradients do shrink for very negative inputs.
  * **No Dead Neuron**
  * **Better Zero-Centered**
  * **Higher Computational Cost**: Calculating the exponential function is relatively expensive.

* ### Scaled ELU (SELU)

  $$ \text{SELU}(x) = \lambda \begin{cases} x & \text{if } x > 0 \\ \alpha (e^x - 1) & \text{if } x \le 0  \end{cases}$$
  $$ \frac{d}{d x}\text{SELU}(x) =\text{SELU}'(x) = \lambda \begin{cases} 1 & \text{if } x > 0 \\ \alpha e^x & \text{if } x < 0 \end{cases}$$

  * Range between $-1.758$ and $\infty$
  * **Self-Normalization** (under LeCun-normal weight initialization and a plain feedforward architecture): Activations automatically drive toward a mean of zero and a standard deviation of one across deep layers, removing the need for extra batch normalization layers.
  * Soft/asymptotic saturation on the negative side only (bounded, smoothly approaches $-\alpha$); no hard zero-gradient region, so neurons don't die, but gradients do shrink for very negative inputs.
  * **No Dead Neuron**

* ### Gaussian Error Linear Unit (GELU)
  
  $$ \text{GELU}(x) = 0.5x \left(1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right) $$
  $$\frac{d}{dx}\text{GELU}(x) = \text{GELU}'(x) = \Phi(x) + x\phi(x)$$

  * Range between $-0.17$ and $\infty$
  * **Probabilistic gating**: Instead of gating inputs by their sign, GELU weights an input by its value, scaling it probabilistically based on its magnitude.
  * **Non-monotonic Function**: It has a smooth, non-monotonic curve that lets small negative inputs pass through with a small negative value instead of zeroing them out.
  * **No Dead Neuron**
  * **Transformer Standard**: Used in Transformers

* ### Sigmoid Linear Unit (SiLU) / Swish
  
  $$ \text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}} $$
  $$ \text{SiLU}'(x) = \text{SiLU}(x) + \sigma(x)(1 - \text{SiLU}(x)) $$

  * Range between $-0.27846$ and $\infty$
  * **Non-monotonic Function**: SiLU is smooth, continuously differentiable everywhere, and features a small negative dip near zero before rising monotonically for positive values.
  * **Self-Gated**: It multiplies the input by a sigmoid gate bounded between 0 and 1, modulating the magnitude of the information passing through.
  * **No Dead Neuron**


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

> †No hard zero-gradient region (so no permanently dead neuron), but gradients do shrink asymptotically for very negative inputs.

> ### Takeaway:
>  - **Avoid Sigmoid and Tanh** in hidden layers of deep networks because their saturated regimes lead to severe vanishing gradient issues.
>  - **Default to ReLU**: It is simple, computationally lightweight, and work well across most standard architectures.
>  - **Try variants for that last 1% of accuracy**. If dead neurons or minor performance bottlenecks occur, switching to Leaky ReLU, ELU, or GELU can offer small incremental gains.


## Initialization

Initialization determines the statistics of activations and, therefore, gradients at the start of training, how well gradients flow, and whether the model can use its full capacity.

* ### Xavier Initialization

* ### Simpler Xavier Initialization

* ### MSRA (Kaiming) Initialization


[(Glorot and Bengio, 2010)](https://proceedings.mlr.press/v9/glorot10a.html)

## Data Preprocessing

### Normalization


## Optimizers


## Regularization


## Data Augmentation

