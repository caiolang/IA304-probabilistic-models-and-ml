# Class 2: Real World Graphical Models Applications
*Frédéric Lehmann*

## Comments to the slides

> Slide number

Comment

### GOT HERE

Coding: adding symbols to the information

Transmitter

Receiver

### Binary Additive White Gaussian Noise Channel (BAWGNC)
- $E_s$ : Energy per symbol
- $E_b=\frac{n}{k}E_s$ : Energy per trasmitted symbol
  - $\frac{n}{k}$ : $\frac{\text{total number of symbols}}{\text{information symbols}}$
  - $n-k$ : redundant bits

The energy to me transmitted must not increase with the redundant symbols

> 15

- $x$ : output of encoder
- $y$ : output of channel (observer by receptor)

Max log-likelihood *(log-maximum de vraisemblence)*
- Maximize the log of the probability of observing an output of channel given the output of the encoder
- Use this result to choose x ???

> 16

> 17

Legend
- $R=k/n$ : Entropy ratio of the source
- $BER$: error
- $E_b/N_0$: SNR (signal to noise ratio)
 
Interpretation
- $R=1$ : no encoding
- $R<1$ : encoding, so there is a minimal SNR in order to make the error tend to zero.

> 18 - 21

## ???


> 25

Factor graph
> [See this StackExchange thread](https://stats.stackexchange.com/questions/85453/why-use-factor-graph-for-bayesian-inference)

hidden variables

factoring of the 

We trace an arc between a variable node and a function node if such a node is a parameter of the function

yi : observed variables, they are fixed parameters (not variables). They are related to a particular realization.

The function nodes (in black) are leafs.

We can apply an algorithm to solve the factor graph
- Caveat: when applying the sum-product algo, we obtain the a posteriori law for the $x_i$ exactly iff the graph has no cycle. Else, we know it converges and we end up with, at least, a good approximation of the law.
- Cycle: more than one way to connect two function nodes
  - In this case, there are cycles such as:
    - $f_A \rarr x_3 \rarr f_C$
    - $f_A \rarr x_5 \rarr f_C$

> 26

Example of the **Sum-Product algo** (presented on previous class):
- Apply the product rule to x_3
  - Get the product of all messages in blue

> 29

We cluster some nodes of the Factor Graph, creating an equivalent graph that is acyclic

We want to minimize the clusterization, while still having an equivalent function

Why minimize? Because the more we cluster, the more expensive becomes the summing on the cluster (???) when applying the Sum-product algorithm

$$P(x) \propto f_A'(x_1, x_2, x_3, x_4, x_5)\:f_B'(x2, x_3, x_4, x_5, x_6)\:f_C'(x2, x_3, x_4, x_5, x_7)$$

$$\forall x_4 \in{0,1 }f_A(x_1, x_2, x_3, x_5) = f_A'(x1, x_2, x_3, x_4, x_5)$$
$$\forall x_5 \in{0,1 }f_B(x_2, x_3, x_4, x_6) = f_B'(x2, x_3, x_4, x_5, x_6)$$
$$\forall x_2 \in{0,1 }f_C(x_3, x_4, x_5, x_7) = f_C'(x2, x_3, x_4, x_5, x_7)$$


# Modern coding technology
> 33-35

## Low-Density Parity-Check (LDPC)

- Matrix H is sparse and picked at random

# Speech recognition

> 41

Feature extraction

Brown box: compensate filtering made by the human body

To avoid artifacts, we window the signal before sampling (e.g. Hamming window)

MFCC

> 46

HMC for phoneme identification

Applying the sum-product algo
- yi are observed parameters, so the black functions are leafs
- we start on p(s0), calculating the alfas (forward messages) and the betas (backward messages). With this we obtain the a posteriori law, and by maximizing this law, we obtain the MAP.

For the min-sum algo
- apply it forwards (???)
- use backtracking technique (???)
- **This is the Vitterbi algorithm**

> 47 

Baum-Welch algotithm
- Forward-Backward to restore phonemes (by using the current observed gaussian mix) + EM algo (to re-evaluate the ???)

> 48

Ways to extend this HMC model

> 51

