
Data: https://cs.nyu.edu/~roweis/data.html


- Stochastic NN
- We have kind of justified it by the universality theorem
- We have suggested a parameter estimation method (Gibbs Sampling)
- We can use it on other data (if they are discrete)
- For continuous data: Gaussian RBF

----

binary alpha digits
39 samples for each digit

20x16 images

```python

digits_to_learn = [10, 11]
# k = len(digits_to_learn)

def lire_alpha_digit(file : .mat, digits_to_learn, ):
    -> X (n=39.k x p=320)

    # Take a random line, convert it from 320 to 20x16 and see if it correctly shows a digit

```

RBM.a
RBM.a



$$
p(h|v) = \prod_j p(h_j|v)
$$
$$
p(v|h) = \prod_i p(v_i|h)
$$

we'll need some intermediary functions to 
v: in data


Obtain for all inputs, for one value of $h$

$$
p(h_{j=1}|v) = \frac{1}{1 + \exp(-(...))}
$$



## Pseudocode


- For each character, 39 samples
- q is hyperparameter


```python

# q is a hyperparameter
# p depends on data ?


def init_RBM(p,q):
    # RMB.a <- p
    RMB.a = np.zeros(p)
    # RMB.b <- q
    RMB.b = np.zeros(q)
    # RMB.w <- p x q
    # TODO

    # pay attention to the initialization of the matrix (use standard deviation, not covariance)
    RMB.a = np.zeros(p)

def in_out (X)
# Inputs X of size n x p
  # n: number of samples per batch
  # p: dim of one sample
# Note: the input is not necessarily the full data, since we can split it in batches for the gradient ascent
# Outputs n x q

  # H = 1/ (1 + exp( -(X @ self.w + self.b)) )
  # return H

def out_in (H)
# Inputs H of size n x q
# Outputs n x p

  # ????
  # X_lin = 1/ (1 + exp( -(H @ self.w.T + self.a)) )
  # return X_lin

```

```python
def train_RBM(X, size_batch, eps:0.1, nb_epoch)
# See algorithm below
# X: input data
# eps: learning rate
# size_batch: 
# nb_epoch: 

  for epoch in range(nb_epoch):
    # Random permutation so that the epoch is not the same at each time
    random.perm(X, axis=0)

    for batch in range(size(X,axis=0) , step = size_batch):
    
        # Create batch
        # If size_batch is not diviser of nb_epoch
        X_batch = X(batch : min(batch+size_batch, size(X,axis=0)), : )
        sz_batch = size(X_batch,0) # X_batch.shape[0]

        v_0 = X_batch # Size (sz_batch x p)

        # This is a proba matrix, independent lines and columns
        # (Each sample is independent)
        # We sample from a bernoulli
        p_h_v_0 = in_out(v_0) # Size (sz_batch x q)
        
        # Bernoulli sample
        h_0 = int(rand(sz_batch, q) < p_h_v_0) # Size (sz_batch x q)
        # (Boolean matrix casted to int)

        p_v_h_0 = out_in(h_0) # Size (sz_batch x p)

        v_1 = int(rand(sz_batch, p) < p_v_h_0) # Size (sz_batch x p)

        # This was the Gibbs Sampler iteration

        p_h_v_1 = in_out(v_1) # Size (sz_batch x q)


        # Now obtain the gradients

        # It does not depend on the batch size
        # If it were one sample, it would be (= v_0 - v_1)
        grad_a = sum(v_0 - v_1, axis=0) # Size: (p)

        grad_b = sum(p_h_v_0 - p_h_v_1, axis=0) # Size: (q)
        
        # If it were one sample, it would be (= v_0 - v_1)
        grad_w = (v_0.T @ p_h_v_0 - v_1.T @ p_h_v_1, axis=0) # Size: (p x q)

        # Update (We are maximizing the likelihood, so we SUM)
        self.a += eps/sz_batch*grad_a
        self.b += eps/sz_batch*grad_b
        self.w += eps/sz_batch*grad_w

    
    # We won't evaluate the loss function, because we wouldn't be able to calculate it
    # What we do instead is to obtain the reconstruction error
    H = in_out(X)
    X_rec = out_in(H)
    print( sum( (X - X_rec)**2 ) / size(X) )



```

## Algorithm

For one data $x$, parameter $\theta$:

- Current $\theta$
- $v^{(0)} = x$
- $h^{(0)} \sim p_\theta(h|x)$
- $v^{(1)} \sim p_\theta(v|h)$
> Parameters: One matrix of weights and two vectors of biases
- $\forall i,j$:
  - $\text{grad\_}w_{ij} = v^{(0)}_i p_\theta(h_{j=1}|v^{(0)}_i) - v^{(1)}_i p_\theta(h_{j=1}|v^{(1)})$
  - $\text{grad\_}a_{i} = v^{(0)}_i - v^{(1)}_i$
  - $\text{grad\_}b_{j} = p_\theta(h_{j=1}|v^{(0)}_i) - p_\theta(h_{j=1}|v^{(1)})$
  - $w_{ij} \leftarrow w_{ij} + \epsilon.\text{grad\_}w_{ij}$
  - $a_{i} \leftarrow a_{i} + \epsilon.\text{grad\_}a_{i}$
  - $b_{j} \leftarrow b_{j} + \epsilon.\text{grad\_}b_{j}$


```python

def generate_data(nb_data, nb_iter_gibbs, shape_img:tuple):
    """
    Generate data based on Gibbs sampling
    """
    
    p = len(self.a) # One option to fetch the dimension p
    q = len(self.b) # One option to fetch the dimension p

    for i in range(nb_data):
        # Initialize the Gibbs sampler
        v = (rand(p) < 1/2) * 1
        # We can also change the parameter of the Bernoulli at each time, to avoid bias in the initialization
        # v = (rand(p) < rand()) * 1

        h = (rand(q) < in_out(v)) * 1
        v = (rand(p) < out_in(h)) * 1

    # image = reshape(v, 20, 16)
    image = reshape(v, shape_img[0], shape_img[1])
    show(image)
        

```