import numpy as np


class RBM:
    def __init__(self, p, q, init_std_w=0.1) -> None:
        """[Restricted Boltzmann Machine](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine)

        Parameters
        ----------
        p : int
            Dimensionality of each sample (width X height)
        q : int
            Hyperparameter - size of the latent variables (hidden units)
        init_std_w : float
            Hyperparameter - standard deviation of the gaussian to sample inital weights from
        """
        # Initialize biases (a, b) and weight matrix (w)
        self.a = np.zeros(p)
        self.b = np.zeros(q)
        self.w = np.reshape(np.random.normal(scale=init_std_w, size=p * q), (p, q))
        self.p = p
        self.q = q

    def encode(self, X):
        """Encode data X into latent (hidden) vector H
        n: samples per batch
        p: dimensionality of one sample (vector of observations)
        q: dimensionality of one hidden vector

        Parameters
        ----------
        X : np.array (n x p)
            (Batch of) data to encode

        Returns
        -------
        H : np.array (n x q)
            Latent vector
        """

        H = 1 / (1 + np.exp(-(X @ self.w + self.b)))
        return H

    def decode(self, H):
        """Decode latent (hidden) vector H into vector X
        n: samples per batch
        p: dimensionality of one sample (vector of observations)
        q: dimensionality of one hidden vector

        Parameters
        ----------
        H : np.array (n x q)
            Latent vector to decode

        Returns
        -------
        X_rec : np.array (n x p)
            (Batch of) decoded (or reconstructed) data
        """
        X_rec = 1 / (1 + np.exp(-(H @ self.w.T + self.a)))
        return X_rec

    def train_RBM(self, x, batch_size, n_epoch=5, eps=0.1, verbose=False):
        """Train the RBM with the Contrastive Divergence-1 algorithm.

        Parameters
        ----------
        x : np.array
            Input data, as np.array of binary np.arrays
        batch_size : int
            Amount of data to be used for one gradient ascent step
        n_epoch : int, optional
            Number of epochs, by default 5
        eps : float, optional
            Learning rate, by default 0.1
        """
        for epoch in range(n_epoch):
            if verbose:
                print(f"Epoch: {epoch+1}/{n_epoch}")

            # Random permutation so that the epoch is not the same at each time
            x = np.random.permutation(x)
            for batch_start in range(0, x.shape[0], batch_size):
                # if verbose: print(f"New batch with start index: {batch_start}")

                # Create batch
                x_batch = x[batch_start : min(x.shape[0], batch_start + batch_size)]  # Deal w/ scenario of batch_size not diviser of nb_epoch
                sz_batch = x_batch.shape[0]

                # Initialize v0
                v0 = x_batch  # sz_batch x p

                p_h_v0 = self.encode(v0)  # sz_batch x q

                # Sample h0 from the obtained distribution p(h|v0)
                h0 = (np.random.random((sz_batch, self.q)) < p_h_v0) * 1  # sz_batch x q

                p_v_h0 = self.decode(h0)  # sz_batch x p

                # Sample v1 from the obtained distribution p(v|h0)
                v1 = (np.random.random((sz_batch, self.p)) < p_v_h0) * 1  # sz_batch x p

                p_h_v1 = self.encode(v1)  # sz_batch x q

                # Obtain gradients
                grad_a = np.sum(v0 - v1, axis=0)  # p
                grad_b = np.sum(p_h_v0 - p_h_v1, axis=0)  # q
                grad_w = v0.T @ p_h_v0 - v1.T @ p_h_v1  # p x q
                # Gradient ascent
                self.a += eps / sz_batch * grad_a
                self.b += eps / sz_batch * grad_b
                self.w += eps / sz_batch * grad_w

            # We won't evaluate the loss function, because we wouldn't be able to calculate it
            # What we do instead is to obtain the reconstruction error
            H = self.encode(x)
            x_reconst = self.decode(H)
            if verbose:
                print("Reconstruction error: ", np.mean(np.power(x - x_reconst, 2)))

    def generate_data(self, nb_data, nb_iter_gibbs=2, height=20, width=16, random_init=True):
        """Generate data using Gibbs Sampling.

        Parameters
        ----------
        nb_data : int
            Number of samples to generate
        nb_iter_gibbs : int, optional
            Number of Gibbs Sampling iterations, by default 2
        height : int, optional
            Height of the sample to be generated, by default 20
        width : int, optional
            Width of the sample to be generated, by default 16
        random_init : bool, optional
            Randomly initialize the Bernoulli parameter to pick v_0 from (or use 0.5 as the parameter), by default True

        Returns
        -------
        images : np.array
            Generated images
        """

        # Initialize the Gibbs sampler
        if random_init:
            # We change the parameter of the Bernoulli at each time, to avoid bias in the initialization
            v = (np.random.random((nb_data, self.p)) < np.random.uniform()) * 1
        else:
            v = (np.random.random((nb_data, self.p)) < 0.5) * 1

        for _ in range(nb_iter_gibbs):
            h = (np.random.random((nb_data, self.q)) < self.encode(v)) * 1
            v = (np.random.random((nb_data, self.p)) < self.decode(h)) * 1

        images = np.reshape(v, (nb_data, height, width))

        return images
