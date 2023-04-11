import numpy as np


class TnsUnsupervised():
    def __int__(self, perplexity=20, n_components=2, n_iter=500, learning_rate=10.0, seed=1):
        """
        :param perplexity: default 20
        :param n_components: default 2
        :param n_iter: default 500
        :param learning_rate: default 10.0
        :param seed: default 1
        """
        self.perplexity = perplexity
        self.n_components = n_components
        self.n_iters = n_iter
        self.learning_rate = learning_rate
        self.seed = seed
        self.momentum = 0.9
        self.plot = 5

    def fit_transform(self, X):
        P = self.compute_joint_probabilities(X, self.perplexity)
        Y = np.random.RandomState(self.seed).normal(0.0, 0.0001, [X.shape[0], 2])

        # Initialise past values (used for momentum)
        if self.momentum:
            Y_m2 = Y.copy()
            Y_m1 = Y.copy()

        # Start gradient descent loop
        for i in range(self.n_iters):

            # Get Q and distances (distances only used for t-SNE)
            Q, distances = self.q_tsne(Y)
            # Estimate gradients with respect to Y
            grads = self.tsne_grad(self, P, Q, Y, distances)

            # Update Y
            Y = Y - self.learning_rate * grads
            if self.momentum:  # Add momentum
                Y += self.momentum * (Y_m1 - Y_m2)
                # Update previous Y's for momentum
                Y_m2 = Y_m1.copy()
                Y_m1 = Y.copy()
        return Y

    def q_tsne(self, Y):
        """t-SNE: Given low-dimensional representations Y, compute
        matrix of joint probabilities with entries q_ij."""
        distances = self.neg_squared_euc_dists(Y)
        inv_distances = np.power(1. - distances, -1)
        np.fill_diagonal(inv_distances, 0.)
        return inv_distances / np.sum(inv_distances), inv_distances

    def neg_squared_euc_dists(self, X):
        """Compute matrix containing negative squared euclidean
        distance for all pairs of points in input matrix X

        # Arguments:
            X: matrix of size NxD
        # Returns:
            NxN matrix D, with entry D_ij = negative squared
            euclidean distance between rows X_i and X_j
        """
        # Math? See https://stackoverflow.com/questions/37009647
        sum_X = np.sum(np.square(X), 1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        return -D

    def tsne_grad(self, P, Q, Y, inv_distances):
        """Estimate the gradient of t-SNE cost with respect to Y."""
        pq_diff = P - Q
        pq_expanded = np.expand_dims(pq_diff, 2)
        y_diffs = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)

        # Expand our inv_distances matrix so can multiply by y_diffs
        distances_expanded = np.expand_dims(inv_distances, 2)

        # Multiply this by inverse distances matrix
        y_diffs_wt = y_diffs * distances_expanded

        # Multiply then sum over j's
        grad = 4. * (pq_expanded * y_diffs_wt).sum(1)
        return grad

    def compute_joint_probabilities(self, X, target_perplexity):
        """Given a data matrix X, gives joint probabilities matrix.
        # Arguments
            X: Input data matrix.
        # Returns:
            P: Matrix with entries p_ij = joint probabilities.
        """
        # Get the negative euclidian distances matrix for our data
        distances = self.neg_squared_euc_dists(X)
        # Find optimal sigma for each row of this distances matrix
        sigmas = self.find_optimal_sigmas(distances, target_perplexity)
        # Calculate the probabilities based on these optimal sigmas
        p_conditional = self.calc_prob_matrix(distances, sigmas)
        # Go from conditional to joint probabilities matrix
        P = self.p_conditional_to_joint(p_conditional)
        return P

    def p_conditional_to_joint(self, P):
        """Given conditional probabilities matrix P, return
        approximation of joint distribution probabilities."""
        return (P + P.T) / (2. * P.shape[0])

    def neg_squared_euc_dists(self, X):
        """Compute matrix containing negative squared euclidean
        distance for all pairs of points in input matrix X
        # Arguments:
            X: matrix of size NxD
        # Returns:
            NxN matrix D, with entry D_ij = negative squared
            euclidean distance between rows X_i and X_j
        """
        # Math? See https://stackoverflow.com/questions/37009647
        sum_X = np.sum(np.square(X), 1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        return -D

    def find_optimal_sigmas(self, distances, target_perplexity):
        """For each row of distances matrix, find sigma that results
        in target perplexity for that role."""
        sigmas = []
        # For each row of the matrix (each point in our dataset)
        for i in range(distances.shape[0]):
            # Make fn that returns perplexity of this row given sigma
            eval_fn = lambda sigma: \
                self.calculate_perplexity(distances[i:i + 1, :], np.array(sigma), i)
            # Binary search over sigmas to achieve target perplexity
            correct_sigma = self.binary_search(eval_fn, target_perplexity)
            # Append the resulting sigma to our output array
            sigmas.append(correct_sigma)
        return np.array(sigmas)

    def binary_search(self, eval_fn, target, tol=1e-10, max_iter=10000,
                      lower=1e-20, upper=1000.):
        """Perform a binary search over input values to eval_fn.
        # Arguments
            eval_fn: Function that we are optimising over.
            target: Target value we want the function to output.
            tol: Float, once our guess is this close to target, stop.
            max_iter: Integer, maximum num. iterations to search for.
            lower: Float, lower bound of search range.
            upper: Float, upper bound of search range.
        # Returns:
            Float, best input value to function found during search.
        """
        for i in range(max_iter):
            guess = (lower + upper) / 2.
            val = eval_fn(guess)
            if val > target:
                upper = guess
            else:
                lower = guess
            if np.abs(val - target) <= tol:
                break
        return guess

    def calculate_perplexity(self, distances, sigmas, zero_index):
        """Wrapper function for quick calculation of
        perplexity over a distance matrix."""
        return self.calc_perplexity(
            self.calc_prob_matrix(distances, sigmas, zero_index))

    def calc_perplexity(self, prob_matrix):
        """Calculate the perplexity of each row
        of a matrix of probabilities."""
        entropy = -np.sum(prob_matrix * np.log2(prob_matrix), 1)
        perplexity = 2 ** entropy
        return perplexity

    def calc_prob_matrix(self, distances, sigmas=None, zero_index=None):
        """Convert a distances matrix to a matrix of probabilities."""
        if sigmas is not None:
            two_sig_sq = 2. * np.square(sigmas.reshape((-1, 1)))
            return self.softmax(distances / two_sig_sq, zero_index=zero_index)
        else:
            return self.softmax(distances, zero_index=zero_index)

    def softmax(self, X, diag_zero=True, zero_index=None):
        """Compute softmax values for each row of matrix X."""

        # Subtract max for numerical stability
        e_x = np.exp(X - np.max(X, axis=1).reshape([-1, 1]))

        # We usually want diagonal probailities to be 0.
        if zero_index is None:
            if diag_zero:
                np.fill_diagonal(e_x, 0.)
        else:
            e_x[:, zero_index] = 0.

        # Add a tiny constant for stability of log we take later
        e_x = e_x + 1e-8  # numerical stability

        return e_x / e_x.sum(axis=1).reshape([-1, 1])


