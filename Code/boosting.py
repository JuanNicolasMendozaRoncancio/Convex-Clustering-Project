# The goal is to implement a simple boosting algorithm from scratch in Python based on the paper
# "A new perspective on Boosting in Linear Regression" in the Papers folder.
# For this we have: A data matrix X in R^{n x p}, a response vector y in R^{n}
# and a regression coefficient b in R^{p}.

# We assume the features have been centered and all have L2 norm equal to 1, and that y has also been
# centered to have zero mean.

# We aim to find b such that y = Xb, so we define the residual r = y - Xb.

import numpy as np

class Boosting:
    def __init__(self, X, y):
        """
        Initialize the Boosting class with data and response.
        
        Parameters:
            X: Data matrix of size (n, p)
            y: Response vector of size (n,)
        """

        self.data = X
        self.response = y

    def LS_Boost(self, numiter=100, epsilon=0.1):
        """
        LS-Boost algorithm for linear regression boosting.
        Parameters:
            X: Data matrix of size (n, p).
            y: Response vector of size (n,).
            numiter: Number of algorithm iterations, default 100.
            epsilon: Learning rate, default 0.1.
        Returns:
            b: Regression coefficient vector of size (p,).
        """
        if epsilon <= 0:
            raise ValueError("The epsilon parameter must be positive.")
        else:
            r, b = self.response.copy(), np.zeros(self.data.shape[1])
            for it in range(numiter+1):
                u_m = [np.dot(self.data[:, m], r) 
                       for m in range(self.data.shape[1])]
                res = [np.sum((r - u_m[m] * self.data[:, m])**2) 
                       for m in range(self.data.shape[1])]
                j_k = np.argmin(res)

                r -= epsilon * self.data[:, j_k] * u_m[j_k]
                b[j_k] += epsilon * u_m[j_k]

        return b

    def FS_Boost(self, numiter=100, epsilon=0.1):
        """
        FS-Boost algorithm for linear regression boosting.
        Parameters:
            X: Data matrix of size (n, p).
            y: Response vector of size (n,).
            numiter: Number of algorithm iterations, default 100.
            epsilon: Learning rate, default 0.1.
        Returns:
            b: Regression coefficient vector of size (p,).
        """
        if epsilon <= 0:
            raise ValueError("The epsilon parameter must be positive.")
        else:
            r, b = self.response.copy(), np.zeros(self.data.shape[1])
            for it in range(numiter):
                corr = np.abs(self.data.T @ r)
                j_k = np.argmax(corr)

                s = np.sign(np.dot(r, self.data[:, j_k]))

                r -=  epsilon * s * self.data[:, j_k]
                b[j_k] +=  epsilon * s

        return b

    def R_FS(self, numiter=100, epsilon=0.1, delta=1):
        """
        R-FS boosting algorithm for linear regression.
        Parameters:
            X: Data matrix of size (n, p).
            y: Response vector of size (n,).
            numiter: Number of algorithm iterations, default 100.
            epsilon: Learning rate, default 0.1.
            delta: Regularization parameter, default 1.
        Returns:
            b: Regression coefficient vector of size (p,).    
        """
        if epsilon <= 0:
            raise ValueError("The epsilon parameter must be positive.")
        elif delta <= 0:
            raise ValueError("The delta parameter must be positive.")
        elif epsilon >= delta:
            raise ValueError("The epsilon parameter must be less than delta.")
        else:
            data = self.data.copy().astype(np.float64)
            response = self.response.copy().astype(np.float64)
            r, b = response.copy(), np.zeros(data.shape[1], dtype=np.float64)
            for it in range(numiter):

                corr = np.abs(data.T @ r)
                j_k = np.argmax(corr)

                s = np.sign(np.dot(r, data[:, j_k]))

                r -= epsilon * (s * data[:, j_k] + (1 / delta) * (r - response))
                b = (1 - epsilon / delta) * b
                b[j_k] += epsilon * s

        return b

    def Path_R_FS(self, numiter=100, epsilon=0.1, delta_list=[0.001, 0.01, 0.1, 1]):
        """
        R-FS boosting algorithm with regularization path for linear regression.
        Parameters: 
            X: Data matrix of size (n, p).
            y: Response vector of size (n,).
            numiter: Number of algorithm iterations, default 100.
            epsilon: Learning rate, default 0.1.
            delta_list: List of regularization values for each iteration, must have length numiter+1.
        Returns:
            b: Regression coefficient vector of size (p,).
        """
        if not isinstance(delta_list, (list, np.ndarray)):
            raise ValueError("The delta_list parameter must be a list or numpy array of bounded positive values.")
        if len(delta_list) != numiter + 1:
            raise ValueError(f"delta_list must contain {numiter + 1} values.")
        if any(d == 0 for d in delta_list):
            raise ValueError("All values in delta_list must be non-zero.")
        if epsilon <= 0:
            raise ValueError("The epsilon parameter must be positive.")
        if epsilon > np.min(delta_list):
            raise ValueError("The epsilon parameter must be less than the minimum value in delta_list.")
        else:

            delta_list = np.sort(np.abs(np.array(delta_list)))
            r, b = self.response.copy(), np.zeros(self.data.shape[1])

            for it in range(numiter + 1):
                corr = np.abs(self.data.T @ r)
                j_k = np.argmax(corr)

                s = np.sign(np.dot(r, self.data[:, j_k]))

                r -= epsilon * (s * self.data[:, j_k] + (1 / delta_list[it]) * (r - self.response))
                b = (1 - epsilon / delta_list[it]) * b
                b[j_k] += epsilon * s

        return b
    
    def evaluation(self, b, verbose=True):
        """
        Evaluate the model by computing the mean squared error (MSE) between the predicted and actual response.
        Parameters:
            b: Regression coefficient vector of size (p,) and the regression coefficients.
            verbose: If True, prints the MSE value and the regression equation.
        Returns:
            mse: Mean squared error value.
        """
        predictions = np.dot(self.data, b)
        mse = np.mean((self.response - predictions) ** 2)
        if verbose:
            print(f"The Mean Squared Error is: {mse} \n")
            print(f"The regression coefficients are: {b} \n")
            return mse,b
        else:
            return mse
