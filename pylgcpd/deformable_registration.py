from builtins import super
import numpy as np
import numbers
from .emregistration import EMRegistration


def gaussian_kernel(X, beta, Y=None):
    if Y is None:
        Y = X
    diff = X[:, None, :] - Y[None, :,  :]
    diff = np.square(diff)
    diff = np.sum(diff, 2)
    return np.exp(-diff / (2 * beta**2))

def low_rank_eigen(G, num_eig):
    """
    Calculate num_eig eigenvectors and eigenvalues of gaussian matrix G.
    Enables lower dimensional solving.
    """
    S, Q = np.linalg.eigh(G)
    eig_indices = list(np.argsort(np.abs(S))[::-1][:num_eig])
    Q = Q[:, eig_indices]  # eigenvectors
    S = S[eig_indices]  # eigenvalues.
    return Q, S


class DeformableRegistration(EMRegistration):
    """
    Deformable registration.

    Attributes
    ----------
    alpha: float (positive)
        Represents the trade-off between the goodness of maximum likelihood fit and regularization.

    beta: float (positive)
        Width of the Gaussian kernel.

    """

    def __init__(self, alpha=None, beta=None, low_rank=False, num_eig=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if alpha is not None and (not isinstance(alpha, numbers.Number) or alpha <= 0):
            raise ValueError(
                "Expected a positive value for regularization parameter alpha. Instead got: {}".format(alpha))

        if beta is not None and (not isinstance(beta, numbers.Number) or beta <= 0):
            raise ValueError(
                "Expected a positive value for the width of the coherent Gaussian kerenl. Instead got: {}".format(beta))

        # Hyper-parameters
        self.alpha = 2 if alpha is None else alpha
        self.beta = 2 if beta is None else beta

        # Not sure what this is....
        self.W = np.zeros((self.M + self.K, self.D))

        # Affinity matrix (of gaussian kernel?)
        self.G = gaussian_kernel(self.Y_points_and_landmarks, self.beta)

        # Are we using low-rank calculations? By default, no (and they're
        # not fully supported anyway).
        self.low_rank = low_rank
        self.num_eig = num_eig
        if self.low_rank is True:
            self.Q, self.S = low_rank_eigen(self.G, self.num_eig)
            self.inv_S = np.diag(1./self.S)
            self.S = np.diag(self.S)
            self.E = 0.

    # [same]
    def update_transform(self):
        """
        Calculate a new estimate of the deformable transformation.
        See Eq. 22 of https://arxiv.org/pdf/0905.2635.pdf.

        """
        if self.low_rank is False:
            # [same] Calculate diagonal matrix of P1
            # If P1 includes landmarks (i.e. is length M+K), then the correct
            # diagonal matrix will be calculated. 
            # Note: If this is taking too long, you may consider:
            # from scipy.sparse import spdiags
            # P1_diag = spdiags(self.P1, 0, self.M+self.K, self.M+self.K)
            P1_diag = np.diag(self.P1)
            # [same] Calc matrix A
            A = np.dot(P1_diag, self.G) + \
                self.alpha * self.sigma2 * np.eye(self.M + self.K)
            # [same] Calc matrix B
            B = self.PX - np.dot(P1_diag, self.Y_points_and_landmarks)
            # [same] Solve linear system AW=B
            self.W = np.linalg.solve(A, B)

        # (ignore, since low rank is not fully supported)
        elif self.low_rank is True:
            # Matlab code equivalent can be found here:
            # https://github.com/markeroon/matlab-computer-vision-routines/tree/master/third_party/CoherentPointDrift
            dP = np.diag(self.P1)
            dPQ = np.matmul(dP, self.Q)
            F = self.PX - np.matmul(dP, self.Y_points)

            self.W = 1 / (self.alpha * self.sigma2) * (F - np.matmul(dPQ, (
                np.linalg.solve((self.alpha * self.sigma2 * self.inv_S + np.matmul(self.Q.T, dPQ)),
                                (np.matmul(self.Q.T, F))))))
            QtW = np.matmul(self.Q.T, self.W)
            self.E = self.E + self.alpha / 2 * np.trace(np.matmul(QtW.T, np.matmul(self.S, QtW)))

    # [same]
    def transform_point_cloud(self, Y=None):
        """
        Update a point cloud using the new estimate of the deformable transformation.

        """
        if Y is not None:
            # (todo, but ignore for now, since it's not used)
            G = gaussian_kernel(X=Y, beta=self.beta, Y=self.Y_points)
            return Y + np.dot(G, self.W)
        else:
            if self.low_rank is False:
                # [same]
                self.TY_points_and_landmarks = self.Y_points_and_landmarks + np.dot(self.G, self.W)

            elif self.low_rank is True:
                # [same] (but won't be verified since low rank is not fully supported)
                self.TY_points_and_landmarks = self.Y_points_and_landmarks + np.matmul(self.Q, np.matmul(self.S, np.matmul(self.Q.T, self.W)))
                return

    # [same]
    def update_variance(self):
        """
        Update the variance of the mixture model using the new estimate of the deformable transformation.
        See the update rule for sigma2 in Eq. 23 of of https://arxiv.org/pdf/0905.2635.pdf.

        """
        qprev = self.sigma2

        # The original CPD paper does not explicitly calculate the objective functional.
        # This functional will include terms from both the negative log-likelihood and
        # the Gaussian kernel used for regularization.
        self.q = np.inf # not sure what this is for

        xPx = np.dot(np.transpose(self.Pt1[:self.N]), np.sum(
            np.multiply(self.X_points, self.X_points), axis=1))
        yPy = np.dot(np.transpose(self.P1[:self.M]),  np.sum(
            np.multiply(self.TY_points_and_landmarks[:self.M], self.TY_points_and_landmarks[:self.M]), axis=1))
        trPXY = np.sum(np.multiply(self.TY_points_and_landmarks[:self.M], self.PX[:self.M]))

        # The matlab implementation includes an absolute value around the sigma2,
        # but it appears that's taken care of below (?).
        self.sigma2 = (xPx - 2 * trPXY + yPy) / (self.Np_without_landmarks * self.D)

        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

        # Here we use the difference between the current and previous
        # estimate of the variance as a proxy to test for convergence.
        self.diff = np.abs(self.sigma2 - qprev)

    def get_registration_parameters(self):
        """
        Return the current estimate of the deformable transformation parameters.

        """
        # The matlab implementation doesn't denormalize the W (and doesn't return the G),
        # so I don't think any denormalization needs to be done here (?).
        return self.G, self.W
