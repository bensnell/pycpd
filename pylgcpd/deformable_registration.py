from builtins import super
import numpy as np
import numbers
from .emregistration import EMRegistration
from .utility import to_numpy, import_cupy_xp
cp, xp = import_cupy_xp()


def gaussian_kernel(X, beta, Y=None):

    # Select the appropriate library, as this function should support both
    # cupy and numpy calculations
    _xp = np if type(X) == np.ndarray else xp

    if Y is None:
        Y = X
    diff = X[:, None, :] - Y[None, :,  :]
    diff = _xp.square(diff)
    diff = _xp.sum(diff, 2)
    return _xp.exp(-diff / (2 * beta**2))

def low_rank_eigen(G, num_eig):
    """
    Calculate num_eig eigenvectors and eigenvalues of gaussian matrix G.
    Enables lower dimensional solving.
    """
    S, Q = xp.linalg.eigh(G)
    eig_indices = list(xp.argsort(xp.abs(S))[::-1][:num_eig])
    Q = Q[:, eig_indices]  # eigenvectors
    S = S[eig_indices]  # eigenvalues.
    return Q, S


class DeformableRegistration(EMRegistration):
    """
    Deformable registration.

    Attributes
    ----------
    alpha: float (positive) or dict following the form { 'start' : float, 'stop' : float, 'power' : float }
        Represents the trade-off between the goodness of maximum likelihood fit and regularization.
        This is the same as `lambda` in the paper.
        The larger this is, the more "coherent" the point cloud is.
        If this is a dict and force_max_iterations is True, then alpha will interpolate between `start` and `stop`
        with the exponent `power` at each iteration.

    beta: float (positive) or dict following the form { 'start' : float, 'stop' : float, 'power' : float }
        Width of the Gaussian kernel.
        The larger this is, the more "rigid" the point cloud is.
        If this is a dict and force_max_iterations is True, then beta will interpolate between `start` and `stop`
        with the exponent `power` at each iteration.

    """

    def __init__(self, alpha=None, beta=None, low_rank=False, num_eig=100, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Validate and create alpha object and numerical params
        if alpha is not None and type(alpha) == dict:
            if "start" not in alpha or "stop" not in alpha:
                raise ValueError("If `alpha` is a dict, it must contain the keys `start` and `stop`.")

        self._alpha = 2 if alpha is None else alpha
        if isinstance(self._alpha, numbers.Number):
            self._alpha = {
                'start' : self._alpha,
                'stop' : self._alpha
            }
        if 'power' not in self._alpha:
            self._alpha['power'] = 1
        self.alpha = self._alpha['start']

        if not isinstance(self._alpha['start'], numbers.Number) or self._alpha['start'] <= 0:
            raise ValueError(
                "Expected a positive value for regularization parameter alpha `start`. Instead got: {}".format(self._alpha['start']))
        if not isinstance(self._alpha['stop'], numbers.Number) or self._alpha['stop'] <= 0:
            raise ValueError(
                "Expected a positive value for regularization parameter alpha `stop`. Instead got: {}".format(self._alpha['stop']))
        if not isinstance(self._alpha['power'], numbers.Number) or self._alpha['power'] <= 0:
            raise ValueError(
                "Expected a positive value for regularization parameter alpha `power`. Instead got: {}".format(self._alpha['power']))


        # Validate and create beta object and numerical params
        if beta is not None and type(beta) == dict:
            if "start" not in beta or "stop" not in beta:
                raise ValueError("If `beta` is a dict, it must contain the keys `start` and `stop`.")

        self._beta = 2 if beta is None else beta
        if isinstance(self._beta, numbers.Number):
            self._beta = {
                'start' : self._beta,
                'stop' : self._beta
            }
        if 'power' not in self._beta:
            self._beta['power'] = 1
        self.beta = self._beta['start']

        if not isinstance(self._beta['start'], numbers.Number) or self._beta['start'] <= 0:
            raise ValueError(
                "Expected a positive value for regularization parameter beta `start`. Instead got: {}".format(self._beta['start']))
        if not isinstance(self._beta['stop'], numbers.Number) or self._beta['stop'] <= 0:
            raise ValueError(
                "Expected a positive value for regularization parameter beta `stop`. Instead got: {}".format(self._beta['stop']))
        if not isinstance(self._beta['power'], numbers.Number) or self._beta['power'] <= 0:
            raise ValueError(
                "Expected a positive value for regularization parameter beta `power`. Instead got: {}".format(self._beta['power']))


        # Not sure what this is....
        self.W = xp.zeros((self.M + self.K, self.D))

        # Affinity matrix (of gaussian kernel?)
        self.G = gaussian_kernel(self.Y_points_and_landmarks, self.beta)

        # Are we using low-rank calculations? By default, no (and they're
        # not fully supported anyway).
        self.low_rank = low_rank
        self.num_eig = num_eig
        if self.low_rank is True:
            self.Q, self.S = low_rank_eigen(self.G, self.num_eig)
            self.inv_S = xp.diag(1./self.S)
            self.S = xp.diag(self.S)
            self.E = 0.

    def update_hyperparameters(self):

        # If force_max_iterations is true, then we can interpolate
        # the hyperparameters
        if self.force_max_iterations:

            # Calculate the new parameters
            alpha_param = (self.iteration / self.max_iterations) ** self._alpha['power']
            alpha = alpha_param * (self._alpha['stop'] - self._alpha['start']) + self._alpha['start']
            beta_param = (self.iteration / self.max_iterations) ** self._beta['power']
            beta = beta_param * (self._beta['stop'] - self._beta['start']) + self._beta['start']

            # Save booleans that indiciate whether the values changed
            alpha_changed = alpha != self.alpha
            beta_changed = beta != self.beta

            # Update the internal hyperparameters
            self.alpha = alpha
            self.beta = beta

            # Change anything that needs to be changed
            if beta_changed:
                # TODO: Optimize this
                self.G = gaussian_kernel(self.Y_points_and_landmarks, self.beta)
                

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
            # P1_diag = xp.sparse.spdiags(self.P1, 0, self.M+self.K, self.M+self.K)
            P1_diag = xp.diag(self.P1)
            # [same] Calc matrix A
            A = xp.dot(P1_diag, self.G) + \
                self.alpha * self.sigma2 * xp.eye(self.M + self.K)
            # [same] Calc matrix B
            B = self.PX - xp.dot(P1_diag, self.Y_points_and_landmarks)
            # [same] Solve linear system AW=B
            self.W = xp.linalg.solve(A, B)

        # (ignore, since low rank is not fully supported)
        elif self.low_rank is True:
            # Matlab code equivalent can be found here:
            # https://github.com/markeroon/matlab-computer-vision-routines/tree/master/third_party/CoherentPointDrift
            dP = xp.diag(self.P1)
            dPQ = xp.matmul(dP, self.Q)
            F = self.PX - xp.matmul(dP, self.Y_points)

            self.W = 1 / (self.alpha * self.sigma2) * (F - xp.matmul(dPQ, (
                xp.linalg.solve((self.alpha * self.sigma2 * self.inv_S + xp.matmul(self.Q.T, dPQ)),
                                (xp.matmul(self.Q.T, F))))))
            QtW = xp.matmul(self.Q.T, self.W)
            self.E = self.E + self.alpha / 2 * xp.trace(xp.matmul(QtW.T, xp.matmul(self.S, QtW)))

    # [same]
    def transform_point_cloud(self):
        """
        Update a point cloud using the new estimate of the deformable transformation.

        """

        if self.low_rank is False:
            # [same]
            self.TY_points_and_landmarks = self.Y_points_and_landmarks + xp.dot(self.G, self.W)

        elif self.low_rank is True:
            # [same] (but won't be verified since low rank is not fully supported)
            self.TY_points_and_landmarks = self.Y_points_and_landmarks + xp.matmul(self.Q, xp.matmul(self.S, xp.matmul(self.Q.T, self.W)))
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
        self.q = xp.inf # not sure what this is for

        xPx = xp.dot(xp.transpose(self.Pt1[:self.N]), xp.sum(
            xp.multiply(self.X_points, self.X_points), axis=1))
        yPy = xp.dot(xp.transpose(self.P1[:self.M]),  xp.sum(
            xp.multiply(self.TY_points_and_landmarks[:self.M], self.TY_points_and_landmarks[:self.M]), axis=1))
        trPXY = xp.sum(xp.multiply(self.TY_points_and_landmarks[:self.M], self.PX[:self.M]))

        # The matlab implementation includes an absolute value around the sigma2,
        # but it appears that's taken care of below (?).
        self.sigma2 = (xPx - 2 * trPXY + yPy) / (self.Np_without_landmarks * self.D)

        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

        # Here we use the difference between the current and previous
        # estimate of the variance as a proxy to test for convergence.
        self.diff = xp.abs(self.sigma2 - qprev)

    def get_registration_parameters(self):
        """
        Return the current estimate of the deformable transformation parameters.

        """
        # G and W do not include the normalization factors. Consequently, you must apply those on 
        # your own or use the get_transformation_function to transform data.
        return to_numpy(self.G), to_numpy(self.W)

    def get_transformation_function(self):
        """
        Return the point cloud transformation function.

        """

        # Assumes lowrank is False

        _, W = self.get_registration_parameters()
        beta = to_numpy(self.beta)
        Y_points_and_landmarks = to_numpy(self.Y_points_and_landmarks)
        normalize = self.normalize_fncts['normalize']
        denormalize = self.normalize_fncts['denormalize']
        
        def transform(Y):
            Y_normalized = normalize(Y,'Y')
            return denormalize(
                Y_normalized + np.dot(
                    gaussian_kernel(X=Y_normalized, beta=beta, Y=Y_points_and_landmarks), 
                    W), 
                'X')
        
        return transform
