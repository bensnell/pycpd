from builtins import super
import numpy as np
import numbers
from .emregistration import EMRegistration
from .utility import is_positive_semi_definite


class RigidRegistration(EMRegistration):
    """
    Rigid registration.

    Attributes
    ----------
    R: numpy array (semi-positive definite)
        DxD rotation matrix. Any well behaved matrix will do,
        since the next estimate is a rotation matrix.

    t: numpy array
        1xD initial translation vector.

    s: float (positive)
        scaling parameter.

    A: numpy array
        DxD utility array used to calculate the rotation matrix.
        Defined in Fig. 2 of https://arxiv.org/pdf/0905.2635.pdf.

    YPY: float
        Denominator value used to update the scale factor.
        Defined in Fig. 2 and Eq. 8 of https://arxiv.org/pdf/0905.2635.pdf.

    X_hat: numpy array
        (N+K)xD centered target point cloud. (includes landmarks)
        Defined in Fig. 2 of https://arxiv.org/pdf/0905.2635.pdf.

    """

    def __init__(self, R=None, t=None, s=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.D != 2 and self.D != 3:
            raise ValueError(
                'Rigid registration only supports 2D or 3D point clouds. Instead got {}.'.format(self.D))

        if R is not None and (R.ndim is not 2 or R.shape[0] is not self.D or R.shape[1] is not self.D or not is_positive_semi_definite(R)):
            raise ValueError(
                'The rotation matrix can only be initialized to {}x{} positive semi definite matrices. Instead got: {}.'.format(self.D, self.D, R))

        if t is not None and (t.ndim is not 2 or t.shape[0] is not 1 or t.shape[1] is not self.D):
            raise ValueError(
                'The translation vector can only be initialized to 1x{} positive semi definite matrices. Instead got: {}.'.format(self.D, t))

        if s is not None and (not isinstance(s, numbers.Number) or s <= 0):
            raise ValueError(
                'The scale factor must be a positive number. Instead got: {}.'.format(s))

        self.R = np.eye(self.D) if R is None else R
        self.t = np.atleast_2d(np.zeros((1, self.D))) if t is None else t
        self.s = 1 if s is None else s

    # [same, tentatively]
    def update_transform(self):
        """
        Calculate a new estimate of the rigid transformation.

        """

        # [same] target point cloud mean (includes landmarks)
        muX = np.divide(np.sum(self.PX, axis=0), self.Np_with_landmarks)
        # [same] source point cloud mean (includes landmarks)
        muY = np.divide(np.sum(np.dot(np.transpose(self.P), \
            self.Y_points_and_landmarks), axis=0), self.Np_with_landmarks)
        
        # [same?] centered target point cloud (includes landmarks)
        self.X_hat = self.X_points_and_landmarks - np.tile(muX, (self.N + self.K, 1))
        # [same?] centered source point cloud (includes landmarks)
        Y_hat = self.Y_points_and_landmarks - np.tile(muY, (self.M + self.K, 1))
        # This is a scalar:
        self.YPY = np.dot(np.transpose(self.P1), np.sum(
            np.multiply(Y_hat, Y_hat), axis=1))

        # [same?] Calculate utility array used for rotation calculations
        self.A = np.dot(np.transpose(self.X_hat), np.transpose(self.P))
        self.A = np.dot(self.A, Y_hat)

        # [same?] Singular value decomposition as per lemma 1 of https://arxiv.org/pdf/0905.2635.pdf.
        U, _, V = np.linalg.svd(self.A, full_matrices=True)
        C = np.ones((self.D, ))
        C[self.D-1] = np.linalg.det(np.dot(U, V))

        # [same] Calculate the rotation matrix using Eq. 9 of https://arxiv.org/pdf/0905.2635.pdf.
        self.R = np.transpose(np.dot(np.dot(U, np.diag(C)), V))
        # [same?] Update scale and translation using Fig. 2 of https://arxiv.org/pdf/0905.2635.pdf.
        self.s = np.trace(np.dot(np.transpose(self.A),
                                 np.transpose(self.R))) / self.YPY
        self.t = np.transpose(muX) - self.s * \
            np.dot(np.transpose(self.R), np.transpose(muY))

    # [same]
    def transform_point_cloud(self, Y=None):
        """
        Update a point cloud using the new estimate of the rigid transformation.

        """
        if Y is None:
            self.TY_points_and_landmarks = self.s * np.dot(self.Y_points_and_landmarks, self.R) + self.t
            return
        else:
            return self.s * np.dot(Y, self.R) + self.t

    # [same as matlab, not same as original python]
    def update_variance(self):
        """
        Update the variance of the mixture model using the new estimate of the rigid transformation.
        See the update rule for sigma2 in Fig. 2 of of https://arxiv.org/pdf/0905.2635.pdf.

        """

        # ===========
        # This is not the same form as deformable matlab and deformable python and 
        # appears to work for guided and work just as well for non-guided.
        # ===========

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
        Return the current estimate of the rigid transformation parameters.

        """

        s = self.s
        R = self.R
        t = self.t

        # Denormalize if needed
        if self.normalize:
            
            X_mean = self.normalize_params['X_mean']
            Y_mean = self.normalize_params['Y_mean']
            X_scale = self.normalize_params['X_scale']
            Y_scale = self.normalize_params['Y_scale']

            # These formula follow the matlab implementation
            s = s * X_scale / Y_scale
            t = X_scale * t + X_mean - s * np.matmul(R, Y_mean)

        return s, R, t
