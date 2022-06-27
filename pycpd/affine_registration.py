from builtins import super
import numpy as np
from .emregistration import EMRegistration
from .utility import is_positive_semi_definite


class AffineRegistration(EMRegistration):
    """
    Affine registration.

    Attributes
    ----------
    B: numpy array (semi-positive definite)
        DxD affine transformation matrix.

    t: numpy array
        1xD initial translation vector.

    YPY: float
        Denominator value used to update the scale factor.
        Defined in Fig. 2 and Eq. 8 of https://arxiv.org/pdf/0905.2635.pdf.

    X_hat: numpy array
        (N+K)xD centered target point cloud.
        Defined in Fig. 2 of https://arxiv.org/pdf/0905.2635.pdf

    """

    def __init__(self, B=None, t=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if B is not None and (B.ndim is not 2 or B.shape[0] is not self.D or B.shape[1] is not self.D or not is_positive_semi_definite(B)):
            raise ValueError(
                'The rotation matrix can only be initialized to {}x{} positive semi definite matrices. Instead got: {}.'.format(self.D, self.D, B))

        if t is not None and (t.ndim is not 2 or t.shape[0] is not 1 or t.shape[1] is not self.D):
            raise ValueError(
                'The translation vector can only be initialized to 1x{} positive semi definite matrices. Instead got: {}.'.format(self.D, t))
        self.B = np.eye(self.D) if B is None else B
        self.t = np.atleast_2d(np.zeros((1, self.D))) if t is None else t

    # [same, tentatively, however matlab has slightly different formulations]
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
        
        # [same?] Calculate utility array
        self.A = np.dot(np.transpose(self.X_hat), np.transpose(self.P))
        self.A = np.dot(self.A, Y_hat)

        # [same?] Calculate denominator scalar
        # Why is this calculated differently than the rigid case?
        self.YPY = np.dot(np.transpose(Y_hat), np.diag(self.P1))
        self.YPY = np.dot(self.YPY, Y_hat)

        # [same?] Calculate the new estimate of affine parameters using update rules for (B, t)
        # as defined in Fig. 3 of https://arxiv.org/pdf/0905.2635.pdf.
        self.B = np.linalg.solve(np.transpose(self.YPY), np.transpose(self.A))
        self.t = np.transpose(
            muX) - np.dot(np.transpose(self.B), np.transpose(muY))

    # [same]
    def transform_point_cloud(self, Y=None):
        """
        Update a point cloud using the new estimate of the affine transformation.

        """
        if Y is None:
            self.TY_points_and_landmarks = np.dot(self.Y_points_and_landmarks, self.B) + np.tile(self.t, (self.M + self.K, 1))
            return
        else:
            return np.dot(Y, self.B) + np.tile(self.t, (Y.shape[0], 1))

    def update_variance(self):
        """
        Update the variance of the mixture model using the new estimate of the affine transformation.
        See the update rule for sigma2 in Fig. 3 of of https://arxiv.org/pdf/0905.2635.pdf.

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
        Return the current estimate of the affine transformation parameters.

        """
        return self.B, self.t
