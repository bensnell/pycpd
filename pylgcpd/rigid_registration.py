from builtins import super
import numpy as np
import numbers
from .emregistration import EMRegistration
from .utility import is_positive_semi_definite, to_numpy, import_cupy_xp
cp, xp = import_cupy_xp()

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

        self.R = xp.eye(self.D) if R is None else R
        self.t = xp.atleast_2d(xp.zeros((1, self.D))) if t is None else t
        self.s = 1 if s is None else s

    # [same, tentatively]
    def update_transform(self):
        """
        Calculate a new estimate of the rigid transformation.

        """

        # [same] target point cloud mean (includes landmarks)
        muX = xp.divide(xp.sum(self.PX, axis=0), self.Np_with_landmarks)
        # [same] source point cloud mean (includes landmarks)
        muY = xp.divide(xp.sum(xp.dot(xp.transpose(self.P), \
            self.Y_points_and_landmarks), axis=0), self.Np_with_landmarks)
        
        # [same?] centered target point cloud (includes landmarks)
        self.X_hat = self.X_points_and_landmarks - xp.tile(muX, (self.N + self.K, 1))
        # [same?] centered source point cloud (includes landmarks)
        Y_hat = self.Y_points_and_landmarks - xp.tile(muY, (self.M + self.K, 1))
        # This is a scalar:
        self.YPY = xp.dot(xp.transpose(self.P1), xp.sum(
            xp.multiply(Y_hat, Y_hat), axis=1))

        # [same?] Calculate utility array used for rotation calculations
        self.A = xp.dot(xp.transpose(self.X_hat), xp.transpose(self.P))
        self.A = xp.dot(self.A, Y_hat)

        # [same?] Singular value decomposition as per lemma 1 of https://arxiv.org/pdf/0905.2635.pdf.
        U, _, V = xp.linalg.svd(self.A, full_matrices=True)
        C = xp.ones((self.D, ))
        C[self.D-1] = xp.linalg.det(xp.dot(U, V))

        # [same] Calculate the rotation matrix using Eq. 9 of https://arxiv.org/pdf/0905.2635.pdf.
        self.R = xp.transpose(xp.dot(xp.dot(U, xp.diag(C)), V))
        # [same?] Update scale and translation using Fig. 2 of https://arxiv.org/pdf/0905.2635.pdf.
        self.s = xp.trace(xp.dot(xp.transpose(self.A),
                                 xp.transpose(self.R))) / self.YPY
        self.t = xp.transpose(muX) - self.s * \
            xp.dot(xp.transpose(self.R), xp.transpose(muY))

    # [same]
    def transform_point_cloud(self):
        """
        Update a point cloud using the new estimate of the rigid transformation.

        """

        self.TY_points_and_landmarks = self.s * xp.dot(self.Y_points_and_landmarks, self.R) + self.t


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

            # Matlab uses a similar, but wrong version of these formula, with
            # a `xp.matmul(R, Y_mean)` term instead of `xp.dot(Y_mean, R)`. Testing
            # has verified that the formula below is correct.
            s = s * X_scale / Y_scale
            t = X_scale * t + X_mean - s * xp.dot(Y_mean, R)

        return to_numpy(s), to_numpy(R), to_numpy(t)

    def get_transformation_function(self):
        """
        Return the point cloud transformation function.

        """

        s, R, t = self.get_registration_parameters()

        def transform(Y):
            return s * np.dot(Y, R) + t
        
        return transform
