from __future__ import division
import numpy as np
import numbers
from warnings import warn


def initialize_sigma2(X, Y):
    (N, D) = X.shape
    (M, _) = Y.shape
    diff = X[None, :, :] - Y[:, None, :]
    err = diff ** 2
    return np.sum(err) / (D * M * N)

def lowrankQS(G, beta, num_eig, eig_fgt=False):
    # if we do not use FGT we construct affinity matrix G and find the
    # first eigenvectors/values directly

    if eig_fgt is False:
        S, Q = np.linalg.eigh(G)
        eig_indices = list(np.argsort(np.abs(S))[::-1][:num_eig])
        Q = Q[:, eig_indices]  # eigenvectors
        S = S[eig_indices]  # eigenvalues.

        return Q, S

    elif eig_fgt is True:
        raise Exception('Fast Gauss Transform Not Implemented!')

class EMRegistration(object):
    """
    Expectation maximization point cloud registration.

    Attributes
    ----------
    X: numpy array
        NxD array of target points.

    X_landmarks: numpy array
        KxD array of target landmarks

    X_and_landmarks: numpy array
        (N+K)xD array of target points and landmarks (concatenated)

    Y: numpy array
        MxD array of source points.

    Y_landmarks: numpy array
        KxD array of source landmarks

    Y_and_landmarks: numpy array
        (M+K)xD array of source points and landmarks (concatenated)

    TY_and_landmarks: numpy array
        (M+K)xD array of transformed source points (and landmarks at end).
        To access just source points, use `TY_and_landmarks[:M]`

    sigma2: float (positive)
        Initial variance of the Gaussian mixture model.

    N: int
        Number of target points.

    M: int
        Number of source points.

    D: int
        Dimensionality of source and target points.

    K: int
        Number of landmarks (0 if not landmark-guided).

    iteration: int
        The current iteration throughout registration.

    max_iterations: int
        Registration will terminate once the algorithm has taken this
        many iterations.

    tolerance: float (positive)
        Registration will terminate once the difference between
        consecutive objective function values falls within this tolerance.

    w: float (between 0 and 1)
        Contribution of the uniform distribution to account for outliers.
        Valid values span 0 (inclusive) and 1 (exclusive).

    q: float
        The objective function value that represents the misalignment between source
        and target point clouds.

    diff: float (positive)
        The absolute difference between the current and previous objective function values.

    P: numpy array
        MxN array of probabilities.
        P[m, n] represents the probability that the m-th source point
        corresponds to the n-th target point.

    Pt1: numpy array
        Nx1 column array.
        Multiplication result between the transpose of P and a column vector of all 1s.

    P1: numpy array
        Mx1 column array.
        Multiplication result between P and a column vector of all 1s.

    Np: float (positive)
        The sum of all elements in P.

    landmark_guided: boolean
        Is this a guided registration?

    ss2: float (positive)
        "Sigma Starred Squared"
        Describes the influence of landmarks. 
        (The smaller the value is set for σ*2, the stronger the constraints on the corresponding landmarks.)

    """

    def __init__(self, 
        X, 
        Y, 
        sigma2=None, 
        max_iterations=None, 
        tolerance=None, 
        w=None, 
        ss2=None,
        X_landmarks=None,
        Y_landmarks=None,
        *args, **kwargs):
        if type(X) is not np.ndarray or X.ndim != 2:
            raise ValueError(
                "The target point cloud (X) must be at a 2D numpy array.")

        if type(Y) is not np.ndarray or Y.ndim != 2:
            raise ValueError(
                "The source point cloud (Y) must be a 2D numpy array.")

        if X.shape[1] != Y.shape[1]:
            raise ValueError(
                "Both point clouds need to have the same number of dimensions.")

        if sigma2 is not None and (not isinstance(sigma2, numbers.Number) or sigma2 <= 0):
            raise ValueError(
                "Expected a positive value for sigma2 instead got: {}".format(sigma2))

        if max_iterations is not None and (not isinstance(max_iterations, numbers.Number) or max_iterations < 0):
            raise ValueError(
                "Expected a positive integer for max_iterations instead got: {}".format(max_iterations))
        elif isinstance(max_iterations, numbers.Number) and not isinstance(max_iterations, int):
            warn("Received a non-integer value for max_iterations: {}. Casting to integer.".format(max_iterations))
            max_iterations = int(max_iterations)

        if tolerance is not None and (not isinstance(tolerance, numbers.Number) or tolerance < 0):
            raise ValueError(
                "Expected a positive float for tolerance instead got: {}".format(tolerance))

        if w is not None and (not isinstance(w, numbers.Number) or w < 0 or w >= 1):
            raise ValueError(
                "Expected a value between 0 (inclusive) and 1 (exclusive) for w instead got: {}".format(w))

        if ss2 is not None and (not isinstance(ss2, numbers.Number) or ss2 <= 0):
            raise ValueError(
                "Expected a positive value for ss2. Instead got: {}".format(ss2))
        
        # Is this landmark-guided?
        self.landmark_guided = X_landmarks is not None and Y_landmarks is not None \
            and X_landmarks.shape[0] != 0 and X_landmarks.shape == Y_landmarks.shape

        if self.landmark_guided:
            print("Enabling landmark-guided registration.")

        if X_landmarks is not None and len(X_landmarks)==0:
            raise ValueError(
                "Expected array of nonzero length for X_landmarks. Instead got: {}".format(X_landmarks))
        
        if Y_landmarks is not None and len(Y_landmarks)==0:
            raise ValueError(
                "Expected array of nonzero length for Y_landmarks. Instead got: {}".format(Y_landmarks))

        if X_landmarks is not None and Y_landmarks is not None and X_landmarks.shape != Y_landmarks.shape:
            raise ValueError(
                "Landmark arrays must be the same shape. Cannot enable landmark-guided registration.")

        # Target points (no landmarks)
        self.X = X
        # Source points (no landmarks)
        self.Y = Y

        # Quantities & dimensionalities of points
        (self.N, self.D) = self.X.shape
        (self.M, _) = self.Y.shape

        # Landmark-guided hyper-parameter (What should default be?)
        self.ss2 = 1e-1 if ss2 is None else ss2
        # Number of landmarks
        self.K = 0 if not self.landmark_guided else X_landmarks.shape[0]
        # Landmarks
        if self.landmark_guided:
            self.X_landmarks = X_landmarks
            self.Y_landmarks = Y_landmarks
        else:
            self.X_landmarks = np.zeros((0,self.D))
            self.Y_landmarks = np.zeros((0,self.D))
        # Points and landmarks concatenated
        self.X_and_landmarks = np.concatenate([self.X, self.X_landmarks])
        self.Y_and_landmarks = np.concatenate([self.Y, self.Y_landmarks])

        # Transformed source points (and landmarks)
        # TODO: Adjust rigid, affine to allow for landmark-guided
        self.TY = np.copy(Y) # Dummy attribute (so rigid, affine compile)
        # There appears to be an error here in the pycpd implementation where Y 
        # is shallow copied into TY, so as TY changes, as does Y.
        self.TY_and_landmarks = np.copy(self.Y_and_landmarks)

        # Iterations
        self.max_iterations = 100 if max_iterations is None else max_iterations
        self.iteration = 0
        self.diff = np.inf

        # Tolerance
        self.tolerance = 0.001 if tolerance is None else tolerance

        # Outlier influence
        # (Default is 0.1 in matlab code)
        self.w = 0.0 if w is None else w

        # Initial variance of GMM
        self.sigma2 = initialize_sigma2(X, Y) if sigma2 is None else sigma2

        # Other matricies used mostly in the expectation step.
        # Their sizes are correct, but I'm not 100% sure
        # what the scalars represent.
        self.q = np.inf
        self.P = np.zeros((self.M + self.K, self.N + self.K))
        self.Pt1 = np.zeros((self.N + self.K, ))
        self.P1 = np.zeros((self.M + self.K, ))
        self.PX = np.zeros((self.M + self.K, self.D))
        self.Np = 0

    def register(self, callback=lambda **kwargs: None):
        self.transform_point_cloud()
        # Should we include an additional check for sigma2 > 1e-8 here?
        while self.iteration < self.max_iterations and self.diff > self.tolerance:
            self.iterate()
            if callable(callback):
                kwargs = {'iteration': self.iteration,
                          'error': self.q, 'X': self.X, 'Y': self.TY_and_landmarks[:self.M]}
                callback(**kwargs)

        return self.TY_and_landmarks[:self.M], self.get_registration_parameters()

    def get_registration_parameters(self):
        raise NotImplementedError(
            "Registration parameters should be defined in child classes.")

    def update_transform(self):
        raise NotImplementedError(
            "Updating transform parameters should be defined in child classes.")

    def transform_point_cloud(self):
        raise NotImplementedError(
            "Updating the source point cloud should be defined in child classes.")

    def update_variance(self):
        raise NotImplementedError(
            "Updating the Gaussian variance for the mixture model should be defined in child classes.")

    def iterate(self):
        self.expectation()
        self.maximization()
        self.iteration += 1

    def expectation(self):
        
        # Calculate Pmn. Don't worry about landmarks at the moment.
        # Begin calculating the Pmn matrix. 
        P = np.sum((self.X[None, :, :] - self.TY_and_landmarks[:self.M][:, None, :]) ** 2, axis=2)
        # Calculate the right hand side of the expression
        # in the denominator for Pmn
        c = (2 * np.pi * self.sigma2) ** (self.D / 2)
        c = c * self.w / (1 - self.w)
        c = c * self.M / self.N
        # Apply exponent to Pmn
        P = np.exp(-P / (2 * self.sigma2))
        # Calculate the full denominator
        den = np.sum(P, axis=0)
        den = np.tile(den, (self.M, 1))
        den[den == 0] = np.finfo(float).eps
        den += c
        # Calculate Pmn. This completes the expectation step for non-guided.
        P = np.divide(P, den)

        # Now, consider the effect of landmarks.
        if self.landmark_guided:
            # Increase the size of P to account for the interaction between landmarks
            P = np.pad(P, ((0,self.K),(0,self.K)), 'constant', constant_values=(0))
            # Make the landmark sub-matrix an identity matrix, where
            # the diagonals are sigma2/ss2.
            P[self.M:,self.N:] = np.identity(self.K) * self.sigma2/self.ss2
        
        self.P = P
        self.Pt1 = np.sum(self.P, axis=0) # [same, I think]
        self.P1 = np.sum(self.P, axis=1) # [same, I think]
        self.Np = np.sum(self.P1[:self.M]) # [same]
        self.PX = np.matmul(self.P, self.X_and_landmarks) # [same, I think ?]

    def maximization(self):
        self.update_transform()
        self.transform_point_cloud()
        self.update_variance()
