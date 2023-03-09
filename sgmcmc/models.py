# https://github.com/williwilliams3/NUTSmonge/blob/main/NUTS2/models.py
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy import stats


def under_over_flow(sp, v, cut_off=np.finfo(float).eps, min_val=np.finfo(float).eps):
    # function checks for under and over flow for nonzero, non infinity variables
    # Note: This function is correct only for sp>=0, since sp<np.finfo(float).eps only for non negative values
    # min_val = 1e-13
    if np.isscalar(sp):
        if np.isinf(sp):
            sp = v
        if sp < cut_off:
            sp = min_val
    else:
        if np.isinf(sp).any():
            sp[np.isinf(sp)] = v[np.isinf(sp)]
        if (sp < cut_off).any():
            sp[sp < cut_off] = min_val
    return sp


def clip_derivative(u, threshold=10e10):
    # I dont see any need to control underflows
    # ind0 = np.abs(u)<np.finfo(float).eps
    # if ind0.any():
    #     u[ind0] = 0
    mg = norm(u)  # Use the norm of Gradient or Hessian
    if (np.isfinite(mg)) and (mg > threshold):
        u = threshold * u / mg  # Gradient clipping-by-norm
    else:
        u[np.isnan(u)] = 1
    # mg = np.max(np.abs(u))
    # if (np.isfinite(mg)) and  (mg>threshold): # Marcelo clipping method
    #     u *= 0.3 # Gradient clipping-by-norm
    return u


def softplus(v):
    # Vectorized softplus function
    sp = np.log(1.0 + np.exp(v))
    sp = under_over_flow(sp, v)
    return sp


def dsoftplus(v):
    # Vectorized derivative softplus function
    sp = 1 / (1.0 + np.exp(-v))
    sp = under_over_flow(sp, v)
    return sp


def d2softplus(v):
    # Vectorized derivative softplus function
    sp = dsoftplus(v) * (1 - dsoftplus(v))
    sp = under_over_flow(sp, v)
    return sp


class funnel:
    def __init__(self, D=2, mmean=2, sig2v=15, alpha=1):
        self.sig2v = sig2v
        self.mmean = mmean
        self.D = D
        self.alpha = alpha

    def logp(self, x):
        """Funnel log density (vectorized)
        Parameters
        ----------
        M : class
            funnel class object
        x : numpy float
            first columns are points at which to evaluate
            last column is funnel parameter v at point
        Returns
        -------
        numpy float
            a numpy array of the log density at given points
        """
        x = np.asarray(x)
        s2 = self.sig2v
        muth = self.mmean
        D1 = self.D - 1
        v = x[-1]
        sig2th = softplus(v)
        lp = (
            -(D1 * 0.5) * (np.log(2.0 * np.pi) + np.log(sig2th))
            - 0.5 * norm(x[:-1] - muth) ** 2.0 / sig2th
            - 0.5 * (np.log(2.0 * np.pi) + np.log(s2) + v**2.0 / s2)
        )
        return lp

    def dlogp(self, x):
        """Funnel partial derivatives log density
        Parameters
        ----------
        M : class
            funnel class object
        x : numpy float
            first columns are points at which to evaluate
            last column is funnel parameter v at point
        Returns
        -------
        numpy float
            a numpy array of the partial derivatives log density at given points
        """
        x = np.asarray(x)
        s2 = self.sig2v
        muth = self.mmean
        D1 = self.D - 1
        v = x[-1]
        # u is a N, D matrix, in each row the gradient [d x, d a]
        u = np.zeros(self.D)
        sig2th = softplus(v)
        dsig2th = dsoftplus(v)
        u[0:D1] = -(x[0:D1] - muth) / sig2th
        u[-1] = (
            -D1 / 2.0 * dsig2th / sig2th
            + 0.5 * norm(x[0:D1] - muth) ** 2.0 / sig2th**2.0 * dsig2th
            - v / s2
        )
        u = clip_derivative(u)
        return u

    def d2logp(self, x):
        """Funnel Hessian of log density
        Parameters
        ----------
        M : class
            funnel class object
        x : numpy float
            first entry is points at which to evaluate
            second entry is funnel parameter v
        Returns
        -------
        numpy float
            a numpy array of the Hessian log density
        """
        x = np.asarray(x)
        s2 = self.sig2v
        muth = self.mmean
        D1 = self.D - 1
        v = x[-1]
        U = np.zeros((self.D, self.D))
        sig2th = softplus(v)
        dsig2th = dsoftplus(v)
        d2sig2th = dsig2th - dsig2th**2.0
        d1 = d2sig2th / sig2th - dsig2th**2.0 / sig2th**2.0
        d2 = d2sig2th / sig2th**2.0 - 2.0 * dsig2th**2.0 / sig2th**3.0
        # Diagonal elements (Second derivatives)
        # First 0:D1 elements
        U[range(D1), range(D1)] = -np.ones((D1)) / sig2th
        # Last D1+1 element
        U[D1, D1] = -D1 / 2.0 * d1 + d2 * 0.5 * norm(x[0:D1] - muth) ** 2.0 - 1.0 / s2
        # Non diagonal entries (Cross derivatives)
        U[-1, range(D1)] = U[range(D1), -1] = (
            (x[0:D1] - muth) * dsig2th / (sig2th**2.0)
        )
        # Clip values
        U = clip_derivative(U)
        return U

    def hvp_logp(self, x, v):
        x = np.asarray(x)
        s2 = self.sig2v
        muth = self.mmean
        D1 = self.D - 1
        y = x[-1]
        hpv = np.zeros(self.D)
        sig2th = softplus(y)
        dsig2th = dsoftplus(y)
        d2sig2th = dsig2th - dsig2th**2.0
        d1 = d2sig2th / sig2th - dsig2th**2.0 / sig2th**2.0
        d2 = d2sig2th / sig2th**2.0 - 2.0 * dsig2th**2.0 / sig2th**3.0
        # Diagonal elements (Second derivatives)
        # First 0:D1 elements
        hpv[0:D1] = (
            -1 / sig2th * v[0:D1] + (x[0:D1] - muth) * dsig2th / (sig2th**2.0) * v[-1]
        )
        hpv[D1] = np.dot((x[0:D1] - muth), v[0:D1]) * dsig2th / (sig2th**2.0)
        hpv[D1] += (
            -D1 / 2.0 * d1 + d2 * 0.5 * norm(x[0:D1] - muth) ** 2.0 - 1.0 / s2
        ) * v[-1]
        # Clip values
        hpv = clip_derivative(hpv)
        return hpv

    def densities(self):
        D = self.D
        s2 = self.sig2v
        muth = self.mmean
        density1 = lambda x: stats.norm.pdf(x, 0, np.sqrt(s2))
        x = np.random.normal(loc=0, scale=np.sqrt(s2), size=10000)
        a = [
            np.random.normal(loc=muth, scale=np.sqrt(np.log(1 + np.exp(xi))))
            for xi in x
        ]
        kde = stats.gaussian_kde(a)
        density2 = lambda y: kde(y)
        return density2, density1


class one_d_gaussian:
    def __init__(self, mu=0, sigma=1, alpha=1):
        self.mu = mu
        self.sigma = sigma
        self.alpha = alpha

    def logp(self, q):
        return (
            -np.log(2 * np.pi) / 2
            - np.log(self.sigma)
            - (q**2 - 2 * self.mu * q + self.mu**2) / 2 / self.sigma**2
        )

    def dlogp(self, q):
        return (self.mu - q) / self.sigma**2

    def d2logp(self, q):
        return -1 / self.sigma**2


class correlated_gaussian:
    def __init__(self, rho=0.90, alpha=1):
        self.rho = rho
        self.D = 2
        self.alpha = alpha

    def logp(self, q):
        rho = self.rho
        x, y = q[0], q[1]
        lp = (
            -np.log(2 * np.pi)
            - 0.5 * np.log(1 - rho**2)
            - (x**2 - rho * x * y - rho * x * y + y**2) / (2 * (1 - rho**2))
        )
        return lp

    def dlogp(self, q):
        rho = self.rho
        x, y = q[0], q[1]
        dlogf = -np.array([(x - rho * y), (y - rho * x)]) / (1 - rho**2)
        return dlogf

    def d2logp(self, x):
        rho = self.rho
        d2f = -np.array([[1, -rho], [-rho, 1]]) / (1 - rho**2)
        return d2f

    def densities(self):
        density = lambda x: stats.norm.pdf(x)
        return density, density

    def hvp_logp(self, x, v):
        rho = self.rho
        hpv = -np.array([v[0] - rho * v[0], -rho * v[0] + v[1]]) / (1 - rho**2)
        hpv = clip_derivative(hpv)
        return hpv


class banana_distribution:
    def __init__(self, a=1, b=100, D=2, alpha=1):
        self.a = a
        self.b = b
        self.D = D
        self.alpha = alpha

    def logp(self, x):
        a = self.a
        b = self.b
        return -((a - x[0]) ** 2) - b * (x[1] - x[0] ** 2) ** 2

    def dlogp(self, x):
        a = self.a
        b = self.b
        # u = np.array([2*(a-x[0]) + 4*b*x[0]*x[1] - 4*b*x[0]**3, 2*b*(x[0]**2-x[1]) ])
        u = np.array(
            [
                4 * b * x[0] * (x[1] - x[0] ** 2) + 2 * (a - x[0]),
                2 * b * (x[0] ** 2 - x[1]),
            ]
        )
        u = clip_derivative(u)
        return u

    def d2logp(self, q):
        a = self.a
        b = self.b
        x, y = q[0], q[1]
        U = np.array(
            [[-2 + 4 * b * y - 12 * b * x**2, 4 * b * x], [4 * b * x, -2 * b]]
        )
        U = clip_derivative(U)
        return U

    def hvp_logp(self, x, v):
        a = self.a
        b = self.b
        hpv = np.array(
            [
                (-2 + 4 * b * x[1] - 12 * b * x[0] ** 2) * v[0] + 4 * b * x[0] * v[1],
                4 * b * x[0] * v[0] - 2 * b * v[1],
            ]
        )
        hpv = clip_derivative(hpv)
        return hpv

    def densities(self):
        a = self.a
        b = self.b
        density1 = lambda x_lambda: stats.norm.pdf(x_lambda, a, np.sqrt(0.5))

        x = np.random.normal(loc=a, scale=np.sqrt(0.5), size=10000)
        a_temp = [np.random.normal(loc=xi**2, scale=np.sqrt(0.5 / b)) for xi in x]
        kde = stats.gaussian_kde(a_temp)
        density2 = lambda y: kde(y)
        return density1, density2


class squiggle:
    def __init__(self, a=5, Sig=np.array([[2.0, 0.25], [0.25, 0.05]]), D=2, alpha=1.0):
        self.a = a
        self.D = D
        self.mu = np.zeros(D)
        self.Sig = Sig
        self.dist = stats.multivariate_normal(mean=np.zeros(D), cov=Sig)
        self.invSig = np.linalg.inv(Sig)
        self.alpha = alpha

    def logp(self, q):
        a = self.a
        dist = self.dist
        y = np.array([q[0], q[1] + np.sin(a * q[0])])
        return dist.logpdf(y)

    def dlogp(self, q):
        a = self.a
        mu = self.mu
        invSig = self.invSig
        y = np.array([q[0], q[1] + np.sin(a * q[0])])
        gy = -np.dot(invSig, y - mu)
        u = np.array([gy[0] + gy[1] * a * np.cos(a * q[0]), gy[1]])
        return u

    def d2logp(self, q):
        a = self.a
        mu = self.mu
        invSig = self.invSig
        y = np.array([q[0], q[1] + np.sin(a * q[0])])
        gy = -np.dot(invSig, y - mu)
        Hy = -invSig
        Jx = np.array([[1.0, 0.0], [a * np.cos(a * q[0]), 1.0]])
        U = np.linalg.multi_dot([Jx.T, Hy, Jx])
        U[0, 0] += gy[1] * (-(a**2.0) * np.sin(a * q[0]))
        return U

    def hvp_logp(self, x, v):
        U = self.d2logp(x)
        hpv = np.dot(U, v)
        hpv = clip_derivative(hpv)
        return hpv


class ring_distribution:
    def __init__(self, mean=5.0, sig2=1.0, alpha=1):
        self.D = 2
        self.mean = mean
        self.sig2 = sig2
        self.alpha = alpha

    def logp(self, q):
        r = np.linalg.norm(q)
        mean = self.mean
        sig2 = self.sig2
        lp = -0.5 * (np.log(2.0 * np.pi) + np.log(sig2) + (r - mean) ** 2.0 / sig2)
        return lp - np.log(r)

    def dlogp(self, q):
        r = np.linalg.norm(q)
        mean = self.mean
        sig2 = self.sig2
        dlogr = -(r - mean) / sig2
        q = np.array(q)
        u = dlogr * (1.0 / r) * q - q / r**2
        u = clip_derivative(u)
        return u

    def d2logp(self, q):
        r = np.linalg.norm(q)
        mean = self.mean
        sig2 = self.sig2
        dlogr = -(r - mean) / sig2 - 1.0 / r
        d2logr = -1 / sig2 + 1.0 / (r**2.0)
        drdx1 = q[0] / r
        drdx2 = q[1] / r
        d2rdx1 = 1.0 / r - q[0] ** 2.0 / (r ** (3.0))
        d2rdx2 = 1.0 / r - q[1] ** 2.0 / (r ** (3.0))
        U = np.array(
            [
                [
                    d2logr * drdx1**2.0 + dlogr * d2rdx1,
                    d2logr * drdx1 * drdx2 - dlogr * q[0] * q[1] / (r**3.0),
                ],
                [
                    d2logr * drdx1 * drdx2 - dlogr * q[0] * q[1] / (r**3.0),
                    d2logr * drdx2**2.0 + dlogr * d2rdx2,
                ],
            ]
        )
        U = clip_derivative(U)
        return U

    def hvp_logp(self, x, v):
        U = self.d2logp(x)
        hpv = np.dot(U, v)
        hpv = clip_derivative(hpv)
        return hpv


class sphere:
    def __init__(self, R=1, alpha=1):
        self.D = 2
        self.R = R
        self.alpha = alpha

    def Pi(self, q):
        R = self.R
        if np.linalg.norm(q) < R:
            lp = (R**2 - np.linalg.norm(q) ** 2) ** 0.5
        else:
            lp = 0
        return lp

    def dPi(self, q):
        R = self.R
        u = -q / self.logp(q)
        return u

    def logp(self, q):
        R = self.R
        lp = 0.5 * np.log(R**2 - np.linalg.norm(q) ** 2)
        return lp

    def dlogp(self, q):
        R = self.R
        u = -q / self.Pi(q) ** 2
        u = clip_derivative(u)
        return u

    def d2logp(self, q):
        R = self.R
        D = self.D
        g = self.dlogp(q)
        dPi = self.dPi(q)
        U = -np.eye(D) / (R**2 - np.linalg.norm(q) ** 2) - 2 * np.outer(g, g)
        # U = -np.eye(D)/self.Pi(q)**2 - 2 * np.outer(q,dPi)/self.Pi(q)**3
        # U = clip_derivative(U)
        return U

    def hvp_logp(self, x, v):
        U = self.d2logp(x)
        hpv = np.dot(U, v)
        hpv = clip_derivative(hpv)
        return hpv


class sphere_notlog:
    def __init__(self, R=1, alpha=1):
        self.D = 2
        self.R = R
        self.alpha = alpha

    def logp(self, q):
        R = self.R
        if np.linalg.norm(q) < R:
            lp = (R**2 - np.linalg.norm(q) ** 2) ** 0.5
        else:
            lp = 0
        return lp

    def dlogp(self, q):
        R = self.R
        u = -q / self.logp(q)
        # u = clip_derivative(u)
        return u

    def d2logp(self, q):
        R = self.R
        D = self.D
        lp = self.logp(q)
        g = self.dlogp(q)
        U = -np.eye(D) / lp + np.outer(q, g) / lp**2
        # U = clip_derivative(U)
        return U

    def hvp_logp(self, x, v):
        U = self.d2logp(x)
        hpv = np.dot(U, v)
        hpv = clip_derivative(hpv)
        return hpv


class rosenbrock:
    # Adapted from https://github.com/scipy/scipy/blob/v0.14.0/scipy/optimize/optimize.py#L153
    def __init__(self, a=1.0, b=100.0, D=2, alpha=1):
        self.a = a
        self.b = b
        self.D = D
        self.alpha = alpha

    def logp(self, x):
        a = self.a
        b = self.b
        x = np.asarray(x)
        # lp = -(a-x)**2 - b*(y-x**2)**2
        return -sum(b * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (a - x[:-1]) ** 2.0)

    def dlogp(self, x):
        a = self.a
        b = self.b
        x = np.asarray(x)
        xm = x[1:-1]
        xm_m1 = x[:-2]
        xm_p1 = x[2:]
        der = np.zeros_like(x)
        der[1:-1] = (
            -2 * b * (xm - xm_m1**2) + 4 * b * (xm_p1 - xm**2) * xm + 2 * (a - xm)
        )
        der[0] = 4 * b * x[0] * (x[1] - x[0] ** 2) + 2 * (a - x[0])
        der[-1] = -2 * b * (x[-1] - x[-2] ** 2)
        der = clip_derivative(der)
        return der

    def d2logp(self, x):
        a = self.a
        b = self.b
        x = np.atleast_1d(x)
        H = np.diag(4 * b * x[:-1], 1) + np.diag(
            4 * b * x[:-1], -1
        )  # non diag elements
        diagonal = np.zeros_like(x)
        diagonal[0] = -12 * b * x[0] ** 2 + 400 * x[1] - 2
        diagonal[-1] = -2 * b
        diagonal[1:-1] = -12 * b * x[1:-1] ** 2 + 4 * b * x[2:] - 2 * b - 2
        H = H + np.diag(diagonal)
        H = clip_derivative(H)
        return H

    def hvp_logp(self, x, p):
        a = self.a
        b = self.b
        x = np.atleast_1d(x)
        Hp = np.zeros(len(x), dtype=x.dtype)
        Hp[0] = (12 * b * x[0] ** 2 - 4 * b * x[1] + 2) * p[0] - 4 * b * x[0] * p[1]
        Hp[1:-1] = (
            -4 * b * x[:-2] * p[:-2]
            + (2 * b + 2 + 12 * b * x[1:-1] ** 2 - 4 * b * x[2:]) * p[1:-1]
            - 4 * b * x[1:-1] * p[2:]
        )
        Hp[-1] = -4 * b * x[-2] * p[-2] + 2 * b * p[-1]
        Hp = clip_derivative(Hp)
        return -Hp

    def densities(self):
        a = self.a
        b = self.b
        density1 = lambda x_lambda: stats.norm.pdf(x_lambda, a, np.sqrt(0.5))
        x = np.random.normal(loc=a, scale=np.sqrt(0.5), size=10000)
        a_temp = [np.random.normal(loc=xi**2, scale=np.sqrt(0.5 / b)) for xi in x]
        kde = stats.gaussian_kde(a_temp)
        density2 = lambda y: kde(y)
        return density1, density2
