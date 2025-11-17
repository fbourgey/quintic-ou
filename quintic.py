import numpy as np
from utils import gauss_hermite, gauss_legendre
from black import black_impvol, black_otm_impvol_mc
from scipy import stats


def std_x_ou_quintic(params, t):
    """
    Compute standard deviation of X_t in the quintic OU model where

    X_t = eps^(H-0.5) * int_0^t exp(-kappa * (t-s)) dW_s
    and kappa = (0.5 - H)/eps.
    """
    H = params["H"]
    eps = params["eps"]
    var = (eps ** (2 * H) / (1 - 2 * H)) * (1 - np.exp(-((1 - 2 * H) / eps) * t))
    return var**0.5


def mean_px_squared(a0, a1, a3, a5, mu, sig, n_quad=None):
    """
    Closed-form E[p(X)^2] for p(x)=a0+a1*x+a3*x**3+a5*x**5 and X~N(mu, sig^2).
    Supports numpy broadcasting over mu/sig.

    Parameters
    ----------
    a0, a1, a3, a5 : float
        Polynomial coefficients.
    mu : float or ndarray, default 0.0
        Mean of X.
    sig : float or ndarray, default 1.0
        Std dev of X.
    n_quad : int, optional
        Number of Gauss-Hermite quadrature points. If None, uses analytical formula.

    Returns
    -------
    float or ndarray
        E[p(X)^2].
    """
    if n_quad is not None:

        def quintic_poly(x):
            return (a0 + a1 * x + a3 * x**3 + a5 * x**5) ** 2

        knots, weights = gauss_hermite(n_quad)
        return np.sum(weights * quintic_poly(mu + sig * knots))

    mu = np.asarray(mu)
    s2 = np.asarray(sig) ** 2
    s4 = s2**2
    s6 = s2**3
    s8 = s2**4
    s10 = s2**5

    term0 = a0**2
    term1 = 2 * a0 * a1 * mu
    term2 = a1**2 * (mu**2 + s2)
    term3 = 2 * a0 * a3 * (mu**3 + 3 * mu * s2)
    term4 = 2 * a1 * a3 * (mu**4 + 6 * mu**2 * s2 + 3 * s4)
    term5 = 2 * a0 * a5 * (mu**5 + 10 * mu**3 * s2 + 15 * mu * s4)
    term6 = (a3**2 + 2 * a1 * a5) * (
        mu**6 + 15 * mu**4 * s2 + 45 * mu**2 * s4 + 15 * s6
    )
    term7 = (
        2
        * a3
        * a5
        * (mu**8 + 28 * mu**6 * s2 + 210 * mu**4 * s4 + 420 * mu**2 * s6 + 105 * s8)
    )
    term8 = a5**2 * (
        mu**10
        + 45 * mu**8 * s2
        + 630 * mu**6 * s4
        + 3150 * mu**4 * s6
        + 4725 * mu**2 * s8
        + 945 * s10
    )

    return term0 + term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8


def cov_delta_X_delta_W(delta, eps, H):
    r"""
    Compute the covariance matrix of (delta_X, delta_W) over time increment delta.

    Here, delta_X = eps^{H-1/2} \int_{t}^{t+delta} e^{-((0.5-H)/eps)(t+delta - s)} dW_s
    and delta_W = W_{t+delta} - W_t.

    Parameters
    ----------
    delta : float
        Time increment.
    eps : float
        Epsilon parameter of the model.
    H : float
        Hurst parameter of the model.

    Returns
    -------
    ndarray
        2x2 covariance matrix.
    """
    var_X = (eps ** (2 * H) / (1 - 2 * H)) * (1 - np.exp(-((1 - 2 * H) / eps) * delta))
    cov_XW = (eps ** (0.5 + H) / (0.5 - H)) * (1 - np.exp(-((0.5 - H) / eps) * delta))
    var_W = delta
    return np.array(
        [
            [var_X, cov_XW],
            [cov_XW, var_W],
        ]
    )


def simulate_quintic_ou(
    params, xi0, T, k, n_steps, n_mc, seed=None, return_paths=False, antithetic=True
):
    """
    Simulate paths of the quintic OU stochastic volatility model.

    The model is defined by:
    - dX_t = -kappa * X_t dt + eps^(H-0.5) dW_t^X, where kappa = (0.5-H)/eps
    - V_t = xi0(t) * p(X_t)^2 / E[p(X_t)^2], with p(x) = a0 + a1*x + a3*x^3 + a5*x^5
    - dS_t/S_t = sqrt(V_t) dW_t^S, where dW^X dW^S = rho dt

    Parameters
    ----------
    params : dict
        Model parameters with keys:
        - 'rho' : float, correlation between X and S Brownian motions
        - 'H' : float, Hurst-like parameter (0 < H < 0.5)
        - 'eps' : float, mean-reversion timescale parameter
        - 'a_vec' : array_like, polynomial coefficients [a0, a1, a3, a5]
    xi0 : callable
        Function xi0(t) defining the deterministic variance level.
    T : float
        Maturity time.
    k : ndarray
        Log-moneyness strikes for implied volatility computation.
    n_steps : int
        Number of time steps for discretization.
    n_mc : int
        Number of Monte Carlo paths.
    seed : int, optional
        Random seed for reproducibility.
    return_paths : bool, optional
        If True, return terminal spot prices. If False (default), return
        implied volatilities.
    antithetic : bool, optional
        If True (default), use antithetic variates for variance reduction.

    Returns
    -------
    ndarray or tuple
        If return_paths=True: array of shape (n_mc,) with terminal spot
        prices S_T.
        If return_paths=False: tuple (iv, iv_stderr) with implied
        volatilities and standard errors.
    """
    if seed is not None:
        np.random.seed(seed)

    dT = T / n_steps
    rho, H, eps, a_vec = (
        params["rho"],
        params["H"],
        params["eps"],
        params["a_vec"],
    )
    a0, a1, a3, a5 = a_vec

    def quintic_poly(x):
        return a0 + a1 * x + a3 * x**3 + a5 * x**5

    # precompute OU/finite-step covariance pieces
    std_X = np.sqrt(
        (eps ** (2 * H) / (1 - 2 * H)) * (1 - np.exp(-((1 - 2 * H) / eps) * dT))
    )
    std_W = dT**0.5
    cov_XW = (
        rho * (eps ** (0.5 + H) / (0.5 - H)) * (1 - np.exp(-((0.5 - H) / eps) * dT))
    )
    rho_XW = cov_XW / (std_X * std_W)
    exp_eps = np.exp(-((0.5 - H) / eps) * dT)

    # antithetic normal generator
    def antithetic_normals(n_steps, n_mc):
        z = np.random.normal(size=(n_steps, n_mc // 2))
        return np.concatenate([z, -z], axis=1)

    if antithetic:
        # normal drives dX; normal_perp is independent piece to complete dW;
        # normal_perp is orthogonal
        normal = antithetic_normals(n_steps, n_mc)
        normal_perp = antithetic_normals(n_steps, n_mc)
    else:
        normal = np.random.normal(size=(n_steps, n_mc))
        normal_perp = np.random.normal(size=(n_steps, n_mc))

    # state arrays
    X = np.zeros((n_steps + 1, n_mc))
    sig = np.zeros((n_steps + 1, n_mc))  # volatility process
    sig[0, :] = xi0(0.0) ** 0.5
    t_grid = np.linspace(0.0, T, n_steps + 1)

    # normalization sqrt(E[p(X_t)^2])
    norm_coef = np.empty(n_steps + 1)
    for i in range(n_steps + 1):
        sig_ti = std_x_ou_quintic(params, t=t_grid[i])  # = sqrt(Var[X_ti])
        norm_coef[i] = (
            mean_px_squared(a0=a0, a1=a1, a3=a3, a5=a5, mu=0.0, sig=sig_ti) ** 0.5
        )

    # generate increments
    dX = std_X * normal
    dW_S = std_W * (rho_XW * normal + np.sqrt(1 - rho_XW**2) * normal_perp)

    for i in range(n_steps):
        # OU update
        X[i + 1, :] = exp_eps * X[i, :] + dX[i, :]
    # variance update
    sig[1:, :] = (
        xi0(t_grid[1:, None]) ** 0.5 * quintic_poly(X[1:, :]) / norm_coef[1:][:, None]
    )

    # compute spot paths
    int_V_dt = np.sum(sig[:-1, :] ** 2 * dT, axis=0)
    int_sqrt_V_dW = np.sum(sig[:-1, :] * dW_S, axis=0)
    S_T = np.exp(-0.5 * int_V_dt + int_sqrt_V_dW)

    if return_paths:
        return S_T
    else:
        # implied volatilities
        return black_otm_impvol_mc(S=S_T, k=k, T=T, mc_error=True)


def f_xi0(t, u, x, params):
    """
    Compute f^u(t, x) where xi_t(u) = xi_0(u) f^u(t, x)
    """
    H = params["H"]
    eps = params["eps"]
    a_vec = params["a_vec"]
    a0, a1, a3, a5 = a_vec

    num = mean_px_squared(
        a0=a0,
        a1=a1,
        a3=a3,
        a5=a5,
        mu=x * np.exp(-((0.5 - H) / eps) * (u - t)),
        sig=np.sqrt(
            (eps ** (2 * H) / (1 - 2 * H))
            * (1 - np.exp(-((1 - 2 * H) / eps) * (u - t)))
        ),
    )
    denom = mean_px_squared(
        a0=a0,
        a1=a1,
        a3=a3,
        a5=a5,
        mu=0.0,
        sig=np.sqrt(
            (eps ** (2 * H) / (1 - 2 * H)) * (1 - np.exp(-((1 - 2 * H) / eps) * u))
        ),
    )
    return num / denom


def fut_vix(params, xi0, T, n_quad=20, delta_vix=30 / 365):
    return price_vix(
        params=params,
        xi0=xi0,
        T=T,
        K=0.0,
        n_quad=n_quad,
        opt=1,
        delta_vix=delta_vix,
    )


def price_vix(params, xi0, T, K, n_quad=20, opt=1, delta_vix=30 / 365):
    opt = np.atleast_1d(opt)
    K = np.atleast_1d(K)

    if opt.shape != K.shape:
        raise ValueError("opt and K must have the same length")

    x_leg, w_leg = gauss_legendre(0, 1, n_quad)
    x_norm, w_norm = stats.norm.ppf(x_leg), w_leg

    std_X_T = std_x_ou_quintic(params, T)

    price = np.zeros_like(K)
    for j in range(len(K)):
        integrand = np.zeros(n_quad)
        for i in range(n_quad):
            f_val = f_xi0(
                t=T, u=T + delta_vix * x_leg, x=x_norm[i] * std_X_T, params=params
            )
            vix = np.sqrt(np.sum(w_leg * xi0(T + delta_vix * x_leg) * f_val))
            integrand[i] = np.maximum(opt[j] * (vix - K[j]), 0)

        price[j] = np.sum(w_norm * integrand)

    return price


def impvol_vix(params, xi0, T, K, n_quad=20, delta_vix=30 / 365):
    K = np.atleast_1d(K)
    F = fut_vix(params=params, xi0=xi0, T=T, n_quad=n_quad, delta_vix=delta_vix)
    otm_price = np.zeros_like(K)
    opttype = 2 * (K > F) - 1  # 1 for call, -1 for put
    otm_price = price_vix(
        params=params,
        xi0=xi0,
        T=T,
        K=K,
        n_quad=n_quad,
        opt=opttype,
        delta_vix=delta_vix,
    )
    print(otm_price)
    return black_impvol(value=otm_price, F=F, K=K, T=T, opttype=opttype)
