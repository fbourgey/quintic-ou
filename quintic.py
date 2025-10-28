import numpy as np
from scipy import special, interpolate
from black import black_impvol


def horner_vector(poly, n, x):
    """
    Evaluate polynomial using Horner's method for vectorized inputs.

    Parameters
    ----------
    poly : ndarray
        Polynomial coefficients.
    n : int
        Number of coefficients.
    x : ndarray
        Evaluation points.

    Returns
    -------
    ndarray
        Polynomial values at x.
    """
    result = poly[0].reshape(-1, 1)
    for i in range(1, n):
        result = result * x + poly[i].reshape(-1, 1)
    return result


def bs_price_call(s, sigma, T, K):
    """
    Compute Black-Scholes call option price.

    Parameters
    ----------
    s : float or ndarray
        Spot price.
    sigma : float or ndarray
        Volatility.
    T : float
        Time to maturity.
    K : float or ndarray
        Strike price.

    Returns
    -------
    float or ndarray
        Call option price.
    """
    d1 = (np.log(s / K) + 0.5 * sigma**2 * T) / (sigma * T**0.5)
    d2 = d1 - sigma * T**0.5
    return s * special.ndtr(d1) - K * special.ndtr(d2)


def gen_bm_path(n_steps, N_sims):
    """
    Generate Brownian motion paths with antithetic variates.

    Parameters
    ----------
    n_steps : int
        Number of time steps.
    N_sims : int
        Number of simulations.

    Returns
    -------
    ndarray
        Array of shape (n_steps, 2*N_sims) with Brownian increments.
    """
    w1 = np.random.normal(0, 1, (n_steps, N_sims))
    w1 = np.concatenate((w1, -w1), axis=1)  # antithetic variates
    return w1


def doublefactorial(n):
    """
    Compute double factorial n!!.

    Parameters
    ----------
    n : int
        Input value.

    Returns
    -------
    int
        Double factorial of n.
    """
    if n <= 0:
        return 1
    else:
        return n * doublefactorial(n - 2)


def mc_polynomial_fwd_var(
    params,
    T,
    S0,
    Ks,
    n_steps,
    n_mc,
    fv_nodes,
    T_array_nodes,
    w1,
    compute_iv=True,
    spine_k_order=3,
):
    """
    Monte Carlo pricing for the quintic OU model.

    Parameters
    ----------
    params : dict
        Model parameters (rho, H, eps, a_vec).
    T : float
        Maturity time.
    S0 : float
        Initial spot price.
    Ks : ndarray
        Strike prices.
    n_steps : int
        Number of time steps.
    n_mc : int
        Number of Monte Carlo paths.
    fv_nodes : ndarray
        Forward variance nodes.
    T_array_nodes : ndarray
        Time nodes for variance curve.
    w1 : ndarray
        Brownian motion paths.
    compute_iv : bool, optional
        If True, compute implied volatilities.
    spine_k_order : int, optional
        Spline interpolation order.

    Returns
    -------
    tuple
        Call prices, standard errors, and optionally implied volatilities.
    """
    rho, H, eps, a_vec = (
        params["rho"],
        params["H"],
        params["eps"],
        params["a_vec"],
    )
    eta_tild = eps ** (H - 0.5)
    kappa_tild = (0.5 - H) / eps

    a_0, a_1, a_3, a_5 = a_vec
    a_k = np.array([a_0, a_1, 0.0, a_3, 0.0, a_5])

    dt = T / n_steps
    tt = np.linspace(0.0, T, n_steps + 1)

    exp1 = np.exp(kappa_tild * tt)
    exp2 = np.exp(2 * kappa_tild * tt)

    diff_exp2 = np.concatenate((np.array([0.0]), np.diff(exp2)))
    std_vec = np.sqrt(diff_exp2 / (2 * kappa_tild))[
        :, np.newaxis
    ]  # to be broadcasted columnwise
    exp1 = exp1[:, np.newaxis]
    X = (1 / exp1) * (eta_tild * np.cumsum(std_vec * w1, axis=0))
    Xt = np.array(X[:-1])
    del X

    tt = tt[:-1]
    std_X_t = np.sqrt(
        eta_tild**2 / (2 * kappa_tild) * (1 - np.exp(-2 * kappa_tild * tt))
    )
    n = len(a_k)

    cauchy_product = np.convolve(a_k, a_k)
    normal_var = np.sum(
        cauchy_product[np.arange(0, 2 * n, 2)].reshape(-1, 1)
        * std_X_t ** (np.arange(0, 2 * n, 2).reshape(-1, 1))
        * np.array([doublefactorial(z) for z in np.arange(0, 2 * n, 2) - 1]).reshape(
            -1, 1
        ),
        axis=0,
    )

    f_func = horner_vector(a_k[::-1], len(a_k), Xt)

    del Xt

    fv_var_curve_spline_sqrt = interpolate.splrep(
        T_array_nodes, np.sqrt(fv_nodes), k=spine_k_order
    )
    fv_curve = (
        interpolate.splev(tt, fv_var_curve_spline_sqrt, der=0).reshape(-1, 1)
    ) ** 2

    volatility = f_func / np.sqrt(normal_var.reshape(-1, 1))
    del f_func
    volatility = np.sqrt(fv_curve) * volatility

    logS1 = np.log(S0)
    for i in range(w1.shape[0] - 1):
        logS1 = (
            logS1
            - 0.5 * dt * (volatility[i] * rho) ** 2
            + np.sqrt(dt) * rho * volatility[i] * w1[i + 1]
        )
    del w1
    ST1 = np.exp(logS1)
    del logS1

    int_var = np.sum(volatility[:-1,] ** 2 * dt, axis=0)
    Q = np.max(int_var) + 1e-9
    del volatility
    X = (
        bs_price_call(ST1, np.sqrt((1 - rho**2) * int_var / T), T, Ks.reshape(-1, 1))
    ).T
    Y = (
        bs_price_call(ST1, np.sqrt(rho**2 * (Q - int_var) / T), T, Ks.reshape(-1, 1))
    ).T
    del int_var
    eY = (bs_price_call(S0, np.sqrt(rho**2 * (Q) / T), T, Ks.reshape(-1, 1))).T

    c = []
    for i in range(Ks.shape[0]):
        cova = np.cov(X[:, i] + 10, Y[:, i] + 10)[0, 1]
        varg = np.cov(X[:, i] + 10, Y[:, i] + 10)[1, 1]
        temp = 1e-40 if (cova or varg) < 1e-08 else np.nan_to_num(cova / varg, 1e-40)
        temp = np.minimum(temp, 2)
        c.append(temp)
    c = np.array(c)

    call_mc_cv1 = X - c * (Y - eY)
    del X
    del Y
    del eY
    p_mc_cv1 = np.average(call_mc_cv1, axis=0)
    std_mc_cv1 = np.std(call_mc_cv1, axis=0)

    if compute_iv:
        imp_mc = black_impvol(K=Ks, T=T, F=S0, value=p_mc_cv1, opttype=1)
        imp_mc_upper = black_impvol(
            K=Ks,
            T=T,
            F=S0,
            value=p_mc_cv1 + 1.96 * std_mc_cv1 / np.sqrt(2 * n_mc),
            opttype=1,
        )
        imp_mc_lower = black_impvol(
            K=Ks,
            T=T,
            F=S0,
            value=p_mc_cv1 - 1.96 * std_mc_cv1 / np.sqrt(2 * n_mc),
            opttype=1,
        )
        return p_mc_cv1, std_mc_cv1, imp_mc, imp_mc_upper, imp_mc_lower

    else:
        return p_mc_cv1, std_mc_cv1


def vix_all_integration_poly_fast_revert_model(
    H,
    eps,
    T,
    a_k_part,
    x_org_vix,
    w_org_vix,
    vix_strike_perc,
    fv_nodes,
    T_array_nodes,
    n_steps=200,
    compute_iv=True,
    lb_vix=-8,
    ub_vix=8,
):
    spine_k_order = 3

    a2, a4 = (0, 0)
    a0, a1, a3, a5 = a_k_part

    a_k = np.array([a0, a1, a2, a3, a4, a5])
    kappa_tild = (0.5 - H) / eps
    eta_tild = eps ** (H - 0.5)

    delt = 30 / 360
    T_delta = T + delt

    w = w_org_vix / 2 * (ub_vix - lb_vix)
    y = (ub_vix - lb_vix) / 2 * x_org_vix + (ub_vix + lb_vix) / 2
    std_X = eta_tild * np.sqrt(1 / (2 * kappa_tild) * (1 - np.exp(-2 * kappa_tild * T)))
    dt = delt / (n_steps)
    tt = np.linspace(T, T_delta, n_steps + 1)

    exp_det = np.exp(-kappa_tild * (tt - T))
    cauchy_product = np.convolve(a_k, a_k)
    std_Gs_T = eta_tild * np.sqrt(
        1 / (2 * kappa_tild) * (1 - np.exp(-2 * kappa_tild * (tt - T)))
    )
    std_X_t = eta_tild * np.sqrt(
        1 / (2 * kappa_tild) * (1 - np.exp(-2 * kappa_tild * tt))
    )
    std_X_T = eta_tild * np.sqrt(
        1 / (2 * kappa_tild) * (1 - np.exp(-2 * kappa_tild * T))
    )
    n = len(a_k)

    normal_var = np.sum(
        cauchy_product[np.arange(0, 2 * n, 2)].reshape(-1, 1)
        * std_X_t ** (np.arange(0, 2 * n, 2).reshape(-1, 1))
        * np.array([doublefactorial(z) for z in np.arange(0, 2 * n, 2) - 1]).reshape(
            -1, 1
        ),
        axis=0,
    )

    fv_var_curve_spline_sqrt = interpolate.splrep(
        T_array_nodes, np.sqrt(fv_nodes), k=spine_k_order
    )
    FV_curve_all_vix = interpolate.splev(tt, fv_var_curve_spline_sqrt, der=0) ** 2

    beta = []
    for i in range(0, 2 * n - 1):
        k_array = np.arange(i, 2 * n - 1)
        beta_temp = (
            (
                std_Gs_T ** ((k_array - i).reshape(-1, 1))
                * ((k_array - i - 1) % 2).reshape(-1, 1)
                * np.array([doublefactorial(z) for z in k_array - i - 1]).reshape(-1, 1)
                * (special.comb(k_array, i)).reshape(-1, 1)
            )
            * exp_det ** (i)
        ) * cauchy_product[k_array].reshape(-1, 1)
        beta.append(np.sum(beta_temp, axis=0))
    beta = np.array(beta) * FV_curve_all_vix / normal_var

    beta = (np.sum((beta[:, :-1] + beta[:, 1:]) / 2, axis=1)) * dt
    vix_T = np.sqrt(
        horner_vector(beta[::-1], len(beta), std_X_T * y.reshape(1, -1)) / delt
    )
    density = np.exp(-0.5 * y**2) / np.sqrt(2 * np.pi)
    Ft = np.sum(density * vix_T * w)
    vix_strike = Ft * vix_strike_perc
    option_prices = np.sum(
        density * np.maximum(vix_T - vix_strike.reshape(-1, 1), 0) * w, axis=1
    )

    if compute_iv:
        flag = "c"
        imp_vol = vec_find_vol_rat(
            option_prices, Ft, vix_strike_perc * Ft, T, 0.0, flag
        )
        return Ft * 100, option_prices * 100, imp_vol
    else:
        return Ft * 100, option_prices * 100


def spx_lm_range_rule(T_mat):
    if T_mat < 2 / 52:
        lm_range = [-0.15, 0.03]
    elif T_mat < 1 / 12:
        lm_range = [-0.25, 0.05]
    elif T_mat < 2 / 12:
        lm_range = [-0.4, 0.1]
    elif T_mat < 3 / 12:
        lm_range = [-0.6, 0.1]
    elif T_mat < 6 / 12:
        lm_range = [-0.7, 0.15]
    else:
        lm_range = [-1, 0.2]
    return lm_range


def mc_polynomial_fwd_var_parametric(
    rho,
    H,
    eps,
    T,
    a_k_part,
    a,
    b,
    c,
    S0,
    strike_array,
    n_steps,
    N_sims,
    w1,
    compute_iv=True,
    spine_k_order=3,
):
    eta_tild = eps ** (H - 0.5)
    kappa_tild = (0.5 - H) / eps

    a_0, a_1, a_3, a_5 = a_k_part
    a_k = np.array([a_0, a_1, 0, a_3, 0, a_5])

    dt = T / n_steps
    tt = np.linspace(0.0, T, n_steps + 1)

    exp1 = np.exp(kappa_tild * tt)
    exp2 = np.exp(2 * kappa_tild * tt)

    diff_exp2 = np.concatenate((np.array([0.0]), np.diff(exp2)))
    std_vec = np.sqrt(diff_exp2 / (2 * kappa_tild))[
        :, np.newaxis
    ]  # to be broadcasted columnwise
    exp1 = exp1[:, np.newaxis]
    X = (1 / exp1) * (eta_tild * np.cumsum(std_vec * w1, axis=0))
    Xt = np.array(X[:-1])
    del X

    tt = tt[:-1]
    std_X_t = np.sqrt(
        eta_tild**2 / (2 * kappa_tild) * (1 - np.exp(-2 * kappa_tild * tt))
    )
    n = len(a_k)

    cauchy_product = np.convolve(a_k, a_k)
    normal_var = np.sum(
        cauchy_product[np.arange(0, 2 * n, 2)].reshape(-1, 1)
        * std_X_t ** (np.arange(0, 2 * n, 2).reshape(-1, 1))
        * np.array([doublefactorial(z) for z in np.arange(0, 2 * n, 2) - 1]).reshape(
            -1, 1
        ),
        axis=0,
    )

    f_func = horner_vector(a_k[::-1], len(a_k), Xt)

    del Xt

    fv_curve = (a * np.exp(-b * tt) + c * (1 - np.exp(-b * tt))).reshape(-1, 1)

    volatility = f_func / np.sqrt(normal_var.reshape(-1, 1))
    del f_func
    volatility = np.sqrt(fv_curve) * volatility

    logS1 = np.log(S0)
    for i in range(w1.shape[0] - 1):
        logS1 = (
            logS1
            - 0.5 * dt * (volatility[i] * rho) ** 2
            + np.sqrt(dt) * rho * volatility[i] * w1[i + 1]
        )
    del w1
    ST1 = np.exp(logS1)
    del logS1

    int_var = np.sum(volatility[:-1,] ** 2 * dt, axis=0)
    Q = np.max(int_var) + 1e-9
    del volatility
    X = (
        bs_price_call(
            ST1, np.sqrt((1 - rho**2) * int_var / T), T, strike_array.reshape(-1, 1)
        )
    ).T
    Y = (
        bs_price_call(
            ST1, np.sqrt(rho**2 * (Q - int_var) / T), T, strike_array.reshape(-1, 1)
        )
    ).T
    del int_var
    eY = (
        bs_price_call(S0, np.sqrt(rho**2 * (Q) / T), T, strike_array.reshape(-1, 1))
    ).T

    c = []
    for i in range(strike_array.shape[0]):
        cova = np.cov(X[:, i] + 10, Y[:, i] + 10)[0, 1]
        varg = np.cov(X[:, i] + 10, Y[:, i] + 10)[1, 1]
        if (cova or varg) < 1e-8:
            temp = 1e-40
        else:
            temp = np.nan_to_num(cova / varg, 1e-40)
        temp = np.minimum(temp, 2)
        c.append(temp)
    c = np.array(c)

    call_mc_cv1 = X - c * (Y - eY)
    del X
    del Y
    del eY
    p_mc_cv1 = np.average(call_mc_cv1, axis=0)
    std_mc_cv1 = np.std(call_mc_cv1, axis=0)

    if compute_iv:
        flag = "c"
        imp_mc = vec_find_vol_rat(p_mc_cv1, S0, strike_array, T, 0.0, flag)
        imp_mc_upper = vec_find_vol_rat(
            p_mc_cv1 + 1.96 * std_mc_cv1 / (np.sqrt(2 * N_sims)),
            S0,
            strike_array,
            T,
            0.0,
            flag,
        )
        imp_mc_lower = vec_find_vol_rat(
            p_mc_cv1 - 1.96 * std_mc_cv1 / (np.sqrt(2 * N_sims)),
            S0,
            strike_array,
            T,
            0.0,
            flag,
        )

        return p_mc_cv1, std_mc_cv1, imp_mc, imp_mc_upper, imp_mc_lower

    else:
        return p_mc_cv1, std_mc_cv1
