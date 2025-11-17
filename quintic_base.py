# Original code

import numpy as np
from scipy import special, interpolate
from black import black_impvol, black_otm_impvol_mc


def horner_vector(poly, n, x):
    # Initialize result
    result = poly[0].reshape(-1, 1)
    for i in range(1, n):
        result = result * x + poly[i].reshape(-1, 1)
    return result


def bs_price_call(s, sigma, T, K):
    # x is defined as St/K
    d1 = (np.log(s / K) + 0.5 * np.power(sigma, 2.0) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = s * special.ndtr(d1) - K * special.ndtr(d2)
    return price


def gen_bm_path(n_steps, N_sims):
    w1 = np.random.normal(0, 1, (n_steps, N_sims))
    # Antithetic variates
    w1 = np.concatenate((w1, -w1), axis=1)
    return w1


def doublefactorial(n):
    if n <= 0:
        return 1
    else:
        return n * doublefactorial(n - 2)


def mc_polynomial_fwd_var(
    rho,
    H,
    eps,
    T,
    a_k_part,
    S0,
    strike_array,
    n_steps,
    N_sims,
    fv_nodes,
    T_array_nodes,
    w1,
    compute_iv=True,
):
    spine_k_order = 3

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
        imp_mc = black_impvol(value=p_mc_cv1, F=S0, K=strike_array, T=T, opttype=1)
        imp_mc_upper = black_impvol(
            value=p_mc_cv1 + 1.96 * std_mc_cv1 / (np.sqrt(2 * N_sims)),
            F=S0,
            K=strike_array,
            T=T,
            opttype=1,
        )
        imp_mc_lower = black_impvol(
            value=p_mc_cv1 - 1.96 * std_mc_cv1 / (np.sqrt(2 * N_sims)),
            F=S0,
            K=strike_array,
            T=T,
            opttype=1,
        )

        return p_mc_cv1, std_mc_cv1, imp_mc, imp_mc_upper, imp_mc_lower

    else:
        return p_mc_cv1, std_mc_cv1


def gauss_dens(mu, sigma, x):
    return 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


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
    density = gauss_dens(0.0, 1, y)
    Ft = np.sum(density * vix_T * w)
    vix_strike = Ft * vix_strike_perc
    option_prices = np.sum(
        density * np.maximum(vix_T - vix_strike.reshape(-1, 1), 0) * w, axis=1
    )

    if compute_iv:
        imp_vol = black_impvol(
            value=option_prices, F=Ft, K=vix_strike_perc * Ft, T=T, opttype=1
        )
        return Ft * 100, option_prices * 100, imp_vol
    else:
        return Ft * 100, option_prices * 100
