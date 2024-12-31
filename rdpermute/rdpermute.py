import typing
import multiprocessing
import numpy as np
import pandas as pd
import scipy.stats
import rdrobust
from rdrobust.funs import rdrobust_output

from rdpermute.enums import RegressionType, Kernel, BandwidthSelector, \
    BandwidthSelectorFunction, Vce, PolynomialDegree, MassPoints, Bound, \
    EstimationProcedure


def rdpermute(
        y: np.typing.ArrayLike,
        x: np.typing.ArrayLike,
        true_cutoff: float,
        placebos: np.typing.ArrayLike,
        alpha: typing.Optional[float] = 0.05,
        regression_type: RegressionType = RegressionType.RDD,
        polynomial_degree: PolynomialDegree = PolynomialDegree.linear,
        polynomial_degree_bias: typing.Optional[int] = None,
        kernel: Kernel = Kernel.triangular,
        bandwidth: typing.Optional[float] = None,
        bandwidth_selector: BandwidthSelector = BandwidthSelector.mse,
        bandwidth_selector_function: BandwidthSelectorFunction = BandwidthSelectorFunction.rd,
        vce: Vce = Vce.nn,
        nnmatch: int = 3,
        fuzzy: typing.Optional[np.typing.ArrayLike] = None,
        covs: typing.Optional[np.typing.ArrayLike] = None,
        weights: typing.Optional[np.typing.ArrayLike] = None,
        number_workers: typing.Optional[int] = -1,
        masspoints: MassPoints = MassPoints.off,
        estimation: EstimationProcedure = EstimationProcedure.robust,
        max_iter: int = 1_00,
) -> typing.Tuple[pd.Series, pd.DataFrame]:
    """
    Perform permutation test proposed by Ganong and Jäger (2018) https://www.tandfonline.com/doi/full/10.1080/01621459.2017.1328356

    Parameters
    ----------
    y: np.typing.ArrayLike
        The dependent variable
    x: np.typing.ArrayLike
        The independent variable (aka running variable or forcing variable)
    true_cutoff: float
        The regression discontinuity/kink point
    placebos: np.typing.ArrayLike
        The locations of placebo regression discontinuity/kink points
    alpha: typing.Optional[float]
        Optional confidence level for the construction of confidence intervals
    regression_type: RegressionType
        The regression design (either discontinuity or kink)
    polynomial_degree: PolynomialDegree
        The degree of the local polynomials fitted to the left and right of the cutoff
    polynomial_degree_bias: typing.Optional[int]
        The degree of the local polynomials used for bias-correction (Calonico et al, 2014) <https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA11757>
    kernel: Kernel
        The kernel function used to construct the local polynomials
    bandwidth: typing.Optional[float]
        The bandwidth around the cutoff
    bandwidth_selector: BandwidthSelector
        The bandwidth selection procedure (Calonico et al, 2020) <https://doi.org/10.1093/ectj/utz022>
    bandwidth_selector_function: BandwidthSelectorFunction
        The bandwidth selector function
    vce: Vce
        The procedure used to compute the variance-covariance matrix estimator
    nnmatch: int
        Indicates the minimum number of neighbors to be used for the heteroskedasticity-robust nearest neighbor variance estimator. To be combined with vce=nn
    fuzzy: typing.Optional[np.typing.ArrayLike]
        The treatment status variable used to implement fuzzy RDD or RKD estimation
    covs: typing.Optional[np.typing.ArrayLike]
        Additional covariates to be used for estimation and inference
    weights: typing.Optional[np.typing.ArrayLike]
        Unit-specific weights applied to the kernel function
    number_workers: typing.Optional[int]
        Number of workers
    masspoints: MassPoints
        Check and control for repeated observations in the running variable
    estimation: EstimationProcedure
        Estimation procedure used in rdrobust
    max_iter: int
        Maximum number of iterations in the construction of confidence intervals

    Returns
    -------
    typing.Tuple[pd.Series, pd.DataFrame]

    """
    results, results_placebos = _run(
        y=y,
        x=x,
        true_cutoff=true_cutoff,
        placebos=placebos,
        regression_type=regression_type,
        polynomial_degree=polynomial_degree,
        polynomial_degree_bias=polynomial_degree_bias,
        kernel=kernel,
        bandwidth=bandwidth,
        bandwidth_selector=bandwidth_selector,
        bandwidth_selector_function=bandwidth_selector_function,
        vce=vce,
        nnmatch=nnmatch,
        fuzzy=fuzzy,
        covs=covs,
        weights=weights,
        number_workers=number_workers,
        masspoints=masspoints,
        estimation=estimation,
    )
    if alpha is not None:
        # invert permutation test to construct confidence intervals
        confidence_interval = _randomization_interval(
            y=y,
            x=x,
            true_cutoff=true_cutoff,
            placebos=placebos,
            beta_asymptotic=results.loc[r'$\beta_1$'],
            se_asymptotic=results.loc['SE'],
            regression_type=regression_type,
            alpha=alpha,
            max_iter=max_iter,
        )
        results = pd.concat([results, confidence_interval], axis=0)
    results.rename(estimation.value, inplace=True)
    return results, results_placebos


def _run(
        y: np.typing.ArrayLike,
        x: np.typing.ArrayLike,
        true_cutoff: float,
        placebos: np.typing.ArrayLike,
        regression_type: RegressionType = RegressionType.RDD,
        polynomial_degree: PolynomialDegree = PolynomialDegree.linear,
        polynomial_degree_bias: typing.Optional[int] = None,
        kernel: Kernel = Kernel.triangular,
        bandwidth: typing.Optional[float] = None,
        bandwidth_selector: BandwidthSelector = BandwidthSelector.mse,
        bandwidth_selector_function: BandwidthSelectorFunction = BandwidthSelectorFunction.rd,
        vce: Vce = Vce.nn,
        nnmatch: int = 3,
        fuzzy: typing.Optional[np.typing.ArrayLike] = None,
        covs: typing.Optional[np.typing.ArrayLike] = None,
        weights: typing.Optional[np.typing.ArrayLike] = None,
        number_workers: typing.Optional[int] = -1,
        masspoints: MassPoints = MassPoints.off,
        estimation: EstimationProcedure = EstimationProcedure.robust,
) -> typing.Tuple[pd.Series, pd.DataFrame]:
    # limit floating point precision
    placebos = np.round(placebos, decimals=10)
    if true_cutoff not in placebos:
        # ensure true cutoff is part of placebo tests
        placebos = np.append(placebos, true_cutoff)
    args = (
        y,
        x,
        regression_type,
        polynomial_degree,
        polynomial_degree_bias,
        kernel,
        bandwidth,
        bandwidth_selector,
        bandwidth_selector_function,
        vce,
        nnmatch,
        fuzzy,
        covs,
        weights,
        masspoints,
        estimation,
    )
    if number_workers is not None:
        # use multiprocessing to run placebos in parallel
        results_placebos = _run_parallel(
            *args,
            placebos=placebos,
            number_workers=number_workers,
        )
    else:
        # loop over placebos sequentially
        results_placebos = _run_sequential(
            *args,
            placebos=placebos,
        )
    # collect results
    results_placebos = pd.concat(
        results_placebos,
        axis=1,
        names=['placebo'],
    )
    # randomization inference
    results = _randomization_pvalue(
        results=results_placebos,
        cutoff=true_cutoff,
    )
    return results, results_placebos


def _run_single(
        cutoff: float,
        y: np.typing.ArrayLike,
        x: np.typing.ArrayLike,
        regression_type: RegressionType = RegressionType.RDD,
        polynomial_degree: PolynomialDegree = PolynomialDegree.linear,
        polynomial_degree_bias: typing.Optional[int] = None,
        kernel: Kernel = Kernel.triangular,
        bandwidth: typing.Optional[float] = None,
        bandwidth_selector: BandwidthSelector = BandwidthSelector.mse,
        bandwidth_selector_function: BandwidthSelectorFunction = BandwidthSelectorFunction.rd,
        vce: Vce = Vce.nn,
        nnmatch: int = 3,
        fuzzy: typing.Optional[np.typing.ArrayLike] = None,
        covs: typing.Optional[np.typing.ArrayLike] = None,
        weights: typing.Optional[np.typing.ArrayLike] = None,
        masspoints: MassPoints = MassPoints.off,
        estimation: EstimationProcedure = EstimationProcedure.robust,
) -> pd.Series:
    bwselect = f'{bandwidth_selector.name}{bandwidth_selector_function.name}'
    result = rdrobust.rdrobust(
        y=y,
        x=x,
        c=cutoff,
        p=polynomial_degree.value,
        q=polynomial_degree_bias,
        deriv=regression_type.value,
        kernel=kernel.name,
        h=bandwidth,
        bwselect=bwselect,
        b=bwselect,
        vce=vce.name,
        nnmatch=nnmatch,
        fuzzy=fuzzy,
        covs=covs,
        weights=weights,
        masspoints=masspoints.name,
    )
    result = _process(result, name=cutoff, estimation=estimation)
    return result


def _run_sequential(
        *args,
        placebos: np.typing.ArrayLike,
):
    results = {
        placebo: _run_single(placebo, *args)
        for placebo in placebos
    }
    return results


def _run_parallel(
        *args,
        placebos: np.typing.ArrayLike,
        number_workers: int = -1,
) -> pd.DataFrame:
    # prepare arguments for multiprocessing.pool.Pool.starmap
    # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool.starmap
    args = [
        (placebo, *args) for placebo in placebos
    ]
    if number_workers < 0:
        # use all available cores
        number_workers = multiprocessing.cpu_count()

    with multiprocessing.Pool(number_workers) as pool:
        results = pool.starmap(
            func=_run_single,
            iterable=args,
        )
    return results


def _process(
        result: rdrobust_output,
        name: typing.Hashable,
        estimation: EstimationProcedure = EstimationProcedure.robust,
) -> pd.Series:
    result = pd.concat(
        {
            r'$\beta_1$': pd.Series(result.coef.squeeze()),
            'SE': pd.Series(result.se.squeeze()),
            r'$t$-stat': pd.Series(result.t.squeeze()),
            r'$p$-value': pd.Series(result.pv.squeeze()),
            'observations (left)': pd.Series(
                result.N_h[0],
                index=result.coef.index,
            ),
            'observations (right)': pd.Series(
                result.N_h[1],
                index=result.coef.index,
            ),
            'bandwidth (left)': pd.Series(
                result.bws.loc['h', 'left'],
                index=result.coef.index,
            ),
            'bandwidth (right)': pd.Series(
                result.bws.loc['h', 'right'],
                index=result.coef.index,
            ),
        },
        axis=0,
        names=['parameter', 'type']
    )
    result = result.xs(estimation.value, level='type')
    result.rename(name, inplace=True)
    return result


def _randomization_pvalue(
        results: pd.DataFrame,
        cutoff: float = 0.0,
) -> pd.Series:
    ranks = results.loc[r'$\beta_1$'].rank()
    # https://github.com/ganong-noel/rdpermute/blob/8ae72bc299e496b48cd0a5203330cb4656c3922a/stata_code/rdpermute.ado#L477
    p_value = 2 * np.fmin(
        ranks[cutoff] / ranks.shape[0],
        1 - (ranks[cutoff] - 1) / ranks.shape[0]
    )
    tmp = results[cutoff]
    tmp.loc[r'$p$-value (randomization)'] = p_value
    return tmp


def _randomization_interval(
        y: np.typing.ArrayLike,
        x: np.typing.ArrayLike,
        true_cutoff: float,
        placebos: np.typing.ArrayLike,
        beta_asymptotic: float,
        se_asymptotic: float,
        regression_type: RegressionType = RegressionType.RKD,
        alpha: typing.Optional[float] = 0.05,
        size_multiplicator: float = 10,
        convergence_threshold: float = 0.05,
        max_iter: int = 1_00,
) -> pd.Series:
    # following Appendix C of (Ganong and Jäger, 2018)
    # https://www.tandfonline.com/doi/full/10.1080/01621459.2017.1328356
    # step 1: identify search region
    bounds = {
        bound: _get_bound(
            beta_asymptotic=beta_asymptotic,
            se_asymptotic=se_asymptotic,
            y=y,
            x=x,
            true_cutoff=true_cutoff,
            placebos=placebos,
            bound=bound,
            size_multiplicator=size_multiplicator,
            alpha=alpha,
            regression_type=regression_type,
            max_iter=max_iter,
        ) for bound in Bound
    }
    # step 2: bisection method
    if convergence_threshold is None:
        # (Ganong and Jäger, 2018): "We define the search as having converged if two subsequent midpoints are less than one tenth the magnitude of the asymptotic confidence interval apart from one another"
        convergence_threshold = 2 * (
            se_asymptotic * scipy.stats.norm.ppf(1 - alpha / 2)
        ) / 10

    confidence_interval = {}
    for bound_enum, bound_value in bounds.items():
        order = bound_enum.value
        interval = [beta_asymptotic, bound_value][::order]  # reverse if left bound
        treatment_effect = 0.5 * (interval[0] + interval[1])
        converged = False
        iteration = 0
        while iteration < max_iter:
            iteration += 1
            y_transformed = _transform(
                y=y,
                x=x,
                treatment_effect=treatment_effect,
                cutoff=true_cutoff,
                regression_type=regression_type,
            )
            tmp, _ = _run(
                y=y_transformed,
                x=x,
                true_cutoff=true_cutoff,
                placebos=placebos,
                regression_type=regression_type,
            )
            pval = tmp.loc[r'$p$-value (randomization)']
            if  order * pval < order * alpha:
                interval = [interval[0], treatment_effect]
            else:
                interval = [treatment_effect, interval[1]]
            treatment_effect_old = treatment_effect
            treatment_effect = 0.5 * (interval[0] + interval[1])
            if abs(treatment_effect - treatment_effect_old) < convergence_threshold:
                converged = True
                break

        if not converged:
            raise ConvergenceError(
                'Maximum number of iterations reached before convergence.'
            )
        confidence_interval.update(
            {f'ci {bound_enum.name} {1-alpha:.1%} (randomization)': treatment_effect}
        )
    confidence_interval = pd.Series(confidence_interval)
    return confidence_interval


def _get_bound(
        beta_asymptotic: float,
        se_asymptotic: float,
        y: np.typing.ArrayLike,
        x: np.typing.ArrayLike,
        true_cutoff: float,
        placebos: np.typing.ArrayLike,
        bound: Bound = Bound.left,
        size_multiplicator: float = 10,
        alpha: float = 0.05,
        regression_type: RegressionType = RegressionType.RKD,
        max_iter: int = 1_00,
) -> float:
    beta = se_asymptotic
    converged = False
    iteration = 0
    while iteration < max_iter:
        iteration += 1
        beta *= size_multiplicator
        treatment_effect = _update_treatment_effect(
            beta=beta_asymptotic,
            update=beta,
            bound=bound,
        )
        y_transformed = _transform(
            y=y,
            x=x,
            treatment_effect=treatment_effect,
            cutoff=true_cutoff,
            regression_type=regression_type,
        )
        tmp, _ = _run(
            y=y_transformed,
            x=x,
            true_cutoff=true_cutoff,
            placebos=placebos,
            regression_type=regression_type,
        )
        if tmp.loc[r'$p$-value (randomization)'] < alpha:
            converged = True
            break
    if not converged:
        raise ConvergenceError(
            'Maximum number of iterations reached before convergence.'
        )
    return treatment_effect


def _update_treatment_effect(
        beta: float,
        update: float,
        bound: Bound,
) -> float:
    return beta + bound.value * update


def _transform(
        y: np.typing.ArrayLike,
        x: np.typing.ArrayLike,
        treatment_effect: float,
        cutoff: float,
        regression_type: RegressionType = RegressionType.RKD,
) -> np.typing.ArrayLike:
    if regression_type is RegressionType.RKD:
        # constant additive in slope
        treatment = treatment_effect * x
    elif regression_type is RegressionType.RDD:
        # constant additive in levels
        # https://github.com/rdpackages/rdlocrand/blob/99d2030a1860454fe78bf393a1a0965aabdabf48/Python/rdlocrand/src/rdlocrand/rdrandinf.py#L388
        treatment = treatment_effect
    else:
        raise NotImplementedError(regression_type)

    transformed = np.where(x < cutoff, y, y - treatment)
    return transformed


class ConvergenceError(Exception):
    pass