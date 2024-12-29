import typing
import multiprocessing
import numpy as np
import pandas as pd
import rdrobust
from rdrobust.funs import rdrobust_output

from rdpermute.enums import RegressionType, Kernel, BandwidthSelector, \
    BandwidthSelectorFunction, Vce, PolynomialDegree


def rdpermute(
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
) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform permutation test proposed by Ganong and Jäger (2018) https://www.tandfonline.com/doi/full/10.1080/01621459.2017.1328356

    Parameters
    ----------
    y: np.typing.ArrayLike
    x: np.typing.ArrayLike
    true_cutoff: float
    placebos: np.typing.ArrayLike
    regression_type: RegressionType
    polynomial_degree: PolynomialDegree
    polynomial_degree_bias: typing.Optional[int]
    kernel: Kernel
    bandwidth: typing.Optional[float]
    bandwidth_selector: BandwidthSelector
    bandwidth_selector_function: BandwidthSelectorFunction
    vce: Vce
    nnmatch: int
    fuzzy: typing.Optional[np.typing.ArrayLike]
    covs: typing.Optional[np.typing.ArrayLike]
    weights: typing.Optional[np.typing.ArrayLike]
    number_workers: typing.Optional[int]

    Returns
    -------
    typing.Tuple[pd.DataFrame, pd.DataFrame]

    """
    # limit floating point precisions
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
    # add randomization inference
    results = _randomization_inference(
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
        # masspoints='off',  # TODO: silence notifications while adjusting mass points
    )
    result = _process(result, name=cutoff)
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
    result.rename(name, inplace=True)
    return result


def _randomization_inference(
        results: pd.DataFrame,
        cutoff: float = 0,
) -> pd.DataFrame:
    ranks = results.loc[
        pd.IndexSlice[r'$\beta_1$', :]
    ].rank(
        axis=1
    )
    p_value = 2 * np.fmin(
        ranks[cutoff].divide(ranks.shape[1]),
        1 - ranks[cutoff].subtract(1).divide(ranks.shape[1])
    )
    p_value = pd.concat(
        {r'$p$-value (randomization)': p_value},
        axis=0,
        names=results.index.names,
    )
    results = pd.concat([results[cutoff], p_value], axis=0)
    return results
