import os
import numpy as np
import pandas as pd

import tests
import rdpermute
from rdpermute.enums import RegressionType, BwSelectRdpermute, \
    PolynomialDegree, MassPoints, EstimationProcedure


def _stata_command(
        y: str,
        x: str,
        true_cutoff: float,
        placebo_left: float,
        placebo_right: float,
        placebo_step: float,
        regression_type: RegressionType,
        polynomial_degree: PolynomialDegree = PolynomialDegree.linear,
        bwselect: BwSelectRdpermute = BwSelectRdpermute.cct,
        silent: bool = True
):
    base = f'rdpermute {y} {x}'
    options = [
        f'placebo_disconts({placebo_left}({placebo_step}){placebo_right})',
        f'true_discont({true_cutoff})',
        f'deriv_discont({regression_type.value})',
        f'{polynomial_degree.name}',
        f'bw({bwselect.name})',
        'silent' if silent else None,
    ]
    command = ', '.join(
        [
            base,
            ' '.join(options),
        ]
    )
    return command


def _run_stata(
        data: pd.DataFrame,
        **command_kwargs,
):
    tests.stata.pdataframe_to_frame(data, stfr='data', force=True)
    command = _stata_command(**command_kwargs)
    tests.stata.run(f'frame data: {command}')
    results = tests.stata.get_ereturn()
    return results


def _test(
        fn: str,
        y: str,
        x: str,
        true_cutoff: float,
        placebo_left: float,
        placebo_right: float,
        placebo_step: float,
        regression_type: RegressionType,
        polynomial_degree: PolynomialDegree = PolynomialDegree.linear,
        bwselect: BwSelectRdpermute = BwSelectRdpermute.cct,
        estimation: EstimationProcedure = EstimationProcedure.robust,
        relative_tolerance: float = 1e-1,  # TODO: very high tolerance; is this due to differences between the Stata and Python versions of rdrobust?
):
    # prepare data and commands
    data = pd.read_stata(os.path.join(tests.path_to_data, fn))
    command_kwargs = dict(
        y=y,
        x=x,
        true_cutoff=true_cutoff,
        placebo_left=placebo_left,
        placebo_right=placebo_right,
        placebo_step=placebo_step,
        regression_type=regression_type,
        polynomial_degree=polynomial_degree,
        bwselect=bwselect,
    )
    # run stata and collect results
    results_stata = _run_stata(
        data=data,
        **command_kwargs,
    )
    # run python and collect results
    results_python, results_python_placebos = rdpermute.rdpermute(
        y=data[command_kwargs['y']],
        x=data[command_kwargs['x']],
        true_cutoff=command_kwargs['true_cutoff'],
        placebos=np.arange(
            start=command_kwargs['placebo_left'],
            stop=command_kwargs['placebo_right'] + command_kwargs[
                'placebo_step'],
            step=command_kwargs['placebo_step'],
        ),
        regression_type=command_kwargs['regression_type'],
        polynomial_degree=command_kwargs['polynomial_degree'],
        masspoints=MassPoints.adjust,
        estimation=estimation,
        alpha=None,  # Stata rdpermute does not estimate confidence intervals
    )
    # assert results are sufficiently close
    correspondence = {
        'e(bw_linear)': 'bandwidth (left)',
        'e(kink_beta_linear)': r'$\beta_1$',
        'e(kink_se_linear)': 'SE',
    }
    for name_stata, name_python in correspondence.items():
        np.testing.assert_allclose(
            results_stata[name_stata].flatten(),
            results_python_placebos.loc[name_python],
            rtol=relative_tolerance,
            err_msg=f'{name_stata} differs between Stata and Python',
        )

    np.testing.assert_allclose(
        results_stata['e(pval_linear)'][-1, -1],
        results_python.loc[r'$p$-value (randomization)'],
        rtol=relative_tolerance,
        err_msg=f'Randomization inference p-values do not match between Stata and Python'
    )


def test_lee_election():
    _test(
        fn='lee_election.dta',  # https://github.com/ganong-noel/rdpermute/blob/8ae72bc299e496b48cd0a5203330cb4656c3922a/example_data/lee_election.dta
        y='y',
        x='x',
        true_cutoff=0.0,
        placebo_left=-50,
        placebo_right=49,
        placebo_step=1,
        regression_type=RegressionType.RDD,
    )


def test_sim1():
    _test(
        fn='sim1.dta',  # https://github.com/ganong-noel/rdpermute/blob/8ae72bc299e496b48cd0a5203330cb4656c3922a/example_data/sim1.dta
        y='y',
        x='x',
        true_cutoff=0.0,
        placebo_left=-0.9,
        placebo_right=0.9,
        placebo_step=0.02,
        regression_type=RegressionType.RKD,
    )


def test_sim2():
    _test(
        fn='sim2.dta',  # https://github.com/ganong-noel/rdpermute/blob/8ae72bc299e496b48cd0a5203330cb4656c3922a/example_data/sim2.dta
        y='y',
        x='x',
        true_cutoff=0.0,
        placebo_left=-0.9,
        placebo_right=0.9,
        placebo_step=0.02,
        regression_type=RegressionType.RKD,
    )
