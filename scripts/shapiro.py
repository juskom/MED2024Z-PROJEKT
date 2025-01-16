from __future__ import annotations
from collections import namedtuple
import warnings

import numpy as np
from numpy import (zeros, sort, array)

from scipy import optimize, interpolate

ShapiroResult = namedtuple('ShapiroResult', ('statistic', 'pvalue'))

def shapiro(x):
    r"""Perform the Shapiro-Wilk test for normality.

    The Shapiro-Wilk test tests the null hypothesis that the
    data was drawn from a normal distribution.

    Parameters
    ----------
    x : array_like
        Array of sample data.

    Returns
    -------
    statistic : float
        The test statistic.
    p-value : float
        The p-value for the hypothesis test.

    See Also
    --------
    anderson : The Anderson-Darling test for normality
    kstest : The Kolmogorov-Smirnov test for goodness of fit.

    Notes
    -----
    The algorithm used is described in [4]_ but censoring parameters as
    described are not implemented. For N > 5000 the W test statistic is
    accurate, but the p-value may not be.

    References
    ----------
    .. [1] https://www.itl.nist.gov/div898/handbook/prc/section2/prc213.htm
    .. [2] Shapiro, S. S. & Wilk, M.B (1965). An analysis of variance test for
           normality (complete samples), Biometrika, Vol. 52, pp. 591-611.
    .. [3] Razali, N. M. & Wah, Y. B. (2011) Power comparisons of Shapiro-Wilk,
           Kolmogorov-Smirnov, Lilliefors and Anderson-Darling tests, Journal of
           Statistical Modeling and Analytics, Vol. 2, pp. 21-33.
    .. [4] ALGORITHM AS R94 APPL. STATIST. (1995) VOL. 44, NO. 4.
    .. [5] B. Phipson and G. K. Smyth. "Permutation P-values Should Never Be
           Zero: Calculating Exact P-values When Permutations Are Randomly
           Drawn." Statistical Applications in Genetics and Molecular Biology
           9.1 (2010).
    .. [6] Panagiotakos, D. B. (2008). The value of p-value in biomedical
           research. The open cardiovascular medicine journal, 2, 97.

    Examples
    --------
    Suppose we wish to infer from measurements whether the weights of adult
    human males in a medical study are not normally distributed [2]_.
    The weights (lbs) are recorded in the array ``x`` below.

    >>> import numpy as np
    >>> x = np.array([148, 154, 158, 160, 161, 162, 166, 170, 182, 195, 236])

    The normality test of [1]_ and [2]_ begins by computing a statistic based
    on the relationship between the observations and the expected order
    statistics of a normal distribution.

    >>> from scipy import stats
    >>> res = stats.shapiro(x)
    >>> res.statistic
    0.7888147830963135

    The value of this statistic tends to be high (close to 1) for samples drawn
    from a normal distribution.

    The test is performed by comparing the observed value of the statistic
    against the null distribution: the distribution of statistic values formed
    under the null hypothesis that the weights were drawn from a normal
    distribution. For this normality test, the null distribution is not easy to
    calculate exactly, so it is usually approximated by Monte Carlo methods,
    that is, drawing many samples of the same size as ``x`` from a normal
    distribution and computing the values of the statistic for each.

    >>> def statistic(x):
    ...     # Get only the `shapiro` statistic; ignore its p-value
    ...     return stats.shapiro(x).statistic
    >>> ref = stats.monte_carlo_test(x, stats.norm.rvs, statistic,
    ...                              alternative='less')
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(figsize=(8, 5))
    >>> bins = np.linspace(0.65, 1, 50)
    >>> def plot(ax):  # we'll re-use this
    ...     ax.hist(ref.null_distribution, density=True, bins=bins)
    ...     ax.set_title("Shapiro-Wilk Test Null Distribution \n"
    ...                  "(Monte Carlo Approximation, 11 Observations)")
    ...     ax.set_xlabel("statistic")
    ...     ax.set_ylabel("probability density")
    >>> plot(ax)
    >>> plt.show()

    The comparison is quantified by the p-value: the proportion of values in
    the null distribution less than or equal to the observed value of the
    statistic.

    >>> fig, ax = plt.subplots(figsize=(8, 5))
    >>> plot(ax)
    >>> annotation = (f'p-value={res.pvalue:.6f}\n(highlighted area)')
    >>> props = dict(facecolor='black', width=1, headwidth=5, headlength=8)
    >>> _ = ax.annotate(annotation, (0.75, 0.1), (0.68, 0.7), arrowprops=props)
    >>> i_extreme = np.where(bins <= res.statistic)[0]
    >>> for i in i_extreme:
    ...     ax.patches[i].set_color('C1')
    >>> plt.xlim(0.65, 0.9)
    >>> plt.ylim(0, 4)
    >>> plt.show
    >>> res.pvalue
    0.006703833118081093

    If the p-value is "small" - that is, if there is a low probability of
    sampling data from a normally distributed population that produces such an
    extreme value of the statistic - this may be taken as evidence against
    the null hypothesis in favor of the alternative: the weights were not
    drawn from a normal distribution. Note that:

    - The inverse is not true; that is, the test is not used to provide
      evidence *for* the null hypothesis.
    - The threshold for values that will be considered "small" is a choice that
      should be made before the data is analyzed [5]_ with consideration of the
      risks of both false positives (incorrectly rejecting the null hypothesis)
      and false negatives (failure to reject a false null hypothesis).

    """
    x = np.ravel(x)

    N = len(x)
    if N < 3:
        raise ValueError("Data must be at least length 3.")

    x = x - np.median(x)

    a = zeros(N, 'f')
    init = 0

    y = sort(x)
    a, w, pw, ifault = _statlib.swilk(y, a[:N//2], init)
    if ifault not in [0, 2]:
        warnings.warn("Input data for shapiro has range zero. The results "
                      "may not be accurate.")
    if N > 5000:
        warnings.warn("p-value may not be accurate for N > 5000.")

    # `swilk` can return negative p-values for N==3; see gh-18322.
    if N == 3:
        # Potential improvement: precision for small p-values
        pw = 1 - 6/np.pi*np.arccos(np.sqrt(w))
    return ShapiroResult(w, pw)


# Values from Stephens, M A, "EDF Statistics for Goodness of Fit and
#             Some Comparisons", Journal of the American Statistical
#             Association, Vol. 69, Issue 347, Sept. 1974, pp 730-737
_Avals_norm = array([0.576, 0.656, 0.787, 0.918, 1.092])
_Avals_expon = array([0.922, 1.078, 1.341, 1.606, 1.957])
# From Stephens, M A, "Goodness of Fit for the Extreme Value Distribution",
#             Biometrika, Vol. 64, Issue 3, Dec. 1977, pp 583-588.
_Avals_gumbel = array([0.474, 0.637, 0.757, 0.877, 1.038])
# From Stephens, M A, "Tests of Fit for the Logistic Distribution Based
#             on the Empirical Distribution Function.", Biometrika,
#             Vol. 66, Issue 3, Dec. 1979, pp 591-595.
_Avals_logistic = array([0.426, 0.563, 0.660, 0.769, 0.906, 1.010])
# From Richard A. Lockhart and Michael A. Stephens "Estimation and Tests of
#             Fit for the Three-Parameter Weibull Distribution"
#             Journal of the Royal Statistical Society.Series B(Methodological)
#             Vol. 56, No. 3 (1994), pp. 491-500, table 1. Keys are c*100
_Avals_weibull = [[0.292, 0.395, 0.467, 0.522, 0.617, 0.711, 0.836, 0.931],
                  [0.295, 0.399, 0.471, 0.527, 0.623, 0.719, 0.845, 0.941],
                  [0.298, 0.403, 0.476, 0.534, 0.631, 0.728, 0.856, 0.954],
                  [0.301, 0.408, 0.483, 0.541, 0.640, 0.738, 0.869, 0.969],
                  [0.305, 0.414, 0.490, 0.549, 0.650, 0.751, 0.885, 0.986],
                  [0.309, 0.421, 0.498, 0.559, 0.662, 0.765, 0.902, 1.007],
                  [0.314, 0.429, 0.508, 0.570, 0.676, 0.782, 0.923, 1.030],
                  [0.320, 0.438, 0.519, 0.583, 0.692, 0.802, 0.947, 1.057],
                  [0.327, 0.448, 0.532, 0.598, 0.711, 0.824, 0.974, 1.089],
                  [0.334, 0.469, 0.547, 0.615, 0.732, 0.850, 1.006, 1.125],
                  [0.342, 0.472, 0.563, 0.636, 0.757, 0.879, 1.043, 1.167]]
_Avals_weibull = np.array(_Avals_weibull)
_cvals_weibull = np.linspace(0, 0.5, 11)
_get_As_weibull = interpolate.interp1d(_cvals_weibull, _Avals_weibull.T,
                                       kind='linear', bounds_error=False,
                                       fill_value=_Avals_weibull[-1])


def _weibull_fit_check(params, x):
    # Refine the fit returned by `weibull_min.fit` to ensure that the first
    # order necessary conditions are satisfied. If not, raise an error.
    # Here, use `m` for the shape parameter to be consistent with [7]
    # and avoid confusion with `c` as defined in [7].
    n = len(x)
    m, u, s = params

    def dnllf_dm(m, u):
        # Partial w.r.t. shape w/ optimal scale. See [7] Equation 5.
        xu = x-u
        return (1/m - (xu**m*np.log(xu)).sum()/(xu**m).sum()
                + np.log(xu).sum()/n)

    def dnllf_du(m, u):
        # Partial w.r.t. loc w/ optimal scale. See [7] Equation 6.
        xu = x-u
        return (m-1)/m*(xu**-1).sum() - n*(xu**(m-1)).sum()/(xu**m).sum()

    def get_scale(m, u):
        # Partial w.r.t. scale solved in terms of shape and location.
        # See [7] Equation 7.
        return ((x-u)**m/n).sum()**(1/m)

    def dnllf(params):
        # Partial derivatives of the NLLF w.r.t. parameters, i.e.
        # first order necessary conditions for MLE fit.
        return [dnllf_dm(*params), dnllf_du(*params)]

    suggestion = ("Maximum likelihood estimation is known to be challenging "
                  "for the three-parameter Weibull distribution. Consider "
                  "performing a custom goodness-of-fit test using "
                  "`scipy.stats.monte_carlo_test`.")

    if np.allclose(u, np.min(x)) or m < 1:
        # The critical values provided by [7] don't seem to control the
        # Type I error rate in this case. Error out.
        message = ("Maximum likelihood estimation has converged to "
                   "a solution in which the location is equal to the minimum "
                   "of the data, the shape parameter is less than 2, or both. "
                   "The table of critical values in [7] does not "
                   "include this case. " + suggestion)
        raise ValueError(message)

    try:
        # Refine the MLE / verify that first-order necessary conditions are
        # satisfied. If so, the critical values provided in [7] seem reliable.
        with np.errstate(over='raise', invalid='raise'):
            res = optimize.root(dnllf, params[:-1])

        message = ("Solution of MLE first-order conditions failed: "
                   f"{res.message}. `anderson` cannot continue. " + suggestion)
        if not res.success:
            raise ValueError(message)

    except (FloatingPointError, ValueError) as e:
        message = ("An error occurred while fitting the Weibull distribution "
                   "to the data, so `anderson` cannot continue. " + suggestion)
        raise ValueError(message) from e

    m, u = res.x
    s = get_scale(m, u)
    return m, u, s