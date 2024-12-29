import enum


class RegressionType(enum.Enum):
    RDD = 0  # regression discontinuity design: discontinuity in level
    RKD = 1  # regression kink design: discontinuity in first derivative


class PolynomialDegree(enum.Enum):
    linear = 1
    quad = 2
    cubic = 3


class Kernel(enum.Enum):
    uniform = enum.auto()
    triangular = enum.auto()
    epanechnikov = enum.auto()


class BandwidthSelector(enum.Enum):
    mse = enum.auto()  # mean squared error (MSE)-optimal bandwidth selector
    cer = enum.auto()  # coverage error rate (CER)-optimal bandwidth selector


class BandwidthSelectorFunction(enum.Enum):
    rd = enum.auto()  # one common bandwidth
    two = enum.auto()  # two different bandwidths (below and above the cutoff)
    sum = enum.auto()  # one common bandwidth for the sum of regression estimates (as opposed to difference thereof)
    comb1 = enum.auto()  # minimum of 'rd' and 'sum'
    comb2 = enum.auto()  # median of 'two', 'rd', and 'sum' for each side of the cutoff separately


class Vce(enum.Enum):
    nn = enum.auto()  # heteroskedasticity-robust nearest neighbor variance estimator with nnmatch the (minimum) number of neighbors to be used
    hc0 = enum.auto()  # heteroskedasticity-robust plug-in residuals variance estimator without weights
    hc1 = enum.auto()  # heteroskedasticity-robust plug-in residuals variance estimator with hc1 weights
    hc2 = enum.auto()  # heteroskedasticity-robust plug-in residuals variance estimator with hc2 weights
    hc3 = enum.auto()  # heteroskedasticity-robust plug-in residuals variance estimator with hc3 weights


class BwSelectRdpermute(enum.Enum):
    cct = enum.auto()