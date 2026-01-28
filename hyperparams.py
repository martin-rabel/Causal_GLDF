from . import data_processing


r"""Submodule for hyperparameter helpers
    ------------------------------------

    Typically the user should configure the marked-independence stages through these helpers.
    To configure stage XYZ use ForXYZ.Configure\\ *subtype*\\ .
"""

import numpy as np
from scipy.stats import binom
from scipy.optimize import root_scalar


class Hyperparams_HomogeneityBinomial_ParCorr(data_processing.IProvideHyperparamsForBinomial):
    """Helper to configure standard hyper-parameter sets for binomial homogeneity test when using partial correlation."""

    def __init__(self, alpha_error_control: float=0.05, min_block_count: int=5, min_effective_sample_count: int=5, regimes_are_large:bool=True):
        """Construct Hyper-parameter set for given parameters.

        :param alpha_error_control: Error-control target :math:`\\alpha`, defaults to 0.05
        :type alpha_error_control: float, optional
        :param min_block_count: Pick block-size to ensure minimum number of blocks, defaults to 5
        :type min_block_count: int, optional
        :param min_effective_sample_count: Pick block-size to ensure minimum effective sample-size per block (for partial correlation), defaults to 5
        :type min_effective_sample_count: int, optional
        :param regimes_are_large: Consider hyper-parameters for large regimes, defaults to True
        :type regimes_are_large: bool, optional
        """
        self.alpha = alpha_error_control
        self.min_block_count = min_block_count
        self.min_effective_sample_count = min_effective_sample_count
        self.tolerance = 1e-5
        self.validate_choice = True
        self.regimes_are_large = regimes_are_large
        self.cache : dict[(int,int),data_processing.IProvideHyperparamsForBinomial.Hyperparams] = dict()    #: Set to None to disable cache.


    def hyperparams_for_binomial(self, N: int, dim_Z: int) -> data_processing.IProvideHyperparamsForBinomial.Hyperparams:
        result = None
        if self.cache is not None:
            result = self.cache.get((N,dim_Z))
            if result is None:
                result = self.compute_for_fixed_sample_count(N=N, dim_Z=dim_Z)
                self.cache[(N,dim_Z)] = result
            return result
        else:
            return self.compute_for_fixed_sample_count(N=N, dim_Z=dim_Z)
        


    @staticmethod
    def get_opt_beta(alpha_homogeneity_err_control_requested: float, block_count: int, beta_start_value: float=0.1) -> tuple[float, int]:
        """
        The binomial test rejects at an integer count. To target a specific error-rate :math:`\\alpha`, this function modifies the quantile :math:`\\beta`-start-value such that
        that for the returned :math:`\\beta'\\geq\\beta` will realize the error-rate :math:`\\alpha` for the (also returned) integer cutoff-count.

        :param alpha_homogeneity_err_control_requested: Requrested error-control :math:`\\alpha`.
        :type alpha_homogeneity_err_control_requested: float
        :param block_count: The total number of blocks.
        :type block_count: int
        :param beta_start_value: Initial value for :math:`\\beta`, defaults to 0.1
        :type beta_start_value: float, optional        
        :raises RuntimeWarning: The solution for :math:`\\beta'` is found numerically. In case of convergence problems, an exception is raised.
        :return: Tuple (:math:`\\beta'`, cutoff-count).
        :rtype: tuple[float, int]
        """
        discrete_cutoff = binom.ppf(1.0 - alpha_homogeneity_err_control_requested, n=block_count, p=beta_start_value)
        result = root_scalar(lambda beta: binom.sf(k=discrete_cutoff, n=block_count, p=beta) - alpha_homogeneity_err_control_requested, x0=beta_start_value)
        if not result.converged:
            raise RuntimeWarning(f"Convergence problem in hyperparam-choice for block count={block_count}, using beta={result.root}.")
        return result.root, int(discrete_cutoff)

    def _get_opt_blocksize_base(self, N: int) -> float:
        """Heuristic hyper-parameter choice for block-size :math:`B` based on sample-count :math:`N` for empty conditioning-set.

        :param N: Sample-size.
        :type N: int
        :return: Block-size (further processed by :py:meth:`get_opt_block_size`, and rounded to an integer there).
        :rtype: float
        """
        if self.regimes_are_large:
            return 30
        else:
            return 5.0*np.log10(N)-3
    
    def _sanitize_blocksize(self, B: int, N: int, dim_Z: int) -> int:
        """Ensure a chosen block-size makes sense for use with partial correlation.

        :param B: Targeted block-size.
        :type B: int
        :param N: Sample-size.
        :type N: int
        :param dim_Z: Size of conditioning-set.
        :type dim_Z: int
        :return: Reasonable block-size to use.
        :rtype: int
        """
        if int(N/B) < self.min_block_count:
            B = int(N/self.min_block_count)
        if B <= dim_Z + 3:
            B = self.min_effective_sample_count + dim_Z + 3
        return B

    def get_opt_blocksize(self, N: int, dim_Z: int) -> int:
        """Get heuristic hyper-parameter choice for block-size :math:`B`.

        :param N: Sample-size.
        :type N: int
        :param dim_Z: Conditioning-set size.
        :type dim_Z: int
        :return: Recommended block-size to use.
        :rtype: int
        """
        base_value = self._get_opt_blocksize_base(N)
        result = int(base_value + 1.5 * dim_Z)
        return self._sanitize_blocksize(result, N, dim_Z)
    
    def _validate(self, k, n, beta):
        if binom.sf(k=k, n=n, p=beta) - self.alpha > self.tolerance:
            raise ValueError(f"Hyperparameter-configuration failed to self-validate at tolerance={self.tolerance}")

    def compute_for_fixed_sample_count(self, N: int, dim_Z: int) -> data_processing.IProvideHyperparamsForBinomial.Hyperparams:
        """Obtain actual runtime parameters. Typically called only by implementation of the corresponding marked-independence stage.

        :param N: Sample-size.
        :type N: int
        :param dim_Z: Conditioning-set size.
        :type dim_Z: int
        :return: Execution-parameters for the corresponding algorithm.
        :rtype: dpl.IProvideHyperparamsForBinomial.Hyperparams
        """
        B = self.get_opt_blocksize(N, dim_Z)
        block_count = int(N/B)
        beta, k0 = self.get_opt_beta(alpha_homogeneity_err_control_requested=self.alpha, block_count=block_count, beta_start_value=0.1)
        if self.validate_choice : self._validate(k0, block_count, beta)
        return data_processing.IProvideHyperparamsForBinomial.Hyperparams( B=B, alpha=self.alpha, beta=beta, max_acceptable_count=k0 )


class Hyperparams_WeakInterval_ParCorr(data_processing.IProvideHyperparamsForAcceptanceInterval):
    """Helper to configure standard hyper-parameter sets for acceptance-interval weak-regime test when using partial correlation."""
    
    def __init__(self, alpha=0.05, regimes_are_large=True):
        """Configure hyper-parameters set based on given parameters.

        :param alpha: Error-control target :math:`\\alpha`, defaults to 0.05
        :type alpha: float, optional
        :param regimes_are_large: Consider hyper-parameters for large regimes, defaults to True
        :type regimes_are_large: bool, optional
        """
        self.alpha = alpha
        self.regimes_are_large = regimes_are_large


    def hyperparams_for_acceptance_interval(self, N: int, dim_Z: int) -> data_processing.IProvideHyperparamsForAcceptanceInterval.Hyperparams:
        return self.get_for_fixed_sample_count(N=N, dim_Z=dim_Z)
    

    def get_opt_blocksize(self, N: int, dim_Z: int) -> int:
        """Heuristic choice of hyper-parameter for block-size :math:`B`.

        :param N: Sample-size.
        :type N: int
        :param dim_Z: Conditioning-set size.
        :type dim_Z: int
        :return: Heuristic block-size choice.
        :rtype: int
        """
        if self.regimes_are_large:
            return int(round(31 + dim_Z * (59-31) / 20))
        else:
            return int(round(11 + dim_Z * (43-11) / 20))
            
    def get_opt_cutoff(self, N: int, dim_Z: int) -> float:
        """Heuristic choice of hyper-parameter for cutoff :math:`c`.

        :param N: Sample-size.
        :type N: int
        :param dim_Z: Conditioning-set size.
        :type dim_Z: int
        :return: Heuristic cutoff choice.
        :rtype: float
        """
        if self.regimes_are_large:
            return 0.2
        else:
            return 0.275 + dim_Z * (0.25-0.275) / 20

    def get_for_fixed_sample_count(self, N: int, dim_Z: int) -> data_processing.IProvideHyperparamsForAcceptanceInterval.Hyperparams:
        """Obtain actual runtime parameters. Typically called only by implementation of the corresponding marked-independence stage.

        :param N: Sample-size.
        :type N: int
        :param dim_Z: Size of the conditioning set.
        :type dim_Z: int
        :return: Execution-parameters for the corresponding algorithm.
        :rtype: dpl.IProvideHyperparamsForAcceptanceInterval.Hyperparams
        """
        return data_processing.IProvideHyperparamsForAcceptanceInterval.Hyperparams(
            B=int(self.get_opt_blocksize(N=N, dim_Z=dim_Z)),
            alpha=self.alpha,
            cutoff=self.get_opt_cutoff(N=N, dim_Z=dim_Z)
        )

