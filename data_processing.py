import numpy as np
from scipy.stats import binom, norm
from .data_management import CIT_DataPatterned, BlockView
from dataclasses import dataclass
from typing import Literal, Callable
from warnings import warn

class ITestCI:
    """
        Interface specifying how to expose (custom) CIT-implementations. Supports the efficient dispatch of
        large collections of blocks.

        .. seealso::
            For use with the current mCIT implementation, also the interfaces for
            :py:class:`IProvideVarianceForCIT` and potentially
            :py:class:`IProvideAnalyticQuantilesForCIT` should be exposed.
            For details, see also :ref:`label-interfaces-custom-data-proc`.
    """

    @dataclass
    class Result:
        """
            Output format for (custom) CIT-implementations.
        """

        global_score : float    #: Score for the entire dataset.
        dependent : bool        #: Dependency on the entire dataset.
        block_scores : np.ndarray|None = None   #: Score for each block individually (if applicable).

    def run_single(self, data: CIT_DataPatterned) -> Result:
        """Run a single CIT on all data.

        :param data: The data-set to use.
        :type data: CIT_DataPatterned
        :return: Structured test-output.
        :rtype: ITestCI.Result
        """
        raise NotImplementedError()
    
    def run_many(self, data: BlockView) -> Result:
        """Run CITs on many blocks (efficiently).

        :param data: The data-set to use.
        :type data: BlockView
        :return: Structured test-output.
        :rtype: ITestCI.Result
        """
        raise NotImplementedError()
    
    @staticmethod
    def _extract_cache_id(fname: str, data: CIT_DataPatterned|BlockView) -> tuple:
        """Extract a cache-id from a given query. A fallback returning :py:obj:`data.cache_id`
        is provided; it is possible, but not typically necessary to overwrite this fallback.

        .. seealso::
            For a discussion of cache-IDs, see :ref:`label-cache-ids`.

        :param fname: name of the method to cache
        :type fname: str
        :param data: data containing the cache-id
        :type data: CIT_DataPatterned | BlockView
        :return: extracted cache-id, here simply :py:obj:`data.cache_id`.
        :rtype: tuple
        """
        return data.cache_id


class ITestHomogeneity:
    """
        Interface specifying how to expose (custom) implementations of homogeneity-tests.
    """

    def is_homogeneous(self, data: CIT_DataPatterned) -> bool:
        """Test if the data supplied by the query is homogenous.

        :param data: The data to inspect.
        :type data: CIT_DataPatterned
        :return: The truth-value indicating if the data-set was accepted as homogenous.
        :rtype: bool
        """
        raise NotImplementedError()
    
class ITestWeakRegime:
    """
        Interface specifying how to expose (custom) implementations of weak-regime tests.
    """

    def has_true_regime(self, data: CIT_DataPatterned) -> bool:
        """Test if the non-homogenous data supplied by the query contains a true or just a weak regime.

        :param data: The non-homogenous data to inspect.
        :type data: CIT_DataPatterned
        :return: The truth-value indicating if the data-set is beliefed to feature a true regime.
        :rtype: bool
        """
        raise NotImplementedError()



class ITestMarkedCI:
    """
        Interface specifying how to expose (custom) implementations of marked conditional independence tests.
    """

    type _three_way_result_t = Literal["dependent", "independent", "regime", "weak or regime"] #: encode internal result/category

    class Result:
        """
            Structured output of mCIT.
        """
        def __init__(self, result_string:'ITestMarkedCI._three_way_result_t'):
            """Initialize from completed query's result.

            :param result_string: The concluded categorization of data.
            :type result_string: Literal["dependent", "independent", "regime", "weak or regime"]
            """
            self.result_string : 'mCIT._three_way_result_t' = result_string # can be dependent, independent, regime or "weak or regime" (in pc1-phase)

        def is_regime(self) -> bool:
            """Inspect the result to check if a true regime was found.

            :return: Truth-value about the presence of a true-regime.
            :rtype: bool
            """
            return self.result_string == "regime"
        
        def is_globally_dependent(self) -> bool:
            """Inspect the result to check if any sort of dependence (global, weak-regime or true-regime) was found.

            :return: Truth-value about the presence of dependence.
            :rtype: bool
            """
            return self.result_string != "independent"
        
        def is_globally_independent(self) -> bool:
            """Inspect the result to check if no dependence was found.

            :return: Truth-value about global independence.
            :rtype: bool
            """
            return self.result_string == "independent"
    
    def marked_independence(self, data: CIT_DataPatterned) -> Result:
        """Test marked conditional independence on the supplied data.

        :param data: The data associated to the test to perform.
        :type data: CIT_DataPatterned
        :return: The structured mCIT output.
        :rtype: ITestMarkedCI.Result
        """
        raise NotImplementedError()
    
class ITestIndicatorImplications:
    """
        Interface specifying how to expose (custom) implementations of indicator-relation tests.
    """

    def is_implied_regime(self, A_list: list[CIT_DataPatterned], B: CIT_DataPatterned) -> bool:
        """Test the indicator implication all test of lhs list are independent :math:`\\Rightarrow` the test on
        the rhs is independent.

        :param A_list: the lhs list of tests
        :type A_list: list[CIT_DataPatterned]
        :param B: the rhs test
        :type B: CIT_DataPatterned
        :return: The truth-value of the given implication.
        :rtype: bool
        """
        raise NotImplementedError()

class IProvideHyperparamsForRobustCIT:
    """
        Interface specifying how to supply (customized) hyper-parameters for robust dependence testing.
    """

    @dataclass
    class Hyperparams:
        B : int         #: block-size

    def hyperparams_for_robust_cit(self, N: int, dim_Z: int) -> 'IProvideHyperparamsForRobustCIT.Hyperparams':
        """Supply hyper-parameters for robust CI testing for the given setup.

        :param N: sample-size
        :type N: int
        :param dim_Z: size of the conditioning set
        :type dim_Z: int
        :return: Hyper-parameters to use.
        :rtype: IProvideHyperparamsForRobustCIT.Hyperparams
        """
        raise NotImplementedError()
    
    @staticmethod
    def from_binomial_homogeneity_test(homogeneity_test_hyperparams: 'IProvideHyperparamsForBinomial') -> 'IProvideHyperparamsForRobustCIT':
        """It is runtime-efficient and simple to obtain hyper-parameters for robust CI testing compatible to
        those used by homogeneity-testing.

        :param homogeneity_test: homogeneity-test hyper-parameters to copy
        :type homogeneity_test: IProvideHyperparamsForBinomial
        :return: Hyper-parameter Provider.
        :rtype: IProvideHyperparamsForRobustCIT
        """
        class _HyperparamsRobustCIT(IProvideHyperparamsForRobustCIT):
            def __init__(self, homog_hyper: IProvideHyperparamsForBinomial):
                self.homog_hyper = homog_hyper
            def hyperparams_for_robust_cit(self, N: int, dim_Z: int) -> 'IProvideHyperparamsForRobustCIT.Hyperparams':
                return IProvideHyperparamsForRobustCIT.Hyperparams(B=self.homog_hyper.hyperparams_for_binomial(N, dim_Z).B)
        return _HyperparamsRobustCIT(homogeneity_test_hyperparams)


class mCIT(ITestMarkedCI):
    """
        (Regime-)marked independence test (mCIT).
    """

    _internal_result_t = Literal["dependent", "independent", "weak", "regime", "weak or regime"]


    def __init__(self, cit: ITestCI, homogeneity_test: ITestHomogeneity|None, weak_test: ITestWeakRegime|None=None,
                 homogeneity_first: bool=True, robust_conditional_testing: IProvideHyperparamsForRobustCIT|bool|None=True):
        """Constructor of mCIT from underlying tests.

        :param ci_test: Underlying CI-test.
        :type ci_test: ITestCI
        :param homogeneity_test: Underyling homogeneity-test.
        :type homogeneity_test: ITestHomogeneity, optional
        :param weak_test: Underlying weak-regime test, defaults to None
        :type weak_test: ITestWeakRegime, optional
        :param homogeneity_first: Test homogeneity first, then global dependency, defaults to True (recommended)
        :type homogeneity_first: bool, optional
        :param robust_conditional_testing: Configuration of conditinonal test (eg of regressors) to rely on data only locally in the pattern
            (recommended for simple parametric tests), defaults to True (which will use
            :py:meth:`IProvideHyperparamsForRobustCIT.from_binomial_homogeneity_test`)
        :type robust_conditional_testing: IProvideHyperparamsForRobustCIT|bool, optional
        :type min_regime_fraction: float, optional
        """
        self.ci_test = cit
        self.homogeneity_test = homogeneity_test
        self.weak_test = weak_test
        self.homogeneity_first = homogeneity_first
        if isinstance(robust_conditional_testing, bool):
            self.robust_conditional_testing = IProvideHyperparamsForRobustCIT.from_binomial_homogeneity_test(homogeneity_test.hyperparams) if robust_conditional_testing else None
        else:
            self.robust_conditional_testing = robust_conditional_testing

        if homogeneity_first:
            assert homogeneity_test is not None
            self.run = self._run_inhom_first
        else:
            self.run = self._run_global_first

    def marked_independence(self, data: CIT_DataPatterned) -> ITestMarkedCI.Result:
        if self.homogeneity_first:
            internal_result = self._run_inhom_first(data)
        else:
            internal_result = self._run_global_first(data)

        three_way_result = self._marked_independence_from_category(internal_result)
        return ITestMarkedCI.Result(three_way_result)


    def _is_globally_dependent(self, data: CIT_DataPatterned) -> bool:
        if self.robust_conditional_testing is not None:
            # by default use same blocks as homogeneity, can use a different hyper-parameter provider
            robust_params = self.robust_conditional_testing.hyperparams_for_robust_cit(data.sample_count(), data.z_dim())
            cit_result = self.ci_test.run_many(data.view_blocks(robust_params.B))
            return cit_result.dependent
        else:
            return self.ci_test.run_single(data).dependent

    def _weak_or_regime(self, data: CIT_DataPatterned) -> 'mCIT._internal_result_t':
        if self.weak_test is None:
            return "weak or regime"
        else:
            if self.weak_test.has_true_regime(data):
                return "regime"
            else:
                return "weak"
            
    def _marked_independence_from_category(self, internal_result: 'mCIT._internal_result_t') -> 'ITestMarkedCI._three_way_result_t':
        # merge 'weak' and 'dependent' into the single output 'dependent'
        return "dependent" if (internal_result == "weak") else internal_result

    def _run_global_first(self, data: CIT_DataPatterned) -> 'mCIT._internal_result_t':
        if self._is_globally_dependent(data):
            if self.homogeneity_test is not None:
                if self.homogeneity_test.is_homogeneous(data):
                    return "dependent"
                else:
                    return self._weak_or_regime(data)
            else:
                return "dependent"
        else:
            return "independent"
    
    def _run_inhom_first(self, data: CIT_DataPatterned) -> 'mCIT._internal_result_t':
        if self.homogeneity_test.is_homogeneous(data):
            if self._is_globally_dependent(data):
                return "dependent"
            else:                
                return "independent"
        else:
            return self._weak_or_regime(data)



class IProvideHyperparamsForBinomial:
    """
        Interface specifying how to supply (customized) hyper-parameters for binomial homogeneity testing.
    """

    @dataclass
    class Hyperparams:
        B : int         #: block-size
        alpha : float   #: error-control target :math:`\alpha`
        beta : float    #: binomial quantile
        max_acceptable_count : float    #: by numerical precision, pvalue at max acceptable count may be within :math:`\alpha +` tolerance (by default :math:`10^{-5}`)

    def hyperparams_for_binomial(self, N: int, dim_Z: int) -> Hyperparams:
        """Supply hyper-parameters for the binomial homogeneity-test for the given setup.

        :param N: sample-size
        :type N: int
        :param dim_Z: size of conditioning set
        :type dim_Z: int
        :return: The hyper-parameters to use.
        :rtype: IProvideHyperparamsForBinomial.Hyperparams
        """
        raise NotImplementedError()
    
class IProvideAnalyticQuantilesForCIT:
    """
        Interface to expose for (custom) CIT-implementations if the homogeneity-test
        :py:class:`Homogeneity_Binomial` is used. Quantiles can also be bootstrapped,
        if no implementation (:py:data:`None`) of this interface is provided.
    """

    def cit_quantile_estimate(self, data: BlockView, cit_result: ITestCI.Result, beta: float, cit_obj: ITestCI) -> float:
        """Provide an estimate of the :math:`\\beta`-quantile for the test implemented by cit_obj.

        :param data: The data-blocks to operate on.
        :type data: BlockView
        :param cit_result: The CIT result for the data (currently always run previously anyway).
        :type cit_result: ITestCI.Result
        :param beta: The quantile :math:`\\beta` to estimate.
        :type beta: float
        :param cit_obj: The underlying cit-instance for which the quantile should be computed.
            (The present interface :py:class:`IProvideAnalyticQuantilesForCIT` can,
            but does not have to, be exposed on the CIT-type itself.)
        :type cit_obj: ITestCI
        :return: Estimate of the dependence-value at the quantile :math:`\\beta`
        :rtype: float
        """
        raise NotImplementedError()


class Homogeneity_Binomial(ITestHomogeneity):
    """
        Homogeneity test based on binomial approach via quantile estimator.
        Implements :py:class:`ITestHomogeneity` interface for use with :py:class:`mCIT`.
    """

    @staticmethod
    def _get_actual_error_control_raw(alpha_homogeneity_err_control_requested, block_count, alpha_binom) -> float:
        discrete_cutoff = binom.ppf(1.0 - alpha_homogeneity_err_control_requested, n=block_count, p=alpha_binom)
        return binom.sf(discrete_cutoff, n=block_count, p=alpha_binom)
    
    def get_actual_error_control(self, N:int, dim_Z:int=0) -> float:
        """
        Gets the actual error-control after accounting for counting-statistics. Depending on the internals
        of the used hyper-parameter set, this may be different from :math:`\\alpha` as specified originally.

        :param N: Sample size N.
        :type N: int
        :param dim_Z: Size of the conditioning-set Z, defaults to 0
        :type dim_Z: int, optional
        :return: Effective error-control target :math:`\\alpha`.
        :rtype: float
        """
        params = self.hyperparams.hyperparams_for_binomial(N, dim_Z)
        block_count = int(N/params.B)
        return Homogeneity_Binomial._get_actual_error_control_raw(params.alpha, block_count, params.beta)


    def __init__(self, hyperparams: IProvideHyperparamsForBinomial, cit: ITestCI, cit_analytic_quantile_estimate: IProvideAnalyticQuantilesForCIT|None = None,
                 bootstrap_block_count: int=5000, next_bootstrap_seed: Callable[[], None|int|np.random.SeedSequence]= lambda : None):
        """Construct from hyper-parameter set, and either cit-specific quantile estimate or bootstrap block-count for generic quantile estimation.

        :param hyperparams: Hyper-parameter set to use.
        :type hyperparams: IProvideHyperparamsForBinomial
        :param cit_analytic_quantile_estimate: Cit-specific quantile estimate (if available), defaults to None
        :type cit_analytic_quantile_estimate: IProvideAnalyticQuantilesForCIT | None, optional
        :param bootstrap_block_count: Block-count for bootstrap of quantile (if no cit-specific quantile was provided), defaults to 5000
        :type bootstrap_block_count: int, optional
        """
        self.hyperparams = hyperparams
        self.cit = cit
        self.analytic_quantile = cit_analytic_quantile_estimate
        self.bootstrap_block_count=bootstrap_block_count
        self.next_bootstrap_seed = next_bootstrap_seed


    def get_quantile(self, data: BlockView, cit_result: ITestCI.Result, beta: float) -> float:
        """Obtain a quantile for the given dataset.

        :param data: Data-blocks associated to current test.
        :type data: BlockView
        :param cit_result: The CIT result for the data (currently always run previously anyway).
        :type cit_result: ITestCI.Result
        :param beta: Target probabilty to get a quantile (lower bound) for.
        :type beta: float
        :return: The estimated quantile lower bound.
        :rtype: float
        """
        if self.analytic_quantile is not None:
            return self.analytic_quantile.cit_quantile_estimate(data, cit_result, beta, cit_obj=self.cit)
        else:
            return self._bootstrap_quantile(data, cit_result, beta)
            
    def _bootstrap_quantile(self, data: BlockView, cit_result: ITestCI.Result, beta: float) -> float:
        d1_positive = (cit_result.global_score > 0.0)
        
        rng = np.random.default_rng(self.next_bootstrap_seed())
        bootstrap_blocks = data.bootstrap_unaligned_blocks(rng, bootstrap_block_count=self.bootstrap_block_count)          

        z_scores_unaligned = self.cit.run_many(bootstrap_blocks).block_scores

        target_quantile = beta if d1_positive else 1.0-beta
        return float(np.quantile(z_scores_unaligned, target_quantile))

    def is_homogeneous(self, data: CIT_DataPatterned) -> bool:
        params = self.hyperparams.hyperparams_for_binomial(data.sample_count(), data.z_dim())
        data_blocks = data.view_blocks(params.B)
        
        cit_result = self.cit.run_many(data_blocks)

        d1 = cit_result.global_score
        d1_is_positive = (d1 > 0.0)

        # get_cutoff is provided by analytic or bootstrap (below)
        cutoff = self.get_quantile(data_blocks, cit_result, params.beta)

        if d1_is_positive:
            binom_count = np.count_nonzero(cit_result.block_scores < cutoff)
        else:
            binom_count = np.count_nonzero(cit_result.block_scores > cutoff)

        # by numerical precision, pvalue at max acceptable count may be within alpha + tolerance (by default 10^-5)
        return binom_count <= params.max_acceptable_count


class IProvideHyperparamsForAcceptanceInterval:
    """
        Interface specifying how to supply (customized) hyper-parameters for acceptance-interval testing.
    """

    @dataclass
    class Hyperparams:
        B : int         #: block-size
        alpha : float   #: error-control target
        cutoff : float  #: cutoff

    def hyperparams_for_acceptance_interval(self, N: int, dim_Z: int) -> Hyperparams:
        """Supply hyper-parameters for acceptance-interval tests for given setup.

        :param N: sample-size
        :type N: int
        :param dim_Z: size of conditioning set
        :type dim_Z: int
        :return: The hyper-parameters to use.
        :rtype: IProvideHyperparamsForAcceptanceInterval.Hyperparams
        """
        raise NotImplementedError()
    
class IProvideVarianceForCIT:
    """
        Interface to expose for (custom) CIT-implementations if one of the acceptance-interval tests
        (:py:class:`IndicatorImplication_AcceptanceInterval` or :py:class:`IndicatorImplication_AcceptanceInterval`) is used.
        These tests require variance-estimates for the block-wise dependence-scores provided by the CIT.
    """

    def get_variance_estimate(self, N: int, dim_Z: int, cit_obj: ITestCI) -> float:
        """Get an estimate of the block-wise variance of the dependence score implemented by cit_obj.

        :param N: The sample count N.
        :type N: int
        :param dim_Z: The dimension of (number of variables in) the condition conditioning set Z.
        :type dim_Z: int
        :param cit_obj: The underlying cit-instance for which the variance should be computed.
            (The present interface :py:class:`IProvideVarianceForCIT` can,
            but does not have to, be exposed on the CIT-type itself.) 
        :type cit_obj: ITestCI
        :return: The estimated value of the variance.
        :rtype: float
        """
        raise NotImplementedError()
    
    def get_std_estimate(self, N: int, dim_Z: int, cit_obj: ITestCI) -> float:
        """Get an estimate of the block-wise standard-deviation of the dependence score implemented by cit_obj.
        Implementation is optional, if this method is not overridden, the square-root of the variance is used. 

        :param N: The sample count N.
        :type N: int
        :param dim_Z: The dimension of (number of variables in) the condition conditioning set Z.
        :type dim_Z: int
        :param cit_obj: The underlying cit-instance for which the standard-deviation should be computed.
            (The present interface :py:class:`IProvideVarianceForCIT` can,
            but does not have to, be exposed on the CIT-type itself.) 
        :type cit_obj: ITestCI
        :return: The estimated value of the standard-deviation.
        :rtype: float
        """
        return np.sqrt( self.get_variance_estimate(N, dim_Z, cit_obj) )
    

class TruncatedNormal:
    """
        Namescope for collection of helpers providing different useful properties of truncated
        normal distributions. Used by :py:class:`WeakRegime_AcceptanceInterval`
        and :py:class:`IndicatorImplication_AcceptanceInterval`.
    """

    @staticmethod
    def mills_ratio(beta: float) -> float:
        """Compute the mills-ratio for :math:`\\beta`.

        :param beta: argument :math:`\\beta`
        :type beta: float
        :return: mills-ratio
        :rtype: float
        """
        return norm.sf(beta) / norm.pdf(beta)

    # For approximations see eg A. Gasull, F. Utzet: "Approximating Mills ratio"
    @staticmethod
    def approx_mills_lower_bound(beta: float) -> float:
        """Lower bound for mills-ratio.

        :param beta: argument :math:`\\beta > 0`
        :type beta: float
        :return: lower bound for mills-ratio
        :rtype: float
        """
        assert beta > 0.0
        return np.pi / (np.sqrt(beta*beta + 2 * np.pi) + (np.pi - 1) * beta)
    @staticmethod
    def approx_mills_upper_bound(beta):
        """Upper bound for mills-ratio.

        :param beta: argument :math:`\\beta > 0`
        :type beta: float
        :return: upper bound for mills-ratio
        :rtype: float
        """
        assert beta > 0.0
        return np.pi / (np.sqrt((np.pi - 2.0)**2 *beta*beta + 2 * np.pi) + 2 * beta)

    # "inverse" in this context traditionally means "reciprocal" (not an inverse function)
    @classmethod
    def inv_mills_ratio(cls, beta: float) -> float:
        """Reciprocal value of the mills-ratio with improved numerical stability.

        :param beta: argument :math:`\\beta > 0`
        :type beta: float
        :return: reciprocal value of mills-ratio
        :rtype: float
        """
        if beta > 5.0:
            if beta > 1e9: # for some reason scipy scalar_root sometimes feeds inf into this function ....
                return 0.0
            else:
                return 1.0 / cls.approx_mills_lower_bound(beta)
        elif beta > -8.0:
            return 1.0 / cls.mills_ratio(beta) # this can be numerically unstable, avoid large absolute betas
        else:
            return 0.0
        
    # compute mean of a truncated normal
    @classmethod
    def mean_cutoff_below(cls, true_mean: float, true_sigma: float, cutoff: float) -> float:
        """Mean-value of a truncated-below normal distribution.

        :param true_mean: the normal-distribution's original mean-value parameter :math:`\\mu`
        :type true_mean: float
        :param true_sigma: the normal-distribution's original standard-deviation parameter :math:`\\sigma`
        :type true_sigma: float
        :param cutoff: the cutoff location :math:`c`
        :type cutoff: float
        :return: :math:`E[X|X\\geq c]`, where :math:`X \\sim \\mathcal{N}(\\mu, \\sigma^2)`.
        :rtype: float
        """
        beta = (cutoff - true_mean) / true_sigma
        return true_mean + true_sigma * cls.inv_mills_ratio(beta)
    
    @classmethod
    def mean_cutoff_above(cls, true_mean: float, true_sigma: float, cutoff: float) -> float:
        """Mean-value of a truncated-above normal distribution.

        :param true_mean: the normal-distribution's original mean-value parameter :math:`\\mu`
        :type true_mean: float
        :param true_sigma: the normal-distribution's original standard-deviation parameter :math:`\\sigma`
        :type true_sigma: float
        :param cutoff: the cutoff location :math:`c`
        :type cutoff: float
        :return: :math:`E[X|X\\leq c]`, where :math:`X \\sim \\mathcal{N}(\\mu, \\sigma^2)`.
        :rtype: float
        """
        return -cls.mean_cutoff_below(-true_mean, true_sigma, -cutoff)

class WeakRegime_AcceptanceInterval(ITestWeakRegime):
    """
        Acceptance-Interval test implementation of the weak-regime test :py:class:`ITestWeakRegime` interface as used by :py:class:`mCIT`.
    """
    
    def __init__(self, hyperparams: IProvideHyperparamsForAcceptanceInterval, cit: ITestCI, cit_variance_estimate: IProvideVarianceForCIT, min_regime_fraction: float=0.15):
        """Construct from hyper-parameter set and cit-specific dependency-score estimator variance.

        :param hyperparams: Hyper-parameter set to use.
        :type hyperparams: IProvideHyperparamsForAcceptanceInterval
        :param cit: Underlying CIT.
        :type cit: ITestCI
        :param cit_variance_estimate: A cit-specific estimate of the dependency-score estimator's variance.
        :type cit_variance_estimate: IProvideVarianceForCIT
        :param min_regime_fraction: Minimum fraction of data-points in a regime to be considered.
        :type min_regime_fraction: float
        """
        self.hyperparams = hyperparams
        self.cit = cit
        self.cit_variance_est = cit_variance_estimate
        self.min_regime_fraction = min_regime_fraction
            

    def has_true_regime(self, data: CIT_DataPatterned) -> bool:
        params = self.hyperparams.hyperparams_for_acceptance_interval(data.sample_count(), data.z_dim())
        data_blocks = data.view_blocks(params.B)

        sigma = self.cit_variance_est.get_std_estimate(N=params.B, dim_Z=data.z_dim(), cit_obj=self.cit)
        cit_result = self.cit.run_many(data_blocks)
        
        d1 = cit_result.global_score
        d1_is_positive = (d1 > 0.0)

        if not d1_is_positive:
            cit_result.block_scores = -cit_result.block_scores  # sign such that higher dependence regime is positive

        marked_as_below_cutoff = cit_result.block_scores < params.cutoff

        data_below_cutoff = cit_result.block_scores[marked_as_below_cutoff]
        approx_count = len(data_below_cutoff)

        if approx_count < self.min_regime_fraction * len(cit_result.block_scores): # approx count and len(data) both count in blocks
            return False
        else:
            tolerance = norm.ppf(1.0-params.alpha, loc=0.0, scale=sigma / np.sqrt(approx_count))
            lower_bound = TruncatedNormal.mean_cutoff_above(true_mean=0.0, true_sigma=sigma, cutoff=params.cutoff)
            m = np.mean(data_below_cutoff)
            return lower_bound - tolerance < m < tolerance




class IndicatorImplication_AcceptanceInterval(ITestIndicatorImplications):
    """
        Indicator Implication test based on acceptance interval.
    """
    def __init__(self, hyperparams: IProvideHyperparamsForAcceptanceInterval, cit: ITestCI, cit_variance_estimate: IProvideVarianceForCIT, min_regime_fraction: float=0.15):
        """Construct from hyper-parameter set and cit-specific dependency-score estimator variance.

        .. seealso::
            This test is based on the :py:class:`WeakRegime_AcceptanceInterval`.

        :param hyperparams: Hyper-parameter set to use.
        :type hyperparams: IProvideHyperparamsForAcceptanceInterval
        :param cit: Underlying CIT.
        :type cit: ITestCI
        :param cit_variance_estimate: A cit-specific estimate of the dependency-score estimator's variance.
        :type cit_variance_estimate: IProvideVarianceForCIT
        :param min_regime_fraction: Minimum fraction of data-points in a regime to be considered.
        :type min_regime_fraction: float
        """
        self.hyperparams = hyperparams
        self.cit = cit
        self.cit_variance_est = cit_variance_estimate
        self.min_regime_fraction = min_regime_fraction
        
    def is_implied_regime(self, A_list: list[CIT_DataPatterned], B: CIT_DataPatterned) -> bool:
        assert len(A_list) > 0

        max_z_dim = max( max([A.z_dim() for A in A_list]), B.z_dim() )
        params = self.hyperparams.hyperparams_for_acceptance_interval(B.sample_count(), max_z_dim)

        sigma = self.cit_variance_est.get_std_estimate(N=params.B, dim_Z=max_z_dim, cit_obj=self.cit)   

        effective_lhs_d_sqr = None
        for A in A_list:
            # recreate weak query locally for uniform block-size (of max dim(Z))
            cit_result = self.cit.run_many(A.view_blocks(params.B))
            d1_is_positive = (cit_result.global_score > 0.0)
            d1_sign = 1.0 if d1_is_positive else -1.0
            d_sqr_contribution = np.square(cit_result.block_scores)
            relative_sign = np.choose( cit_result.block_scores > 0.0, [-d1_sign, d1_sign] )
            eff_d_sqr_contribution = relative_sign * d_sqr_contribution

            if effective_lhs_d_sqr is None:
                # first entry on lhs
                effective_lhs_d_sqr = eff_d_sqr_contribution
            else:
                # on time-series with differnt max lag in different conditioning-sets,
                # the total number of blocks can be off by one ...
                if abs(len(effective_lhs_d_sqr) - len(eff_d_sqr_contribution)) > 1:
                    warn("implied tests should never have block-count off by more than 1???")

                # add squared contributions (to compute 'euclidean' [with opposite sign correction] distance), but trim (for ts case) entry count
                if len(effective_lhs_d_sqr) < len(eff_d_sqr_contribution):
                    # if last block (of this contribution) not in all others, discard
                    effective_lhs_d_sqr += eff_d_sqr_contribution[:len(effective_lhs_d_sqr)]
                elif len(effective_lhs_d_sqr) > len(eff_d_sqr_contribution):
                    # if this contribution has one block less, trim all others (discard from aggregate result)
                    effective_lhs_d_sqr = effective_lhs_d_sqr[:len(eff_d_sqr_contribution)] + eff_d_sqr_contribution
                else:
                    # same length: simply add
                    effective_lhs_d_sqr += eff_d_sqr_contribution

        effective_lhs_d = np.sqrt(np.maximum(effective_lhs_d_sqr, 0.0) / len(A_list)) # max ok if c>0
        below_cutoff = effective_lhs_d < params.cutoff

        # recreate weak query locally for uniform block-size (of max dim(Z)) for B (rhs)
        blocks_B = B.view_blocks(params.B)
        cit_result = self.cit.run_many(blocks_B)
        d1_is_positive = (cit_result.global_score > 0.0)
        block_d = cit_result.block_scores if d1_is_positive else -cit_result.block_scores


        # on time-series with differnt max lag in different conditioning-sets,
        # the total number of blocks can be off by one ...
        if abs(len(block_d) - len(below_cutoff)) > 1:
            warn("implied tests should never have block-count off by more than 1???")
        if len(block_d) < len(below_cutoff):
            data_below_cutoff = block_d[below_cutoff[:len(block_d)]]
        elif len(block_d) > len(below_cutoff):
            data_below_cutoff = (block_d[:len(below_cutoff)])[below_cutoff]
        else:
            data_below_cutoff = block_d[below_cutoff]
        approx_count = len(data_below_cutoff)

        if approx_count < self.min_regime_fraction * blocks_B.block_count():              
            return False
        else:      
            tolerance = norm.ppf(1.0-params.alpha, loc=0.0, scale=sigma / np.sqrt(approx_count))
            lower_bound = TruncatedNormal.mean_cutoff_above(true_mean=0.0, true_sigma=sigma, cutoff=params.cutoff)
            m = np.mean(data_below_cutoff)
            a = np.clip(approx_count / blocks_B.block_count(), a_min=self.min_regime_fraction, a_max=1.0-self.min_regime_fraction)
            d1_est = ( abs(cit_result.global_score) - a * m ) / ( 1 - a )
            upper_bound = d1_est * norm.sf(params.cutoff, loc=0.0, scale=sigma)
            return lower_bound - tolerance < m < tolerance + upper_bound