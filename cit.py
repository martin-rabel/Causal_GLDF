import numpy as np
from scipy.stats import norm
from numpy.linalg import inv as matrix_inv
from .data_management import CIT_DataPatterned, BlockView
from .data_processing import ITestCI, IProvideAnalyticQuantilesForCIT, IProvideVarianceForCIT
from typing import Literal

class ParCorr(ITestCI, IProvideAnalyticQuantilesForCIT, IProvideVarianceForCIT):
    """Implementation of partial correlation independence test and interface for use with mCIT.
        Can run on many blocks at once efficiently.
    """

    def __init__(self, alpha: float=0.05, lower_bound_clip_value: float=0.3, force_regression_global: bool=False,
                 analytic_approximation_for_cutoff: Literal["by effective count", "by large N expansion"]="by effective count"):
        """Constructor for partial correlation CIT.

        :param alpha: target for FPR-control, defaults to 0.05
        :type alpha: float, optional
        :param lower_bound_clip_value: to avoid numeric instability for high dependencies, the implementation
            provided for :py:meth:`IProvideAnalyticQuantilesForCIT.cit_quantile_estimate<GLDF.data_processing.IProvideAnalyticQuantilesForCIT.cit_quantile_estimate>`
            clips bounds to a predefined range
            (this is consistent, and does not cost substantial power), defaults to 0.3
        :type lower_bound_clip_value: float, optional
        :param force_regression_global: By default (disabled) regressions are computed locally per block, while in principle
            slightly less sample-efficient on IID data, this is more robust against non-stationarities, defaults to False
        :type force_regression_global: bool, optional        
        :param analytic_approximation_for_cutoff: analytic approximation used to implement
            :py:class:`IProvideAnalyticQuantilesForCIT<GLDF.data_processing.IProvideAnalyticQuantilesForCIT>`,
            defaults to "by effective count"
        :type analytic_approximation_for_cutoff: Literal["by effective count", "by large N expansion"]
        """
        self.alpha = alpha
        self.lower_bound_clip_value = lower_bound_clip_value
        self.force_regression_global = force_regression_global
        self.analytic_approximation_for_cutoff = analytic_approximation_for_cutoff

    def run_single(self, data: CIT_DataPatterned) -> ITestCI.Result:
        global_score = self.score_single(data)
        pvalue = self.pvalue(score=global_score, N=data.sample_count(), dim_Z=data.z_dim())
        return ITestCI.Result(
            global_score=global_score,
            dependent=self.is_pvalue_dependent(pvalue)
        )
    
    def run_many(self, data: BlockView) -> ITestCI.Result:
        block_scores = self.score_many(data)
        global_score = np.mean(block_scores)
        pvalue = self.pvalue_of_mean(score_mean=global_score, block_size=data.block_size(), block_count=data.block_count(), dim_Z=data.z_dim())
        return ITestCI.Result(
            global_score=np.mean(block_scores),
            block_scores=block_scores,
            dependent=self.is_pvalue_dependent(pvalue)
        )
    
    def cit_quantile_estimate(self, data: BlockView, cit_result: ITestCI.Result, beta: float, cit_obj: ITestCI) -> float:
        assert type(cit_obj) == type(self)
        d1 = cit_result.global_score
        d1_abs = abs(d1)
        d1_is_positive = (d1 > 0.0)

        cutoff_abs = self._lower_bound_from_pvalue(
            d1_abs, beta, data.block_size(), conditioning_set_size=data.z_dim(),
            how=self.analytic_approximation_for_cutoff, N_global=data.sample_count_used()
        )

        return cutoff_abs if d1_is_positive else -cutoff_abs
    
    def get_variance_estimate(self, N: int, dim_Z: int, cit_obj: ITestCI) -> float:
        assert type(cit_obj) == type(self)
        return self.analytic_score_var(n=N, z_dim=dim_Z)
    

    @staticmethod
    def effective_sample_count(n: int, z_dim: int) -> int:
        """Compute effective sample-size

        :param n: actual sample-size
        :type n: int
        :param z_dim: size of conditioning set
        :type z_dim: int
        :return: effective sample-size
        :rtype: int
        """
        return n - 3 - z_dim
    
    @staticmethod
    def _n_required_for_eff_sample_count(effective_sample_count: int, z_dim:int) -> int:
        return effective_sample_count + z_dim + 3
    @classmethod
    def _analytic_score_var_at_effective_sample_size(cls, effective_size: int) -> float:
        return 1.0 / effective_size
    @classmethod
    def analytic_score_var(cls, n: int, z_dim: int) -> float:
        """Analytic approximation for score variance.

        :param n: sample size
        :type n: int
        :param z_dim: size of conditioning set
        :type z_dim: int
        :return: score variance
        :rtype: float
        """
        return 1.0 / cls.effective_sample_count(n, z_dim)
    @classmethod
    def analytic_score_std(cls, n: int, z_dim: int) -> float:
        """Analytic approximation for score standard deviation.

        :param n: sample size
        :type n: int
        :param z_dim: size of conditioning set
        :type z_dim: int
        :return: score standard dceviation
        :rtype: float
        """
        return np.sqrt(cls.analytic_score_var(n, z_dim))


    @staticmethod
    def _score_z_pair(x_blocks: np.ndarray, y_blocks: np.ndarray, var_ddof: int=1) -> np.ndarray:
        mean_x = np.mean( x_blocks, axis=1 ).reshape(-1,1)
        mean_y = np.mean( y_blocks, axis=1 ).reshape(-1,1)
        covars = np.mean( (x_blocks-mean_x) * (y_blocks-mean_y), axis=1 )
        var_x = np.var(x_blocks, ddof=var_ddof, axis=1) + 0.001 # add small value to avoid instability
        var_y = np.var(y_blocks, ddof=var_ddof, axis=1) + 0.001
        corr = np.clip( covars/np.sqrt(var_x*var_y), -0.999, 0.999 ) # clip to avoid instability
        z = np.arctanh(corr) # np.atanh(corr) only in np>2.0?
        return z
    
    @staticmethod
    def _regression_coefficients_many(source_blocks_mean_0: np.ndarray, target_blocks_mean_0: np.ndarray) -> np.ndarray:
        # somehow numpys lstsq does not parallelize well, use np.linalg.inv directly instead 
        block_count, block_size, Z_dim = source_blocks_mean_0.shape
        assert block_size > Z_dim, "Cannot invert the matrix for regession with Z_dim >= block_size."
        X = source_blocks_mean_0
        X_transpose = np.transpose(source_blocks_mean_0, [0,2,1])
        X_t_X = np.matmul(X_transpose, X)
        X_t_X_inv = matrix_inv(X_t_X)
        X_t_X_inv_X_t = np.matmul(X_t_X_inv, X_transpose)
        coeffs = np.matmul(X_t_X_inv_X_t, target_blocks_mean_0.reshape(block_count,block_size,1))
        return coeffs.reshape(block_count,Z_dim)
    
    @classmethod
    def _regress_out_raw(cls, x_blocks_mean_0: np.ndarray, y_blocks_mean_0: np.ndarray, z_blocks_mean_0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        block_count, block_size, var_count = z_blocks_mean_0.shape

        coeffs_zx = cls._regression_coefficients_many(z_blocks_mean_0, x_blocks_mean_0)
        coeffs_zy = cls._regression_coefficients_many(z_blocks_mean_0, y_blocks_mean_0)
        
        residuals_x = x_blocks_mean_0 - np.sum(coeffs_zx.reshape([block_count,1,var_count]) * z_blocks_mean_0, axis=2)
        residuals_y = y_blocks_mean_0 - np.sum(coeffs_zy.reshape([block_count,1,var_count]) * z_blocks_mean_0, axis=2)
        
        return residuals_x, residuals_y

    def _regress_out(self, data: BlockView) -> tuple[np.ndarray, np.ndarray]:
        if self.force_regression_global:
            data_centered = data.trivialize().copy_and_center()
            residuals = self._regress_out_raw(data_centered.x_blocks, data_centered.y_blocks, data_centered.z_blocks)
            residuals = data.apply_blockformat(*residuals)
            return residuals.x_blocks, residuals.y_blocks
        else:
            data_centered = data.copy_and_center()
            return self._regress_out_raw(data_centered.x_blocks, data_centered.y_blocks, data_centered.z_blocks)

    def score_many(self, data: BlockView) -> np.ndarray:
        """Compute score (z-transformed partial correlation) on blocks

        :param data: data blocks
        :type data: BlockView
        :return: score per block
        :rtype: np.ndarray
        """
        if data.z_dim() > 0:
            return self._score_z_pair(*self._regress_out(data))
        else:
            return self._score_z_pair(data.x_blocks, data.y_blocks)

    def score_single(self, data: CIT_DataPatterned) -> float:  
        """Compute score (z-transformed partial correlation)

        :param data: data
        :type data: CIT_DataPatterned
        :return: score
        :rtype: float
        """
        return float(self.score_many(data.view_blocks_trivial())[0])
    

    def _pvalue(self, score: float|np.ndarray, sigma: float) -> float|np.ndarray:        
        return 2.0 * (1.0 - norm.cdf( np.abs(score), scale=sigma ))

    def pvalue(self, score: float|np.ndarray, N: int, dim_Z: int) -> float|np.ndarray:
        """Compute p-value for a given score and setup (possibly per block).

        :param score: z-value(s)
        :type score: float | ndarray
        :param N: sample-size
        :type N: int
        :param dim_Z: size of conditioning set
        :type dim_Z: int
        :return: p-value(s)
        :rtype: float | ndarray
        """
        return self._pvalue(score, sigma=self.analytic_score_std(N, dim_Z))
    
    def pvalue_of_mean(self, score_mean: float, block_size: int, block_count: int, dim_Z: int) -> float:
        """Compute p-value for a given score-mean over blocks and setup.

        :param score_mean: mean z-value
        :type score_mean: float
        :param block_size: block-size
        :type block_size: int
        :param block_count: block-count
        :type block_count: int
        :param dim_Z: size of conditioning set
        :type dim_Z: int
        :return: p-value
        :rtype: float
        """
        if self.force_regression_global:
            v_block = self.analytic_score_var(block_size, z_dim=0)
            n = block_count * block_size
            n_eff = self.effective_sample_count(n=n, z_dim=dim_Z)
            v_global = (v_block / block_count) * (n/n_eff)
        else:                
            v_block = self.analytic_score_var(block_size, dim_Z)
            v_global = v_block / block_count
            return self._pvalue(score_mean, sigma=np.sqrt(v_global))
    
    def is_pvalue_dependent(self, pvalue: float) -> bool:
        """Decide if a given p-value should be considered evidence for a dependent test.

        :param pvalue: p-value
        :type pvalue: float
        :return: test considered dependent
        :rtype: bool
        """
        return pvalue < self.alpha



    def _lower_bound_from_pvalue(self, reference: float, pvalue: float, count: int, conditioning_set_size: int, how: Literal["by effective count", "by large N expansion"], N_global: int=None) -> float:
        if how == "by effective count":
            return self._lower_bound_from_pvalue_by_effective_count(reference, pvalue, count, conditioning_set_size, N_global=N_global)
        elif how == "by large N expansion":
            return self._lower_bound_from_pvalue_by_large_N_expansion(reference, pvalue, count, conditioning_set_size, N_global=N_global)
        else:
            assert False, "unknown analytical approximation"

    def _lower_bound_from_pvalue_by_effective_count(self, reference: float, pvalue: float, count: int, conditioning_set_size: int, N_global: int) -> float:
        # z is var-stabilized, so should not depend on reference
        assert reference >= 0.0, "remove sign first"
        if self.force_regression_global:
            # Heuristically account for |Z| samples "lost" globally by the fraction of samples (count/N_global) used here.
            eff_n = count-3-conditioning_set_size*(count/N_global)
        else:
            eff_n = self.effective_sample_count(count, conditioning_set_size)
        distance = norm.ppf(1.0 - pvalue) / np.sqrt(eff_n)
        return self._clip_lower_bound( reference - distance )
    
    def _lower_bound_from_pvalue_by_large_N_expansion(self, reference: float, pvalue: float, count: int, conditioning_set_size: int, N_global: int) -> float:
        assert reference >= 0.0, "remove sign first"
        corr = np.tanh(reference)
        if self.force_regression_global:
            # Heuristically account for |Z| samples "lost" globally by the fraction of samples (count/N_global) used here.
            N = count-conditioning_set_size*(count/N_global)
        else:
            N = count-conditioning_set_size
        v = 1 / N + (6.0 - corr*corr)/(2 * N * N) # leading terms in 1/N expansion
        distance = norm.ppf(1.0 - pvalue) * np.sqrt(v)
        return self._clip_lower_bound( reference - distance )
    
    def _clip_lower_bound(self, lower_bound_raw: float) -> float:
        # Avoid instability for large dependence-values (does not seem to affect relevant power vs. true-regimes).
        return self.lower_bound_clip_value if lower_bound_raw > self.lower_bound_clip_value else lower_bound_raw