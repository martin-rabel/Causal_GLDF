from dataclasses import dataclass
import numpy as np
from typing import TypeVar, Generic

# For python >= 3.12, there is a new simplified syntax for generics.
# Unfortunately this new syntax is not backwards-compatible
# (it does not parse to a valid expression for older versions ...).
var_index = TypeVar('var_index') #, default=int) 3.13 # , SupportsRichComparisonT, hashable)

class CI_Identifier(Generic[var_index]):
    """
        A multi-index defining a conditional independence-statement

        .. note::
        
            The index type :py:obj:`var_index` must be comparable (totally ordered) and hashable, eg int or tuple of int,
            type-annotations are given for the type-var var_index; for time-series, see also :py:class:`CI_Identifier_TimeSeries`.
    """
    type var_index = var_index

    def __init__(self, idx_x:var_index, idx_y:var_index, idx_list_z:list[var_index]):
        """Construct representation of CIT for single-variables X, Y and a list of indices for the variables in
        the set of conditios Z. The representation is undirected and for a (logical) set Z:
        Reordering X and Y or changing the order in the list of inices for Z will result in the same
        representation, see also :py:meth:`__hash__` and :py:meth:`__eq__`.

        :param idx_x: Index of variable X
        :type idx_x: var_index
        :param idx_y: Index of variable Y
        :type idx_y: var_index
        :param idx_list_z: List of indices of variables in set of conditions Z.
        :type idx_list_z: list[var_index]
        """
        self.idx_x = min(idx_x, idx_y)
        self.idx_y = max(idx_x, idx_y)
        self.idx_list_z = list([z_idx for z_idx in sorted(idx_list_z)])

    def undirected_link(self)->tuple[var_index,var_index]:
        """Get the associated (undirected) link as a tuple.

        :return: The associated undirected link.
        :rtype: tuple[var_index,var_index]
        """
        return self.idx_x, self.idx_y
    
    def _as_tuple(self)->tuple[var_index, var_index, tuple[var_index, ...]]:
        """Transcode to a tuple-representation. Used for hashing and comparison-operations.

        :return: _description_
        :rtype: tuple
        """
        return self.idx_x, self.idx_y, tuple(self.idx_list_z)
    
    def __hash__(self) -> int:
        """Hash for unordered containers (dict, set).

        :return: A hash value.
        :rtype: int
        """
        return hash( self._as_tuple() )
    
    def __eq__(self, other: 'CI_Identifier') -> bool:
        """Equality compare two CI-identifiers as undirected (X and Y are exchangable) test
        with a set (order does not matter) of condtions.

        :param other: Other CI-Identifier to compare to.
        :type other: CI_Identifier
        :return: Equality
        :rtype: bool
        """
        return self._as_tuple() == other._as_tuple()
    
    def z_dim(self) -> int:
        """Get dimension of (number of variables in) conditioning set Z.

        :return: dim(Z)
        :rtype: int
        """
        return len(self.idx_list_z)
    
    def conditioning_set(self) -> set[var_index]:
        """Get the conditioning set Z (as set).

        :return: Z
        :rtype: set[var_index]
        """
        return set(self.idx_list_z)
    
    
class CI_Identifier_TimeSeries(CI_Identifier[tuple[int,int]]):
    """
        A multi-index defining a conditional independence-statement for timeseries,
        using tigramite's indexing-convention: Individual nodes are indexed by a
        pair (index, -lag).
    """    
    def max_abs_timelag(self)->int:
        """Get the maximum (abolute) time-lag of any variable involved in the test.
        This means, the maximum over -lag for the lags stored in X, Y or any member of Z.

        :return: Maximum absolute time-lag.
        :rtype: int
        """
        max_timelag_xy = max(-self.idx_x[1], -self.idx_y[1])
        if self.z_dim() > 0:
            max_timelag_z = max([-idx_z_lag for _, idx_z_lag in self.idx_list_z])
            return max(max_timelag_xy, max_timelag_z)
        else:            
            return max_timelag_xy






@dataclass
class BlockView:
    """
        View data as pattern-aligned blocks.
    """

    pattern_provider: 'CIT_DataPatterned' #: Pattern-provider used to generate this view. Primarily for internal use in convenience-functions like :py:meth:`match_blocksize`.
    cache_id : object|None #: unique identifier associated to the data by the data-manager, to be used for caching results of tests. None to disable caching (eg for bootstrap)
    x_blocks: np.ndarray #: shape=(n,B) with n the block-count, B the block-size
    y_blocks: np.ndarray #: shape=(n,B) with n the block-count, B the block-size
    z_blocks: np.ndarray #: shape=(n,B,k) with n the block-count, B the block-size, k=dim(Z)

    def copy_and_center(self) -> 'BlockView':
        """Copy and subtract mean. (Used internally by cit to structure
        residuals correctly, no cach-id required.)

        :return: Centered copy
        :rtype: BlockView
        """
        return BlockView(
            self.pattern_provider,
            None,
            self.x_blocks-np.mean(self.x_blocks, axis=1).reshape(-1,1),
            self.y_blocks-np.mean(self.y_blocks, axis=1).reshape(-1,1),
            self.z_blocks-np.mean(self.z_blocks, axis=1).reshape(self.block_count(),1,self.z_dim()) if self.z_blocks is not None else None
        )
    
    def block_size(self) -> int:
        """Get block-size (total number of samples per block).

        :return: block-size
        :rtype: int
        """
        return self.x_blocks.shape[1]
    
    def block_count(self) -> int:
        """Get block-count.

        :return: block-count
        :rtype: int
        """
        return self.x_blocks.shape[0]
    
    def sample_count_used(self) -> int:
        """Get number of used (contained in a block) data-points.

        :return: used sample-size
        :rtype: int
        """
        return self.block_size() * self.block_count()
    
    def z_dim(self) -> int:
        """Get z-dimension (number of variables in the conditioning set Z).

        :return: dim(Z)
        :rtype: int
        """
        if self.z_blocks is None:
            return 0
        else:
            return self.z_blocks.shape[2]
    
        
    def trivialize(self) -> 'BlockView':
        """View data as trivial (a single block of block-size=sampe-size) blocks.

        :return: trivial block-view
        :rtype: BlockView
        """
        return self.pattern_provider.view_blocks_trivial()
    
    def match_blocksize(self, other: 'BlockView') -> 'BlockView':
        """View data as blocks with size matching another block-view.

        :param other: block-view whose block-size should be matched
        :type other: BlockView
        :return: block-view of matching size
        :rtype: BlockView
        """
        return self.pattern_provider.view_blocks_match(other)
    
    def apply_blockformat(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray|None=None) -> 'BlockView':
        """Given data for X, Y, Z, match to current block-size settings. (Used internally by cit to structure
        residuals correctly, no cach-id required.)

        :param X: X
        :type X: np.ndarray
        :param Y: Y
        :type Y: np.ndarray
        :param Z: Z, defaults to None
        :type Z: np.ndarray | None, optional
        :return: block-view of given data matching current size
        :rtype: BlockView
        """
        return self.pattern_provider.clone_from_data(X, Y, Z).view_blocks_match(self)

    
    def bootstrap_unaligned_blocks(self, rng: np.random.Generator, bootstrap_block_count: int) -> 'BlockView':
        """Bootstrap random (unaligned) blocks.

        :param rng: A random number generator.
        :type rng: np.random.Generator (or similar, requires .integers behaving as numpy.random.Generator)
        :param bootstrap_block_count: Number of blocks to bootstrap
        :type bootstrap_block_count: int
        :return: The bootstrapped blocks (actually a copy, not a view, despite being typed as block-"view").
        :rtype: BlockView
        """
        return self.pattern_provider.bootstrap_unaligned_blocks(rng, bootstrap_block_count, self.block_size())

@dataclass
class CIT_Data:
    """
        Data for CIT.

        .. seealso::
            Used through :py:class:`CIT_DataPatterned`.
    """

    x_data: np.ndarray #: Shape specified by data-manager/pattern-provider. See :py:class:`CIT_DataPatterned`.
    y_data: np.ndarray #: Same shape as x_data.
    z_data: np.ndarray #: Shape=(shape_xy,k), where shape_xy is the shape of x_data/y_data and k=dim(Z) is the size of the conditioning set.
    cache_id : tuple|None #: Unique identifier associated to the data by the data-manager, to be used for caching results. None to disable caching.



class CIT_DataPatterned(CIT_Data):
    """
        Patterned data for mCIT.

        .. seealso::
            Pattern-related aspects are to be overwritten by custom pattern providers,
            for example :py:class:`CIT_DataPatterned_PersistentInTime` or
            :py:class:`CIT_DataPatterned_PesistentInSpace`.
    """


    
    def view_blocks(self, block_size:int) -> BlockView:
        """View as blocks of given size. The layout of blocks encodes the (prior) knowledge about patterns.

        :param block_size: requested block-size (the block-size of the result may not exactly match this number,
            if the underlying pattern provider cannot construct arbitrary block-sizes).
        :type block_size: int
        :return: view as pattern-aligned blocks
        :rtype: BlockView
        """
        raise NotImplementedError()
    
    @staticmethod
    def get_actual_block_format(requested_size: int) -> int|tuple[int,...]:
        """Get the actual (possibly multi-dimensional) format of blocks produced. Used for plotting.

        :param requested_size: The size of blocks requested.
        :type requested_size: int
        :return: Format of blocks produced.
        :rtype: int|tuple[int,...]
        """
        raise NotImplementedError()
    
    @staticmethod
    def reproject_blocks(value_per_block: np.ndarray, block_configuration: BlockView, data_configuration: tuple[int,...]) -> np.ndarray:
        """Reproject a function :math:`f` on blocks to the original index-set layout (for example time, space etc). Used for plotting.

        :param value_per_block: values of :math:`f` for each block
        :type value_per_block: np.ndarray
        :param block_configuration: the block-configuration (eg block-size) used
        :type block_configuration: BlockView
        :param data_configuration: the data-shape (per-variable) in the original data
        :type data_configuration: tuple[int,...]
        :return: plottable layout of :math:`f` as function of the original index-space
        :rtype: np.ndarray
        """
        raise NotImplementedError()



    def sample_count(self) -> int:
        """Get sample size.

        :return: sample-size N
        :rtype: int
        """
        return self.x_data.size
    
    def z_dim(self) -> int:
        """Get dimension (number of variables) of conditioning set Z.

        :return: dim(Z)
        :rtype: int
        """
        if self.z_data is None:
            return 0
        else:
            return self.z_data.shape[-1]
    
    def copy_and_center(self) -> 'CIT_DataPatterned':
        """Copy and subtract mean.

        :return: centered copy
        :rtype: CIT_DataPatterned
        """
        return self.clone_from_data(
            self.x_data-np.mean(self.x_data),
            self.y_data-np.mean(self.y_data),
            self.z_data-np.mean(self.z_data, axis=tuple(range(self.x_data.ndim))).reshape(*((1,)*self.x_data.ndim),self.z_dim()) if self.z_blocks is not None else None,
        )
    
    def view_blocks_trivial(self) -> BlockView:
        """View as trivial blocks (a single block of size N).

        :return: view by trivial blocks
        :rtype: BlockView
        """
        return BlockView(
            pattern_provider=self,
            cache_id=None if self.cache_id is None else (*self.cache_id, -1),
            x_blocks=self.x_data.reshape((1, -1)),
            y_blocks=self.y_data.reshape((1, -1)),
            z_blocks=self.z_data.reshape((1, -1, self.z_dim())) if self.z_dim() > 0 else None
        )
    
    def view_blocks_match(self, other: BlockView) -> BlockView:
        """View as blocks matching configuration (block-sizes etc) of another block-view.

        :param other: The other block-view, whose configuration should be copied.
        :type other: BlockView
        :return: A block-view of the data with same configuration as :param other:.
        :rtype: BlockView
        """
        return self.view_blocks(other.block_size())

    def bootstrap_unaligned_blocks(self, rng: np.random.Generator, bootstrap_block_count: int, block_size: int) -> BlockView:
        """Bootstrap random (unaligned) blocks.

        :param rng: A random number generator.
        :type rng: np.random.Generator (or similar, requires .integers of numpy.random.Generator)
        :param bootstrap_block_count: Number of blocks to bootstrap
        :type bootstrap_block_count: int
        :param block_size: Size per block
        :type block_size: int
        :return: A the bootstrapped blocks (actually a copy, not a view)
        :rtype: BlockView
        """
        indices = rng.integers(0, self.sample_count(), (bootstrap_block_count, block_size))
        return BlockView(self, None, self.x_data.reshape(-1)[indices], self.y_data.reshape(-1)[indices], self.z_data.reshape(-1,self.z_dim())[indices,:])

    @classmethod
    def clone_from_data(cls, X:np.ndarray,Y:np.ndarray,Z:np.ndarray) -> 'CIT_DataPatterned':
        """Attach the currently used pattern-provider to given data.

        :param X: X-data
        :type X: np.ndarray
        :param Y: Y-data
        :type Y: np.ndarray
        :param Z: Z-data
        :type Z: np.ndarray
        :return: Patterned data.
        :rtype: decltype(self), a type derived from :py:class:`CIT_DataPatterned`
        """
        return cls(X, Y, Z, cache_id=None)

    


class CIT_DataPatterned_PersistentInTime(CIT_DataPatterned):
    """
        Patterned data for mCIT. The implemented pattern captures persistent regimes in a single (eg time) direction.

        | x_data has shape=(N), where N is sample-size
        | y_data has shape=(N), where N is sample-size
        | z_data has shape=(N,k), where N is sample-size and k=dim(Z)

        .. seealso::
            See :ref:`overview on custom patterns <label-patterns>`.
            Methods are specified and documented on :py:class:`CIT_DataPatterned`.
    """
    
    
    def view_blocks(self, block_size:int) -> BlockView:
        block_count = int(self.sample_count()/block_size)
        aligned_N = block_size * block_count
        return BlockView(
            pattern_provider=self,
            cache_id=None if self.cache_id is None else (*self.cache_id, block_size),
            x_blocks=self.x_data[:aligned_N].reshape((block_count, block_size)),
            y_blocks=self.y_data[:aligned_N].reshape((block_count, block_size)),
            z_blocks=self.z_data[:aligned_N, :].reshape((block_count, block_size, -1)) if self.z_dim() > 0 else None
        )
    
    
    @staticmethod
    def get_actual_block_format(requested_size: int) -> int:
        return requested_size
    
    @staticmethod
    def reproject_blocks(value_per_block: np.ndarray, block_configuration: BlockView, data_configuration: tuple[int,...]) -> np.ndarray:
        return value_per_block.repeat(block_configuration.block_size())


class CIT_DataPatterned_PesistentInSpace(CIT_DataPatterned):
    """
        Patterned data for mCIT. The implemented pattern captures persistent regimes in two (eg spatial) direction.

        | x_data has shape=(w,h), where w, h are the width and height of the sample-grid.
        | y_data has shape=(w,h), where w, h are the width and height of the sample-grid.
        | z_data has shape=(w,h,k), where w, h are the width and height of the sample-grid, and k=dim(Z)

        .. seealso::
            See :ref:`overview on custom patterns <label-patterns>`.
            Methods are specified and documented on :py:class:`CIT_DataPatterned`.
    """
    

    def _get_full_size(self) -> tuple[int,int]:
        return self.x_data.shape
    @staticmethod
    def get_actual_block_format(requested_size: int) -> tuple[int,int]:
        sqr_side = int(np.ceil(np.sqrt(requested_size)-0.001))
        return (sqr_side, sqr_side)    
    
    def view_blocks(self, block_size:int) -> BlockView:        
        actual_block_size = self.get_actual_block_format(block_size)
        actual_block_size_total = np.prod(actual_block_size)
        block_counts = list(map(int, np.divide( self._get_full_size(), actual_block_size )))
        block_count_total = np.prod(block_counts)
        aligned_N = np.multiply(block_counts, actual_block_size)
        def extract_blocks(data):
            block_individual_axes = data[:aligned_N[0], :aligned_N[1]].reshape((block_counts[0], actual_block_size[0], block_counts[1], actual_block_size[1]))
            return block_individual_axes.transpose(0,2,1,3).reshape(block_count_total,actual_block_size_total)
        def extract_blocks_z(data_z):
            block_individual_axes = data_z[:aligned_N[0], :aligned_N[1],:].reshape((block_counts[0], actual_block_size[0], block_counts[1], actual_block_size[1], self.z_dim()))
            return block_individual_axes.transpose(0,2,1,3,4).reshape(block_count_total,actual_block_size_total, self.z_dim())

        return BlockView(
            pattern_provider=self,
            cache_id=None if self.cache_id is None else (*self.cache_id, actual_block_size),
            x_blocks=extract_blocks(self.x_data),
            y_blocks=extract_blocks(self.y_data),
            z_blocks=extract_blocks_z(self.z_data) if self.z_dim() > 0 else None
        )

    @staticmethod    
    def reproject_blocks(value_per_block: np.ndarray, block_configuration: BlockView, data_configuration: tuple[int,...]) -> np.ndarray:
        actual_block_size = block_configuration.pattern_provider.get_actual_block_format(block_configuration.block_size())
        directional_block_count = np.floor(np.asarray(data_configuration)/np.asarray(actual_block_size)).astype(int)
        reprojected_values = value_per_block.reshape(directional_block_count)
        for idx, directional_size in enumerate(actual_block_size):
            reprojected_values = reprojected_values.repeat(directional_size, axis=idx)
        return reprojected_values


    




class IManageData:
    """Specification of data-manager interface. Implement this to provide a custom data-manager.
    """

    def get_patterned_data(self, ci: CI_Identifier) -> CIT_DataPatterned: 
        """Get CIT-data with attached pattern-information.

        .. seealso::
            Details on patterns are provided at :ref:`label-patterns`.
            Details on cache-IDs are given at :ref:`label-cache-ids`.

        :param ci: The CI identified by its variable indices.
        :type ci: CI_Identifier
        :return: The CIT-data with attached pattern-provider.
        :rtype: CIT_DataPatterned
        """
        raise NotImplementedError() 

    def number_of_variables(self) -> int:
        """Get the number of variables (as used e.g. by PCMCI) in the current data-set.

        :return: Number of (contemporaneous) variables.
        :rtype: int
        """
        raise NotImplementedError()
    
    def total_sample_size(self) -> int:
        """Get the total sample-size.

        :return: sample-size
        :rtype: int
        """
        raise NotImplementedError()
    
    def reproject_blocks(self, value_per_block: np.ndarray, block_configuration: BlockView) -> np.ndarray:
        """Project function-values given on blocks back to original data-layout for plotting.

        :param value_per_block: function-values taken on blocks
        :type value_per_block: np.ndarray
        :param block_configuration: the block-layout (e.g. block-size)
        :type block_configuration: BlockView
        :return: the function-values taken in the original index-space.
        :rtype: np.ndarray
        """
        return self.pattern.reproject_blocks(value_per_block, block_configuration, self._data.shape[:-1])
    



    
class DataManager_NumpyArray_IID(IManageData):
    """Data-manager designed for use with IID data.
    """

    def __init__(self, data_indexed_by_sampleidx_variableidx:np.ndarray, copy_data:bool=True, pattern=CIT_DataPatterned_PersistentInTime, reproject_pattern_for_plotting=None):
        self._data = data_indexed_by_sampleidx_variableidx.copy() if copy_data else data_indexed_by_sampleidx_variableidx
        # protect against accidential modification:
        self._data.flags['WRITEABLE'] = False
        self.pattern = pattern
        self.reproject_pattern_for_plotting = reproject_pattern_for_plotting

    def get_patterned_data(self, ci: CI_Identifier[int]) -> CIT_DataPatterned:        
        # Multi-indexing (Z) in numpy will copy (ie data_z has its own malloc and memcpy),
        # thus we may not want to store data_z with the query in cache?
        # Note: x[...,k] accesses index k in the last axis ie [:,k] or [:,:,k] etc
        data_z = self._data[...,ci.idx_list_z] if len(ci.idx_list_z)>0 else None
        data_x = self._data[...,ci.idx_x]
        data_y = self._data[...,ci.idx_y]
        return self.pattern(x_data=data_x, y_data=data_y, z_data=data_z, cache_id=(self, ci))
    
    def number_of_variables(self) -> int:
        return self._data.shape[-1]
       
    def total_sample_size(self) -> int:
        return np.prod( self._data.shape[:-2] )



    
class DataManager_NumpyArray_Timeseries(IManageData):
    """Data-manager designed for use with time-series data.
    """

    def __init__(self, data_indexed_by_sampleidx_variableidx:np.ndarray, copy_data:bool=True, pattern=CIT_DataPatterned_PersistentInTime, reproject_pattern_for_plotting=None):
        self._data = data_indexed_by_sampleidx_variableidx.copy() if copy_data else data_indexed_by_sampleidx_variableidx
        # protect against accidential modification:
        self._data.flags['WRITEABLE'] = False
        self.pattern = pattern
        self.reproject_pattern_for_plotting = reproject_pattern_for_plotting

    def get_patterned_data(self, ci: CI_Identifier_TimeSeries) -> CIT_DataPatterned:      
        T, var_count_total = self._data.shape
        max_timelag = ci.max_abs_timelag()
        window = np.lib.stride_tricks.sliding_window_view(self._data, [max_timelag+1,var_count_total]) \
            .reshape(T-max_timelag, max_timelag+1, var_count_total)  # (window count, window length, variables)
        
        # -1 is last in window, so e.g. lag=0 is last in window, lag=-1 is second to last etc
        x_var, x_lag = ci.idx_x
        data_x = window[:, -1+x_lag, x_var]
        y_var, y_lag = ci.idx_y
        data_y = window[:, -1+y_lag, y_var]
        data_z = None
        if ci.z_dim() > 0:            
            z_vars = np.array([z_var for z_var, _ in ci.idx_list_z])
            z_lags = np.array([z_lag for _, z_lag in ci.idx_list_z])
            # Multi-indexing (Z) in numpy will copy (ie data_z has its own malloc and memcpy),
            # thus we may not want to store data_z with the query in cache?
            data_z = window[:, -1+z_lags, z_vars]
        return self.pattern(x_data=data_x, y_data=data_y, z_data=data_z, cache_id=(self, ci))
    
    def number_of_variables(self) -> int:
        return self._data.shape[1]
       
    def total_sample_size(self) -> int:
        return self._data.shape[0]



