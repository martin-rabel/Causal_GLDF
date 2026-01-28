from . import hyperparams
from . import data_management
from .cit import ParCorr
from .data_processing import Homogeneity_Binomial, WeakRegime_AcceptanceInterval, IndicatorImplication_AcceptanceInterval, mCIT, \
    ITestCI, IProvideAnalyticQuantilesForCIT, IProvideVarianceForCIT, ITestHomogeneity, ITestWeakRegime, ITestIndicatorImplications, ITestMarkedCI
from .independence_atoms import IndependenceAtoms_Backend, IndependenceAtoms_TimeSeries, IProvideIndependenceAtoms
from .hccd import Controller, ControllerTimeseriesMCI, ControllerTimeseriesLPCMCI, IPresentResult, IResolveRegimeStructure, graph_t, abstract_cd_t, IConstructStateSpace
from . import state_space_construction
import numpy as np
from dataclasses import dataclass

class cache_layer:
    """
        Flexible cache-layer.
    """
    def __init__(self, where: object, on: str, only_latest: bool):
        """Inject a cache-layer caching :py:meth:`!where.on`.

        :param where: stage where to inject the layer
        :type where: object
        :param on: name of the method to cache
        :type on: str
        :param only_latest: cache only the latest request (if true),
            or keep a dictionary of all previously executed requests (if false).
        :type only_latest: bool
        """
        assert hasattr(where, on)

        # if layer below is also a cache, should insert fname later
        self.extract_cache_id = where._extract_cache_id
        self.fname=on

        self.call_lower_layer = getattr(where, on)
        self.cache_object = (None, None) if only_latest else {}

        if only_latest:
            setattr(where, on, self._call_cached_latest)
        else:
            setattr(where, on, self._call_cached_full)

    def _call_cached_latest(self, *args1, **args2):
        request_id = self.extract_cache_id(self.fname, *args1, **args2)
        latest_id, latest_result = self.cache_object
        if (latest_id is not None) and (latest_id == request_id):
            return latest_result
        result = self.call_lower_layer(*args1, **args2)
        if request_id is not None:
            self.cache_object = (request_id, result)
        return result

    def _call_cached_full(self, *args1, **args2):
        request_id = self.extract_cache_id(self.fname, *args1, **args2)
        cached_result = self.cache_object.get(request_id)
        if cached_result is not None:
            return cached_result
        else:
            result = self.call_lower_layer(*args1, **args2)
            if request_id is not None:
                self.cache_object[request_id] = result
            return result


color_scheme = [ "blue", "orange", "cyan", "red", "green", "darkgray" ] #: These colors are used (in order, repeating) in labeled union graphs as "labels".

class LinkInfo:
    """
        Information attached to links for plotting graphs.
    """

    def __init__(self, r_info) -> None:
        if len(r_info) == 0 or isinstance( next(iter(r_info.keys()))[0], tuple ):
            self.regimy_links = {self._consistent_indexing(key[0][0], key[1][0]): value for key, value in r_info.items()}
        else:
            self.regimy_links = {self._consistent_indexing(key[0], key[1]): value for key, value in r_info.items()}

    @staticmethod
    def _consistent_indexing(u,v):
        return (u,0), (v,0)

    def vanishing_link(self, u,v) -> bool:
        """Check for the vanishing (regime-dependence) of a union-graph link.

        :return: regime-dependence
        :rtype: bool
        """
        u, v = self._consistent_indexing(u, v)
        return (u,v) in self.regimy_links or (v,u) in self.regimy_links
    def get_color(self, u,v):
        """
            Get color-label for a link in the "labeled" union-graph.
        """
        u, v = self._consistent_indexing(u, v)
        c = self.regimy_links[(u,v)] if (u,v) in self.regimy_links else self.regimy_links[(v,u)]
        return c




class ResolvableModelIndicator:
    """
        Structured result presenting results about a particular model-indicator.
    """

    def __init__(self, indicator_resolution: IResolveRegimeStructure, model_indicator, assigned_color):
        """Construct a structured result.

        :param indicator_resolution: Indicator resolution strategy.
        :type indicator_resolution: IResolveRegimeStructure
        :param model_indicator: underlying model-indicator
        :type model_indicator: state-space construction specific
        :param assigned_color: color "label"
        :type assigned_color: str
        """
        self._model_indicator = model_indicator
        self._indicator_resolution = indicator_resolution
        self.assigned_color = assigned_color

    def undirected_link(self) -> tuple:
        """The underlying undirected link.

        :return: the changing link in the model
        :rtype: tuple[data_management.var_index,data_management.var_index]
        """
        return self._model_indicator.undirected_link

    def resolve(self) -> np.ndarray|dict[str, np.ndarray]:
        """Approximately resolve the indicator relative to index-space.

        :return: resolved indicator (possibly as dictionary "label": values)
        :rtype: np.ndarray|dict[str, np.ndarray]
        """
        return self._indicator_resolution.resolve_model_indicator(self._model_indicator)

    def _plot_resolution_2D(self, resolved_data: np.ndarray) -> None:
        from matplotlib.colors import LinearSegmentedColormap
        color_map = LinearSegmentedColormap.from_list("temp_indicator_colormap", colors=[self.assigned_color, "white"], N=256, gamma=1.0)
        import matplotlib.pyplot as plt
        plt.imshow(resolved_data, cmap=color_map)

    def _plot_resolution_1D(self, resolved_data: np.ndarray, label=None, linestyle="solid") -> None:
        import matplotlib.pyplot as plt
        plt.plot(resolved_data, color=self.assigned_color, label=label, linestyle=linestyle)


    def _plot_resolution_auto(self, resolved_data, label=None, linestyle="solid")->None:
        if len(resolved_data.shape) == 1:
            self._plot_resolution_1D(resolved_data, label=label, linestyle=linestyle)
        elif len(resolved_data.shape) == 2:
            self._plot_resolution_2D(resolved_data)
        else:
            raise NotImplementedError("Currently auto-plotting is only implemented for 1D and 2D data. Use '.resolve()' and plot the data externally.")

    def plot_resolution(self) -> None:
        """Generate and plot an approximate resolution of the indicator in index-space.

        :raises NotImplementedError: Currently only supports 1D and 2D data.
        """
        resolved_data = self.resolve()
        if isinstance(resolved_data, dict):
            for idx, label_data in enumerate(resolved_data.items()):
                label, data = label_data
                self._plot_resolution_auto(data, label=label, linestyle=["solid", "dashed", "dotted", "dashdot"][idx])
        else:
            self._plot_resolution_auto(resolved_data)
        



class Result:
    """
        Structured result presenting output obtained from HCCD.
    """

    var_names : list[str] #: Write to this field before plotting to provide variable-names.

    def __init__(self, hccd_result: IPresentResult, indicator_resolution: IResolveRegimeStructure|None = None):
        """Construct from backend-result.

        :param hccd_result: backend hccd result
        :type hccd_result: IPresentResult
        :param indicator_resolution: indicator resolution strategy, defaults to None
        :type indicator_resolution: IResolveRegimeStructure | None, optional
        """
        self.hccd_result = hccd_result
        self.indicator_resolution = indicator_resolution
        self.var_names = None

    def union_graph(self) -> graph_t:
        """Get the estimated union-graph.

        :return: union-graph (tigramite format)
        :rtype: graph_t
        """
        return self.hccd_result.union_graph()

    def state_graphs(self) -> list[graph_t]:
        """Get state-specific graphs.

        :return: list of state-specific graphs (tigramite format)
        :rtype: list[graph_t]
        """
        return self.hccd_result.state_graphs()

    def model_indicators(self) -> list[ResolvableModelIndicator]:
        """Get structured result representing discovered model-indicators (changing links).

        :return: list of model indicators
        :rtype: list[ResolvableModelIndicator]
        """
        return [ResolvableModelIndicator(self.indicator_resolution, mi, color_scheme[idx%len(color_scheme)]) for idx, mi in enumerate(self.hccd_result.model_indicators())]

    def plot_labeled_union_graph(self, **args):
        """Plot the (color-)labeled union-graph.

        Keyword arguments are forwarded to (a modified version of) :py:meth:`!tigramite.plotting.plot_graph`.

        .. seealso::

            Uses colors from :py:data:`color_scheme`.
        """
        link_info = LinkInfo( {mi.undirected_link(): mi.assigned_color for mi in self.model_indicators()} )
        from .bridges.tigramite_plotting_modified import plot_graph
        if self.var_names is not None and "var_names" not in args:
            args["var_names"] = self.var_names
        plot_graph(graph=self.union_graph(), figsize=(4,4), special_links=link_info, **args)



# def unique_get(config_cls):
#     """Type decorator, use as @unique_get class ...

#     Make methods whose names start with "get_" instance-unique on this type, i.e.
#     modify them such that they are executed only on the first invocation, and
#     automatically cached for later requests.
#     """
#     # pythons for-loop-scoping bug still f***s lambdas as of v3.13, while partials are still
#     # exposed inconsistently for use as methods, so we need some 'creative' approach ...
#     def bind_captures(name, get_value):
#         def unique_entry(self):
#             if hasattr(self, f"_unique_{name}"):
#                 return getattr(self, f"_unique_{name}")
#             else:
#                 value = get_value(self)
#                 setattr(self, f"_unique_{name}", value)
#                 return value
#         return unique_entry
#     for elem in [elem for elem in dir(config_cls) if elem.startswith("get_")]:
#         setattr(config_cls, elem, bind_captures(name=elem, get_value=getattr(config_cls, elem)))
#     return config_cls
class Config:
    """
        Helper for setting up configuration-objects.
        Calling :py:meth:`finalize` will return a copy which is modified such as to execute each method prefixed with 'get\\_'
        once, caching the result.
    """
    # pythons for-loop-scoping bug still f***s lambdas as of v3.13, while partials are still
    # exposed inconsistently for use as methods, so we need some 'creative' approach ...
    def _make_getters_unqiue(self):
        def bind_captures(name, get_value):
            def unique_entry():
                if hasattr(self, f"_unique_{name}"):
                    return getattr(self, f"_unique_{name}")
                else:
                    value = get_value()
                    setattr(self, f"_unique_{name}", value)
                    return value
            return unique_entry
        for elem in [elem for elem in dir(self) if elem.startswith("get_")]:
            setattr(self, elem, bind_captures(name=elem, get_value=getattr(self, elem)))

    def finalize(self):
        """
            Modify methods prefixed by 'get\\_' to be executed at most once (on first invokation) and cached for further calls.
        """
        from copy import copy
        cpy = copy(self)
        cpy._make_getters_unqiue()
        return cpy



@dataclass
class ConfigureBackend(Config):
    """
        Modular configuration of extended independence atom backend.
    """

    data_manager : data_management.IManageData

    alpha : float = 0.01
    alpha_homogeneity : float = 0.01
    regimes_are_large : bool = True

    min_regime_fraction : float = 0.15

    enable_weak_test : bool = True
    enable_implication_test : bool = True


    def get_cit(self) -> ITestCI:
        cit = ParCorr(self.alpha)
         # Prepare for robust ci by caching latest result (if robust testing is with same block-size as homogeneity)
        self.cit_cache_layer = cache_layer(cit, "run_many", only_latest=True)
        return cit

    def cit_analytic_quantile_estimate(self) -> IProvideAnalyticQuantilesForCIT:
        return self.get_cit()
    def cit_variance_estimate(self) -> IProvideVarianceForCIT:
        return self.get_cit()

    def get_homogeneity_test(self) -> ITestHomogeneity:
        return Homogeneity_Binomial(hyperparams=hyperparams.Hyperparams_HomogeneityBinomial_ParCorr(alpha_error_control=self.alpha_homogeneity, regimes_are_large=self.regimes_are_large),
                                    cit=self.get_cit(), cit_analytic_quantile_estimate=self.cit_analytic_quantile_estimate())

    def get_weak_test(self) -> ITestWeakRegime:
        return WeakRegime_AcceptanceInterval(hyperparams=hyperparams.Hyperparams_WeakInterval_ParCorr(regimes_are_large=self.regimes_are_large),
                                      cit=self.get_cit(), cit_variance_estimate=self.cit_variance_estimate(), min_regime_fraction=self.min_regime_fraction)

    def get_mcit(self) -> ITestMarkedCI:
        return mCIT(cit=self.get_cit(), homogeneity_test=self.get_homogeneity_test(), weak_test=self.get_weak_test() if self.enable_weak_test else None)

    def get_implication_test(self) -> ITestIndicatorImplications:
        return IndicatorImplication_AcceptanceInterval(cit=self.get_cit(), cit_variance_estimate=self.cit_variance_estimate(),
                                                       hyperparams=hyperparams.Hyperparams_WeakInterval_ParCorr(regimes_are_large=self.regimes_are_large))

    def get_backend(self) -> IProvideIndependenceAtoms:
        m_cit = self.get_mcit()
        test_indicator_implication = self.get_implication_test() if self.enable_weak_test else None
        independence_atoms = IndependenceAtoms_Backend(data_manager=self.data_manager, m_cit=m_cit, implication_test=test_indicator_implication)
        cache_layer(independence_atoms, "marked_independence", only_latest=False)
        cache_layer(independence_atoms, "regime_implication", only_latest=False)
        return independence_atoms


#@unique_get applied late in run to ensure user-overwrites are treated consistently
@dataclass
class ConfigureHCCD(Config):
    """
        Modular configuration of backend for HCCD.
    """
    is_timeseries : bool #: is time-series data
    alpha : float = 0.01 #: confidence-parameter :math:`\alpha` for hypothesis-tests
    min_regime_fraction : float = 0.15 #: minimum regime-fraction to consider
    indicator_resolution_granularity : int =100 #: granularity used for post-process approximate indicator resultion
    regimes_are_large : bool = True #: use hyper-parameter sets optimized for large regimes (True is recommended, especially for time-series data)
    tau_max : int = 1 #: maximum lag to use for time-series algorithms (ignored for IID algorithms)

    # ts-specific
    alpha_pc1 : float = 0.1 #: confidence-parameter :math:`\alpha` for independence-tests in PC1-phase of PCMCI-family algorithms.

    _data : np.ndarray = None

    def get_data_manager(self) -> data_management.IManageData:
        if self.is_timeseries:
            return data_management.DataManager_NumpyArray_Timeseries(self._data)
        else:
            dim = len(self._data.shape) - 1
            if dim == 1:
                return data_management.DataManager_NumpyArray_IID(self._data)
            elif dim == 2:
                return data_management.DataManager_NumpyArray_IID(self._data, pattern=data_management.CIT_DataPatterned_PesistentInSpace)

    def get_backend_config(self) -> ConfigureBackend:
        return ConfigureBackend(
            data_manager=self.get_data_manager(),
            alpha=self.alpha,
            alpha_homogeneity=self.alpha,
            regimes_are_large=self.regimes_are_large,
            min_regime_fraction=self.min_regime_fraction,
            enable_weak_test=True,
            enable_implication_test=True
        ).finalize()
    def get_backend(self) -> IProvideIndependenceAtoms:
        config_backend = self.get_backend_config()
        return config_backend.get_backend()

    # time-series only
    def get_mci_backend(self) -> IProvideIndependenceAtoms:
        return self.get_backend()

    def alpha_homogeneity_pc1(self) -> float:
        return self.alpha # "keep" configuration
    def get_pc1_backend(self) -> IProvideIndependenceAtoms:
        config_backend = ConfigureBackend(
            data_manager=self.get_data_manager(),
            alpha=self.alpha_pc1,
            alpha_homogeneity=self.alpha_homogeneity_pc1(),
            regimes_are_large=self.regimes_are_large,
            min_regime_fraction=self.min_regime_fraction,
            enable_weak_test=False,
            enable_implication_test=False
        ).finalize()
        return config_backend.get_backend()

    def get_transitionable_backend(self) -> IndependenceAtoms_TimeSeries:
        return IndependenceAtoms_TimeSeries(independence_atoms_pc1=self.get_pc1_backend(), independence_atoms_mci=self.get_mci_backend())


    def get_universal_cd(self) -> abstract_cd_t:
        if self.is_timeseries:
            from .bridges import tigramite
            return tigramite.alg_pcmciplus(data_format=self.get_data_manager(), mci_transition_callback=self.get_transitionable_backend(), pcmci_obj_run_args=dict(tau_max=self.tau_max))
        else:
            from .bridges import causal_learn
            return causal_learn.alg_fci(data_format=self.get_data_manager())


    def get_state_space_construction(self) -> IConstructStateSpace:
        return state_space_construction.NoUnionCycles()

    def get_controller(self) -> Controller:
        if self.is_timeseries:
            return ControllerTimeseriesMCI(universal_cd=self.get_universal_cd(), testing_backend=self.get_transitionable_backend(), state_space_construction=self.get_state_space_construction())
        else:
            return Controller(universal_cd=self.get_universal_cd(), testing_backend=self.get_backend(), state_space_construction=self.get_state_space_construction())


    def get_cit(self) -> ITestCI:
        return self.get_backend_config().get_cit()

    def get_indicator_resultion(self) -> IResolveRegimeStructure:
        def resolve_by_dependence_score(data_blocks: data_management.BlockView) -> np.ndarray:
            return self.get_cit().run_many(data_blocks).block_scores
        return state_space_construction.ResolveByRepresentor(indicator_resolution_score=resolve_by_dependence_score, data_mgr=self.get_data_manager(), block_size=self.indicator_resolution_granularity)


    def _run(self, data: np.ndarray) -> Result:
        self._data = data
        internal_result = self.get_controller().run_hccd()
        return Result(hccd_result=internal_result, indicator_resolution=self.get_indicator_resultion())


    def run(self, data: np.ndarray) -> Result:
        return self.finalize()._run(data)


class ConfigureHCCD_LPCMCI(ConfigureHCCD):
    def __init__(self, regimes_are_large: bool=True, alpha: float=0.01, alpha_pc1: float=0.1, tau_max: int=1):
        ts_config_no_latents = configure_hccd_temporal_regimes(regimes_are_large=regimes_are_large, alpha=alpha, alpha_pc1=alpha_pc1, allow_latent_confounding=False, tau_max=tau_max)
        from dataclasses import asdict
        super().__init__(**asdict(ts_config_no_latents))


    def get_universal_cd(self) -> abstract_cd_t:
        from .bridges import tigramite
        return tigramite.alg_lpcmci(data_format=self.get_data_manager(),lpcmci_obj_run_args=dict(tau_max=self.tau_max))

    def get_controller(self) -> Controller:
        return ControllerTimeseriesLPCMCI(universal_cd=self.get_universal_cd(), testing_backend=self.get_transitionable_backend(), state_space_construction=self.get_state_space_construction())



def configure_hccd_temporal_regimes(regimes_are_large: bool=True, alpha: float=0.01, alpha_pc1: float=0.1, allow_latent_confounding: bool=False, tau_max: int=1) -> ConfigureHCCD:
    """Get standard-configuration of HCCD for temporal regimes. Uses PCMCI+ [R20]_ as default CD-algorithm.

    :param regimes_are_large: use hyper-parameter sets optimized for large regimes (True is recommended, especially for time-series data), defaults to True
    :type regimes_are_large: bool, optional
    :param alpha: confidence-parameter :math:`\\alpha` for hypothesis-tests, defaults to 0.01
    :type alpha: float, optional
    :param alpha_pc1: confidence-parameter :math:`\\alpha` for independence-tests in the PC1-phase, defaults to 0.1
    :type alpha_pc1: float, optional
    :param allow_latent_confounding: consider the possibility of latent confounding by using LPCMCI [GR20]_ (this implementation is currently experimental
        and regime-discovery may have low recall, see [RR25]_\\ ), defaults to False
    :type allow_latent_confounding: bool, optional
    :param tau_max: maximum lag in considered time-window, defaults to 1
    :type tau_max: int, optional
    :return: HCCD configuration used by :py:func:`run_hccd_temporal_regimes`.
    :rtype: ConfigureHCCD
    """
    if allow_latent_confounding:
        return ConfigureHCCD_LPCMCI(
            alpha=alpha,
            alpha_pc1=alpha_pc1,
            regimes_are_large=regimes_are_large,
            tau_max=tau_max
        )
    else:
        return ConfigureHCCD(
            is_timeseries=True,
            alpha=alpha,
            alpha_pc1=alpha_pc1,
            regimes_are_large=regimes_are_large,
            tau_max=tau_max
        )

def run_hccd_temporal_regimes(data: np.ndarray, regimes_are_large: bool=True, alpha: float=0.01, alpha_pc1: float=0.1, allow_latent_confounding: bool=False, tau_max: int=1) -> Result:
    """Run preconfigured HCCD for temporal regimes. Uses PCMCI+ [R20]_ as default CD-algorithm.

    :param data: observed data
    :type data: np.ndarray
    :param regimes_are_large: use hyper-parameter sets optimized for large regimes (True is recommended, especially for time-series data), defaults to True
    :type regimes_are_large: bool, optional
    :param alpha: confidence-parameter :math:`\\alpha` for hypothesis-tests, defaults to 0.01
    :type alpha: float, optional
    :param alpha_pc1: confidence-parameter :math:`\\alpha` for independence-tests in the PC1-phase, defaults to 0.1
    :type alpha_pc1: float, optional
    :param allow_latent_confounding: consider the possibility of latent confounding by using LPCMCI [GR20]_ (this implementation is currently experimental
        and regime-discovery may have low recall, see [RR25]_\\ ), defaults to False
    :type allow_latent_confounding: bool, optional
    :param tau_max: maximum lag in considered time-window, defaults to 1
    :type tau_max: int, optional
    :return: Structured HCCD result.
    :rtype: Result
    """
    config = configure_hccd_temporal_regimes(regimes_are_large=regimes_are_large, alpha=alpha, alpha_pc1=alpha_pc1, allow_latent_confounding=allow_latent_confounding, tau_max=tau_max)
    return config.run(data)

def configure_hccd_spatial_regimes(regimes_are_large: bool=True, alpha: float=0.01) -> ConfigureHCCD:
    """Get standard-configuration of HCCD for spatial (2 dimensional, non-time-series) data. Uses FCI [SGS01]_ as default CD-algorithm.

    :param regimes_are_large: use hyper-parameter sets optimized for large regimes (True is recommended, especially for time-series data), defaults to True
    :type regimes_are_large: bool, optional
    :param alpha: confidence-parameter :math:`\\alpha` for hypothesis-tests, defaults to 0.01
    :type alpha: float, optional
    :return: HCCD configuration used by :py:func:`run_hccd_spatial_regimes`.
    :rtype: ConfigureHCCD
    """
    return ConfigureHCCD(
        is_timeseries=False,
        alpha=alpha,
        regimes_are_large=regimes_are_large
    )

def run_hccd_spatial_regimes(data: np.ndarray, regimes_are_large: bool=True, alpha: float=0.01) -> Result:
    """Run preconfigured HCCD for spatial (2 dimensional, non-time-series) data. Uses FCI [SGS01]_ as default CD-algorithm.

    :param data: observed data
    :type data: np.ndarray
    :param regimes_are_large: use hyper-parameter sets optimized for large regimes (True is recommended, especially for time-series data), defaults to True
    :type regimes_are_large: bool, optional
    :param alpha: confidence-parameter :math:`\\alpha` for hypothesis-tests, defaults to 0.01
    :type alpha: float, optional
    :return: Structured HCCD result.
    :rtype: Result
    """
    config = configure_hccd_spatial_regimes(regimes_are_large=regimes_are_large, alpha=alpha)
    return config.run(data)
