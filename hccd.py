from .data_management import CI_Identifier, CI_Identifier_TimeSeries
from .data_processing import ITestMarkedCI
import numpy as np
from collections.abc import Callable
from typing import Literal



type abstract_cit_t = Callable[[CI_Identifier],bool]        #: Specifies the signature of abstract cits as used by CD-algorithms :py:type:`abstract_cd_t`.
type graph_t = np.ndarray                                   #: Used to annotate graphs in tigramite-format.
type abstract_cd_t = Callable[[abstract_cit_t], graph_t]    #: Specifies the signature of abstract CD-algorithms.


class IProvideIndependenceAtoms:
    """
        Interface specifying how to expose (custom) implementations of independence-atom providing backends.
        Provide (as opposed to test) here means that :py:mod:`data_management<GLDF.data_management>` and
        :py:mod:`data_processing<GLDF.data_processing>` functionality are bundled.
    """

    def marked_independence(self, ci: CI_Identifier) -> ITestMarkedCI.Result:
        """Provide a marked-independence statement.

        .. seealso::
            Provides functionality described by
            :py:meth:`TestMarkedCI.marked_independence<GLDF.TestMarkedCI.marked_independence>`
            but with actual data made opaque.

        :param ci: conditional independence to test
        :type ci: CI_Identifier
        :return: marked CIT result
        :rtype: ITestMarkedCI.Result
        """
        raise NotImplementedError()

    def regime_implication(self, lhs: list[CI_Identifier], rhs: CI_Identifier) -> bool:
        """Provide a regime-implication statement.

        .. seealso::
            Provides functionality described by
            :py:meth:`ITestIndicatorImplications.is_implied_regime<GLDF.ITestIndicatorImplications.is_implied_regime>`
            but with actual data made opaque.

        :param lhs: lhs of the implication
        :type lhs: list[CI_Identifier]
        :param rhs: rhs of the implication
        :type rhs: CI_Identifier
        :return: test-result for truth-value of the implication
        :rtype: bool
        """
        raise NotImplementedError()

    def found_globally_independent_for_some_Z(self, undirected_link: tuple) -> bool:
        """Provide information about wheter among the test-results provided so far,
        there was a CIT on the specified :py:obj:`undirected_link` for any conditioning
        set Z which was reported globally independent.

        :param undirected_link: the undirected link on which to check for independent CIT-results
        :type undirected_link: tuple
        :return: Truth-value about wheter any globally independent outcome was encountered on this link.
        :rtype: bool
        """
        raise NotImplementedError()

    def _extract_cache_id(self, fname: str, *args, **kwargs) -> tuple:
        args = (*args, *(kwarg for kwarg in kwargs.values()))
        if fname == "marked_independence":
            return args[0] # "ci"
        else:
            return (tuple(sorted(args[0])), args[1]) # "lhs", "rhs"



class IHandleExplicitTransitionToMCI:
    """
        Interface specifying how to expose (custom) implementations of independence-atom providing backends
        for PCMCI-familiy [RNK+19]_ algorithms. These backends typically handle PC1 and MCI tests differently,
        and through this interface are notified by the HCCD-controller about transitions between these phases.
    """
    def enter_pc1(self) -> None:
        """Callback for notification that the underlying cd-algorithm has (re)entered
        (what is considered by the controller) to be part of the PC1-phase.
        """
        raise NotImplementedError()

    def enter_mci(self) -> None:
        """Callback for notification that the underlying cd-algorithm has (re)entered
        (what is considered by the controller) to be part of the MCI-phase.
        """
        raise NotImplementedError()




class IPresentResult:

    """
        Interface specifying how to expose (custom) implementations of backend hccd-results.

        .. seealso::
            The :py:mod:`frontend<GLDF.frontend>` typically translates this into a more
            user-friendly :py:class:`frontend.Result<GLDF.frontend.Result>`.
    """

    def union_graph(self) -> graph_t:
        """Get the union-graph

        :return: union-graph (tigramite encoded)
        :rtype: graph_t
        """
        raise NotImplementedError()

    def state_graphs(self) -> list[graph_t]:
        """Get the state-specific graphs.

        :return: list of state-specific graphs (tigramite encoded)
        :rtype: list[graph_t]
        """
        raise NotImplementedError()

    def model_indicators(self) -> list:
        """Get the model-indicators. Translation is state-space-construction specific.

        :return: model-indicators
        :rtype: list
        """
        raise NotImplementedError()


class IResolveRegimeStructure:
    """
        Interface specifying how to expose (custom) implementations of approximate
        regime-structure resolution for post-processing.
    """
    def resolve_model_indicator(self, model_indicator) -> np.ndarray:
        """Resolve a model-indicator.

        :param model_indicator: model-indicator
        :type model_indicator: state-space construction specific encoding
        :return: Resolved (in index-space) indicator.
        :rtype: np.ndarray
        """
        raise NotImplementedError()


class IRepresentState:
    """
        Interface specifying how to expose (custom) implementations of state-representation
        during state-space construction for use with :py:class:`Controller`.
        State-space construction will typically internally attach further information,
        this interface only specifies which aspects must be exposed for the :py:class:`Controller`
        during HCCD.
    """

    def state_space(self) -> 'IRepresentStateSpace':
        """Get containing state-space.

        :return: state-space
        :rtype: IRepresentStateSpace
        """
        raise NotImplementedError()

    def overwrites_ci(self, ci: CI_Identifier) -> bool:
        """Is this conditional independence marked?

        :param ci: conditional independence
        :type ci: CI_Identifier
        :return: truth-value of CI being marked
        :rtype: bool
        """
        raise NotImplementedError()

    def get_ci_pseudo_value(self, ci: CI_Identifier) -> bool:
        """Get a (state-encoded) value for a marked conditional independence.

        :param ci: marked conditional independence
        :type ci: CI_Identifier
        :return: state-specific dependence-value (true for dependent).
        :rtype: bool
        """
        raise NotImplementedError()


class IRepresentStateSpace:
    """
        Interface specifying how to expose (custom) implementations of state-space-representation
        during state-space construction for use with :py:class:`Controller`.
        State-space construction will typically internally attach further information,
        this interface only specifies which aspects must be exposed for the :py:class:`Controller`
        during HCCD.
    """
    def states(self) -> list[IRepresentState]:
        """States contained in this state-space.

        :return: list of states
        :rtype: list[IRepresentState]
        """
        raise NotImplementedError()

    #todo
    def finalize(self, graphs: dict[IRepresentState,graph_t]) -> IPresentResult:
        """Translate to model-properties and transfer information between states
        into a result-summary.

        :param graphs: state-specific graph for each state
        :type graphs: dict[IRepresentState,graph_t]
        :return: the summarized result
        :rtype: IPresentResult
        """
        # translate_and_transfer
        raise NotImplementedError()


class IConstructStateSpace:
    """
        Interface specifying how to expose (custom) implementations of state-space-construction
        for use with :py:class:`Controller`.
    """
    def construct_statespace(self, testing_backend: IProvideIndependenceAtoms, marked_tests: set[CI_Identifier], previous_graphs: dict[IRepresentState,graph_t]) -> IRepresentStateSpace:
        """Construct state-space.

        :param testing_backend: the independece-atom backend
        :type testing_backend: IProvideIndependenceAtoms
        :param marked_tests: the set of marked conditional independencies
        :type marked_tests: set[CI_Identifier]
        :param previous_graphs: state-specific graph for each state found in the previous iteration
        :type previous_graphs: dict[IRepresentState,graph_t]
        :return: state-space
        :rtype: IRepresentStateSpace
        """
        raise NotImplementedError()



class Controller:
    """
    The HCCD-controller. Coordinates repeated CD-algorithm runs with state-space construction.

    .. seealso::
        Specialized versions for PCMCI-family time-series algorithms :py:class:`ControllerTimeseriesMCI`
        and for LPCMCI :py:class:`ControllerTimeseriesLPCMCI` are available.
    """

    def __init__(self, universal_cd: abstract_cd_t, testing_backend: IProvideIndependenceAtoms, state_space_construction: IConstructStateSpace):
        """Construct from components.

        :param universal_cd: universal (underlying) CD-algorithm
        :type universal_cd: abstract_cd_t
        :param testing_backend: independence-testing backend to use
        :type testing_backend: IProvideIndependenceAtoms
        :param state_space_construction: state-space construction strategy to use
        :type state_space_construction: IConstructStateSpace
        """
        self.CD = universal_cd
        self.testing_backend = testing_backend
        self.state_space_construction = state_space_construction

    def get_marked_independence(self, ci):
        """provide an extension hook to attach an observer to marked-independence
        lookups by overriding this method, see e.g. :py:class:`ControllerTimeseriesLPCMCI`.
        """
        return self.testing_backend.marked_independence(ci)

    def run_cd(self, state: IRepresentState) -> tuple[graph_t, set[CI_Identifier]]:
        """*Implements part of the "core-algorithm" Algo. 1 in* [RR25]_.

        Run underlying CD with a state-specific pseudo-cit.

        :param state: state considered active
        :type state: IRepresentState
        :return: tuple consisting of state-graph and set of (newly) marked tests.
        :rtype: tuple[graph_t, set[CI_Identifier]]
        """
        newly_marked_tests = set()

        def pseudo_cit_is_dependent(ci: CI_Identifier) -> bool:
            if state.overwrites_ci(ci):
                return state.get_ci_pseudo_value(ci)
            else:
                result = self.get_marked_independence(ci)
                if result.is_regime():
                    newly_marked_tests.add(ci)
                    return True
                else:
                    return result.is_globally_dependent()

        graph = self.CD( generalized_cit=pseudo_cit_is_dependent )
        return graph, newly_marked_tests


    def run_hccd(self, max_iterations: int=10) -> IPresentResult:
        """*Implements part of the "core-algorithm" Algo. 1 from* [RR25]_.

        Run HCCD.

        :param max_iterations: limit for iterations, defaults to 10
        :type max_iterations: int, optional
        :raises RuntimeError: Throws an exception if maximum iterations
            are reached.
        :return: hccd result
        :rtype: IPresentResult
        """
        marked_tests = set()
        converged = False
        graphs = None

        while not converged:
            state_space = self.state_space_construction.construct_statespace(testing_backend=self.testing_backend, marked_tests=marked_tests, previous_graphs=graphs)
            graphs = {}
            all_newly_marked_tests = set()
            for state in state_space.states():
                graphs[state], newly_marked_tests_from_current_state = self.run_cd(state)
                all_newly_marked_tests = set.union(all_newly_marked_tests, newly_marked_tests_from_current_state)

            # Check for convergence
            marked_tests, converged = self.check_for_convergence(previously_marked=marked_tests, newly_marked=all_newly_marked_tests, graphs=graphs)

            # Bail if there is a serious convergence issue
            if max_iterations == 1:
                raise RuntimeError("run_regime_cd reached maximum number of iterations. Terminating with last state.")
            else:
                max_iterations -= 1

        return state_space.finalize(graphs=graphs)



    def check_for_convergence(self, previously_marked: set[CI_Identifier], newly_marked: set[CI_Identifier], graphs)\
        ->  tuple[set[CI_Identifier], bool]:
        """Check if the core-algorithm has converged. May be overridden by more derived controllers.
        """
        previous_count = len(previously_marked)
        marked = set.union(previously_marked, newly_marked)
        return marked, (previous_count == len(marked))



class ControllerTimeseriesMCI(Controller):
    """Timer-series specific controller, for PCMCI [RNK+19]_ family algorithms. See §B.6 in [RR25]_\\ .

    .. seealso::
        Details are descibed at :py:class:`Controller`.
    """

    def __init__(self, universal_cd: abstract_cd_t, testing_backend: IProvideIndependenceAtoms, state_space_construction: IConstructStateSpace):
        super().__init__(universal_cd, testing_backend, state_space_construction)
        if not ( isinstance(testing_backend, IHandleExplicitTransitionToMCI) ):
            raise RuntimeError( "Timeseries with MCI should use a transitionable backend, see 'independence_atoms.IndependenceAtoms_TimeSeries'." )



class ControllerTimeseriesLPCMCI(Controller):
    """LPCMCI [GR20]_ specific controller.

    .. seealso::
        Details are descibed at :py:class:`Controller`.
    """
    def __init__(self, universal_cd: abstract_cd_t, testing_backend: IProvideIndependenceAtoms, state_space_construction: IConstructStateSpace):
        super().__init__(universal_cd, testing_backend, state_space_construction)
        if not ( isinstance(testing_backend, IHandleExplicitTransitionToMCI) ):
            raise RuntimeError( "Timeseries with MCI should use a transitionable backend, see 'independence_atoms.IndependenceAtoms_TimeSeries'." )
        self.union_graph = None



    def check_for_convergence(self, previously_marked: set[CI_Identifier], newly_marked: set[CI_Identifier], graphs: dict)\
        ->  tuple[set[CI_Identifier], bool]:
        """
            Overriden from :py:class:`Controller` to reset after an initial iteration (union-graph discovery),
            cf :py:meth:`get_marked_independence`.
        """
        if self.union_graph is None:
            assert len(graphs) == 1
            _, self.union_graph = graphs.popitem()
            return set(), False
        else:
            return super().check_for_convergence(previously_marked, newly_marked, graphs)

    def lagged_parents(self, node: int, lag_shift: int=0) -> set[tuple[int,int]]:
        lagged_links_into_node = (self.union_graph[:,node,1:] == '-->')
        parents, parent_lags = np.nonzero(lagged_links_into_node)
        parent_lags = -(parent_lags + 1 - lag_shift)
        parent_ts = zip(parents, parent_lags)
        return set(parent_ts)

    def contains_all_lagged_parents(self, ci: CI_Identifier_TimeSeries) -> bool:
        # use union-graph to determine this
        return ( ( self.lagged_parents(*ci.idx_x) <= ci.conditioning_set() )   # <= is "subset"-test on sets
             and ( self.lagged_parents(*ci.idx_y) <= ci.conditioning_set() ) )

    def should_consider_mci(self, ci: CI_Identifier_TimeSeries) -> bool:
        return (self.union_graph is not None) and self.contains_all_lagged_parents(ci)

    def get_marked_independence(self, ci):
        """
            Overriden from :py:class:`Controller` to attach logic for deciding
            if this test should be considerd MCI based on the union-graph.
        """
        if self.should_consider_mci(ci):
            self.testing_backend.enter_mci()
        else:
            self.testing_backend.enter_pc1()
        return super().get_marked_independence(ci)
