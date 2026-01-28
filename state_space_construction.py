from .hccd import IRepresentState, IRepresentStateSpace, IConstructStateSpace, IPresentResult, IProvideIndependenceAtoms, IResolveRegimeStructure, graph_t
from .data_management import CI_Identifier, BlockView

from typing import Callable

import numpy as np
from dataclasses import dataclass
from itertools import chain, combinations, product
# cf https://docs.python.org/3/library/itertools.html#itertools-recipes:
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
def powerset_as_list_of_sets(iterable):
    return [set(as_tuple) for as_tuple in powerset(iterable)]
def all_binary_combinations(bit_count):
    return product([0, 1], repeat=bit_count)


@dataclass
class ModelIndicator:
    """
        Represents a model-indicator in the current baseline implementation.
    """
    undirected_link: tuple      #: The link in the model on which a change was detected.
    representor: CI_Identifier  #: A representing test (this test is independent iff the model-indicator is zero.

    def __hash__(self):
        return hash(self.undirected_link)
    def __eq__(self, other: 'ModelIndicator | tuple'):
        if isinstance(other, ModelIndicator):
            return other.undirected_link == self.undirected_link
        else:
            return other == self.undirected_link


class State(IRepresentState):
    """
        Represents a state in the current baseline implementation.
    """

    def __init__(self, state_space: 'StateSpace', model_indicator_active: dict[ModelIndicator, bool]):
        self._state_space = state_space
        self.implied = set()
        self.model_indicator_active = model_indicator_active

    def state_space(self) -> IConstructStateSpace:
        return self._state_space

    def add_implication(self, ci: CI_Identifier):
        self.implied.add(ci)

    def overwrites_ci(self, ci: CI_Identifier):
        return self._state_space.controls_ci(ci)
    
    def get_ci_pseudo_value(self, ci: CI_Identifier):
        return ci not in self.implied
    

    def _implies_all_conditions(self, set_of_conditions: set[tuple]) -> bool:
        return all( [self.model_indicator_active[mi] for mi in set_of_conditions] )
    
    def _implies_all_conditions_in_at_least_one_listed_set(self, list_of_sets_of_conditions: list[set[tuple]]) -> bool:
        return any( [self._implies_all_conditions(set_of_condition) for set_of_condition in list_of_sets_of_conditions] )

class StateSpace(IRepresentStateSpace):
    """
        Represents the state-space in the current baseline implementation.
    """
    def __init__(self, model_indicators: 'list[ModelIndicator]'=[], marked_ci: set[CI_Identifier]=set()):
        self.model_indicators = model_indicators
        self.marked_ci = marked_ci
        self._states = self._build_states()
    
    def _fold_model_indicator_activity_from_list_into_dict(self, model_indicator_activity: tuple) -> 'dict[ModelIndicator, bool]':
        return {mi: value for mi, value in zip(self.model_indicators, model_indicator_activity)}        

    def _build_states(self) -> list[State]:
        return [State(state_space=self, model_indicator_active=self._fold_model_indicator_activity_from_list_into_dict(model_indicator_activity))
                for model_indicator_activity
                in all_binary_combinations(len(self.model_indicators))]

    def is_trivial(self) -> bool:
        return len(self.model_indicators) == 0

    def controls_ci(self, ci: CI_Identifier):
        return ci in self.marked_ci

    def states(self) -> list[State]:
        return self._states

    def finalize(self, graphs: dict[IRepresentState,graph_t]) -> IPresentResult:
        return Unionize_Translate_Transfer_NoUnionCylces.obtain_translation_result(state_space=self, graphs=graphs)
    
    
    def states_which_imply(self, list_of_conditions: list[set[tuple]]) -> list[State]:
        return [state for state in self.states()
                if state._implies_all_conditions_in_at_least_one_listed_set(list_of_conditions)]





class ModelIndicators_NoUnionCycles:    
    """
        In the current baseline implementation, in a first phase, model indicators are
        found as a maximum by semi-ordering based on indicator-implications.

        *Implements phase I of Algo. 3 from* [RR25]_\\ *.*
    """

    def __init__(self, testing_backend: IProvideIndependenceAtoms, marked_tests: set[CI_Identifier]):
        self.testing_backend = testing_backend
        self.marked_tests = marked_tests
        self.model_indicators = list(self._initialize_model_indicators())
        self.state_space = StateSpace(model_indicators=self.model_indicators, marked_ci=self.marked_tests)
    

    def _initialize_model_indicators(self) -> list[ModelIndicator]:
        links = set(ci.undirected_link() for ci in self.marked_tests
                    if not self.testing_backend.found_globally_independent_for_some_Z(ci.undirected_link()))
        return [ModelIndicator(undirected_link=link, representor=self._model_indicator_representator(link=link))
                for link in links]


    def _model_indicator_representator(self, link) -> CI_Identifier:
        smallest_element = None
        for relevant_test in [marked_test for marked_test in self.marked_tests if marked_test.undirected_link() == link]:
            if smallest_element is None:
                smallest_element = relevant_test
            elif self.testing_backend.regime_implication([relevant_test], smallest_element) \
                and not self.testing_backend.regime_implication([smallest_element], relevant_test):
                # switch conservatively (ordering is such that small conditioning sets are first)
                smallest_element = relevant_test
        assert smallest_element is not None, "This should always find a representor for each model indicator!"
        return smallest_element
    




class NoUnionCycles(IConstructStateSpace):
    """
        Current baseline implementation.

        *Implements Algo. 3 from* [RR25]_\\ *.*
    """

    def __init__(self):       
        self.testing_backend = None 
        self.model_indicators = None
        self.state_space = None

    def construct_statespace(self, testing_backend: IProvideIndependenceAtoms, marked_tests: set[CI_Identifier], previous_graphs) -> IRepresentStateSpace:
        self.testing_backend = testing_backend 

        model_indicator_construction = ModelIndicators_NoUnionCycles(testing_backend, marked_tests)
        self.model_indicators = model_indicator_construction.model_indicators
        self.state_space = model_indicator_construction.state_space

        for ci in marked_tests:
            self.translate_ci(ci)

        return self.state_space


    def store_translated_ci(self, ci: CI_Identifier, implied_by: list[set[tuple]]) -> None:
        for state in self.state_space.states_which_imply(implied_by):
            # state => implied_by => ci
            state.add_implication(ci)

    def translate_ci(self, ci: CI_Identifier) -> None:
        """        
        Translate detected indicators.

        *Implements phase II of Algo. 3 in* [RR25]_\\ *.*
        
        :param ci: marked CI test
        :type ci: CI_Identifier
        """""
        candidates = powerset_as_list_of_sets(self.model_indicators)
        empty_set = candidates.pop(0) # pop empty set at index 0
        assert len(empty_set) == 0

        necessary = []

        if ci.undirected_link() in self.model_indicators:
            # the X and Y are always dependent if the direct link is there
            candidates = list([c for c in candidates if ci.undirected_link() in c])

        while len(candidates) > 0: # python lists are not lists, so iterator-lifetime is wonky
            if len(candidates) == 1 and len(necessary) == 0:
                return self.store_translated_ci(ci, implied_by=candidates)
            c: set[ModelIndicator] = candidates.pop(0)
            representors_of_c = list([mi.representor for mi in c])
            if self.testing_backend.regime_implication(representors_of_c, ci):
                candidates = [c_ for c_ in candidates if not c <= c_] # "<=" is subset operator
                necessary.append(c)
    
        return self.store_translated_ci(ci, implied_by=necessary)


    
from .data_processing import ITestCI
from .data_management import IManageData
    
class StructuredResultWithTranslation(IPresentResult):     
    """
        Represents the (translated and transfered) result/labeled union graph
        in the current baseline implementation.
    """   

    def __init__(self, union_graph: graph_t, model_indicators: list[ModelIndicator], graphs: dict[IRepresentState, graph_t]):
        self._union_graph = union_graph
        self._model_indicators = model_indicators
        self._state_graphs = list(graphs.values())

    def union_graph(self) -> graph_t:
        return self._union_graph    
    
    def state_graphs(self) -> list[graph_t]:
        raise self._state_graphs
    
    def model_indicators(self) -> list:
        return self._model_indicators

        

class Unionize_Translate_Transfer_NoUnionCylces:   
    """
        Namescope for helpers for constructiong a labeled union-graph.
    """
    @staticmethod 
    def unionize_edgemark(a, b):        
        if a == b:
            return a
        elif a == 'x' or b == 'x':
            return 'x'
        elif a == 'o':
            return b
        elif b == 'o':
            return a
        else:
            return 'x'

    @classmethod
    def unionize_edge(cls, a, b):
        if a == '':
            return b
        if b == '':
            return a
        lhs = cls.unionize_edgemark(a[0], b[0])
        rhs = cls.unionize_edgemark(a[2], b[2])
        return lhs + "-" + rhs

    @classmethod
    def unionize_and_transfer(cls, graphs):
        result = None
        for g in graphs:
            if result is None:
                result = g
            else:
                result = np.array([cls.unionize_edge(edge_a, edge_b) for edge_a, edge_b in zip(result.flatten(), g.flatten())]).reshape(result.shape)
        return result
    
    @classmethod
    def obtain_translation_result(cls, state_space: StateSpace, graphs: dict[IRepresentState,graph_t]):
        # make results ready for serialization and easy plotting etc
        return StructuredResultWithTranslation(
            union_graph = cls.unionize_and_transfer(graphs.values()),
            model_indicators = state_space.model_indicators,
            graphs = graphs
        )


class ResolveByRepresentor(IResolveRegimeStructure):
    """
        Primitive approximate resolution of model-indicators by representors.
    """

    def __init__(self, indicator_resolution_score: Callable[[BlockView], np.ndarray], data_mgr: IManageData, block_size: int):
        self.indicator_resolution_score = indicator_resolution_score
        self.data_mgr = data_mgr
        self.block_size = block_size

    def resolve_model_indicator(self, model_indicator: ModelIndicator) -> np.ndarray:
        patterned_data = self.data_mgr.get_patterned_data(model_indicator.representor).view_blocks(self.block_size)
        result = self.indicator_resolution_score( patterned_data )
        return self.data_mgr.reproject_blocks(result, block_configuration=patterned_data)