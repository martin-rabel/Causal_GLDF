try:
    import causallearn.search
except ModuleNotFoundError:
    raise RuntimeError("Causal-learn bridge could not be loaded. Is causal-learn installed?")

from causallearn.utils import cit

import numpy as np
from functools import partial
from typing import Callable

from ..hccd import  CI_Identifier, abstract_cit_t, abstract_cd_t
from ..data_management import IManageData


# in theory pyhton can copy functions, in practise it cannot (as of v3.13), so we copy-paste the code fromcausallearn.utils instead
def named_CIT(data, method='fisherz', **kwargs):
    if method == cit.fisherz:
        return cit.FisherZ(data, **kwargs)
    elif method == cit.kci:
        return cit.KCI(data, **kwargs)
    elif method in [cit.chisq, cit.gsq]:
        return cit.Chisq_or_Gsq(data, method_name=method, **kwargs)
    elif method == cit.mv_fisherz:
        return cit.MV_FisherZ(data, **kwargs)
    elif method == cit.mc_fisherz:
        return cit.MC_FisherZ(data, **kwargs)
    elif method == cit.d_separation:
        return cit.D_Separation(data, **kwargs)
    else:
        raise ValueError("Unknown method: {}".format(method))



def pick_CIT(data, method, **kwargs):
    if isinstance(method, str):
        return named_CIT(data, method, **kwargs)
    else:
        return method.provide_with_cl_args(**kwargs)
    
cit.CIT = pick_CIT
    
cit.CIT_Base = object # fix "validation" in fci

from causallearn.search.ConstraintBased.PC import pc as _cl_impl_pc
from causallearn.search.ConstraintBased.FCI import fci as _cl_impl_fci
import causallearn.search.ConstraintBased.FCI as _FCI_module

# in 'get_color_edges', l. 627 of causallearn.search.ConstraintBased.FCI there is a print command
# apparently left over from debugging, which will spam links,
# suppress all prints from this modul by brute-force:
_FCI_module.print = lambda *args : None 


def causallearn_graph_to_tigramite_graph_pc(G):
    N, validate = G.shape
    assert validate == N
    result = np.full_like(G, '', dtype='<U3')
    for i in range(N):
        for j in range(N):
            if G[i,j] == +1:
                if G[j,i] == +1:
                    result[i,j] = "<->"
                else:
                    assert G[j,i] == -1
                    result[i,j] = "<--"
            elif G[i,j] == -1:
                if G[j,i] == +1:
                    result[i,j] = "-->"
                else:
                    assert G[j,i] == -1
                    result[i,j] = "o-o"
    return result
    
def _np_replace_char(data, i, j, idx, value):
    current = list(data[i,j])
    current[idx] = value
    data[i,j] = "".join(current)


def causallearn_graph_to_tigramite_graph_fci(G):
    N, validate = G.shape
    assert validate == N
    result = np.full_like(G, '', dtype='<U3')
    for i in range(N):
        for j in range(N):
            if G[i,j] == 0:
                assert G[j,i] == 0
                continue
            result[i,j] = "?-?"
            if G[i,j] == +1:
                _np_replace_char( result, i, j, 0, '<' )
            elif G[i,j] == -1:
                _np_replace_char( result, i, j, 0, '-' )
            elif G[i,j] == 2:
                _np_replace_char( result, i, j, 0, 'o' )
            else:
                assert False
            if G[j,i] == +1:
                _np_replace_char( result, i, j, 2, '>' )
            elif G[j,i] == -1:
                _np_replace_char( result, i, j, 2, '-' )
            elif G[j,i] == 2:
                _np_replace_char( result, i, j, 2, 'o' )
            else:
                assert False
    return result
                    

def causallearn_graph_to_tigramite_graph(graph):
    G = None
    if isinstance(graph, tuple):
        general_graph, _ = graph
        G = general_graph.graph # fci
        return causallearn_graph_to_tigramite_graph_fci(G)
    else:
        G = graph.G.graph #  pc
        return causallearn_graph_to_tigramite_graph_pc(G)

 
class WrapCIT_CausalLearnIndexing:
    def __init__(self, generalized_cit: abstract_cit_t):
        self.generalized_cit = generalized_cit
        self.method = "custom"

    def __call__(self, x_idx: int, y_idx: int, Z: list[int]) -> float:
        ci = CI_Identifier(idx_x=x_idx, idx_y=y_idx, idx_list_z=Z)
        result = self.generalized_cit(ci)
        # cl wants a pvalue under independence (signigicance is decided by test, cf run_pc)
        return 0.0 if result else 1.0 
    
    def provide_with_cl_args(self, **args) -> Callable[[int, int, list[int]], float]:
        return self
    

def run_pc(data_format: IManageData, generalized_cit: abstract_cit_t, **args):
    # Note: alpha param of cl is unused (signigicance is decided by test, cf cit-wrapper)
    data_ignored_but_need_correct_shape_and_type = np.empty(shape=(data_format.total_sample_size(), data_format.number_of_variables()), dtype=np.void)
    result = _cl_impl_pc(data=data_ignored_but_need_correct_shape_and_type, indep_test=WrapCIT_CausalLearnIndexing(generalized_cit), show_progress=False, **args)
    return causallearn_graph_to_tigramite_graph(result)



def run_fci(data_format: IManageData, generalized_cit: abstract_cit_t, **args):
    # Note: alpha param of cl is unused (signigicance is decided by test, cf cit-wrapper)
    data_ignored_but_need_correct_shape_and_type = np.empty(shape=(data_format.total_sample_size(), data_format.number_of_variables()), dtype=np.void)
    result = _cl_impl_fci(dataset=data_ignored_but_need_correct_shape_and_type, independence_test_method=WrapCIT_CausalLearnIndexing(generalized_cit), show_progress=False, **args)
    return causallearn_graph_to_tigramite_graph(result)


def alg_pc(data_format: IManageData, **runtime_args) -> abstract_cd_t:
    """
    Get PC [SG91]_ implementation from causal learn.
    
    :param data_format: data manager
    :type data_format: IManageData
    :param runtime_args: forwarded to causal-learns run_pc (together with "stable=False")
    :return: PC as abstract CD-algorithm.
    :rtype: abstract_cd_t
    """
    return partial(run_pc, data_format=data_format, stable=False, **runtime_args)

def alg_pc_stable(data_format: IManageData, **runtime_args) -> abstract_cd_t:
    """
    Get PC-stable [CM+14]_ implementation from causal learn.
    
    :param data_format: data manager
    :type data_format: IManageData
    :param runtime_args: forwarded to causal-learns run_pc (together with "stable=True")
    :return: PC-stable as abstract CD-algorithm.
    :rtype: abstract_cd_t
    """
    return partial(run_pc, data_format=data_format, stable=True, **runtime_args)

def alg_fci(data_format: IManageData, **runtime_args) -> abstract_cd_t:
    """
    Get FCI [SGS01]_ implementation from causal learn.
    
    :param data_format: data manager
    :type data_format: IManageData
    :param runtime_args: forwarded to causal-learns run_fci
    :return: FCI as abstract CD-algorithm.
    :rtype: abstract_cd_t
    """
    return partial(run_fci, data_format=data_format, **runtime_args)
