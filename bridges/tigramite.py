try:
    from tigramite.pcmci import PCMCI
    from tigramite.lpcmci import LPCMCI
except ModuleNotFoundError:
    raise RuntimeError("Tigramite bridge could not be loaded. Is tigramite installed?")

import numpy as np
from typing import Literal
np.fastCopyAndTranspose = lambda a: a.T.copy() # this was deprecated since 1.24.0, removed in 2.0.0, but is used in some places in tigramite

from ..hccd import IHandleExplicitTransitionToMCI, CI_Identifier_TimeSeries, abstract_cit_t, abstract_cd_t
from ..data_management import IManageData

class WrapCIT_TigramiteIndexing:
    def __init__(self, generalized_cit: abstract_cit_t):
        self.generalized_cit = generalized_cit
        self.method = "custom"
        self.measure = "custom" # used only on higher verbosity of PCMCI(?)
        self.confidence = None
        self.significance = "custom"
    def run_test(self, X: list[tuple[int,int]], Y: list[tuple[int,int]], Z: list[tuple[int,int]], tau_max: int, alpha_or_thres: float) -> tuple[float, float, bool]:
        if len(X) != 1 or len(Y) != 1:
            raise NotImplementedError("Currently this implementation supports only univariate X, Y.")
        ci = CI_Identifier_TimeSeries(idx_x=X[0], idx_y=Y[0], idx_list_z=Z)
        result = self.generalized_cit(ci)
        # return dependency-score, pvalue, dependent (yes/no)
        return 1.0 if result else 0.0, 0.0 if result else 1.0, result
    def get_confidence(*args, **args_dict):
        return None # run_mci with val_only=True does not seem to work

class WrapPCMCI(PCMCI):
    def __init__(self, data_format: IManageData, mode: Literal["PCMCI", "PCMCI+"]="PCMCI", mci_transition_callback: IHandleExplicitTransitionToMCI=None, pcmci_obj_init_args: dict={}, pcmci_obj_run_args: dict={}):
        class PlaceholderTest:
            def set_dataframe(self, not_a_dataframe):
                pass
        class PlaceholderDataframe:
            def __init__(self):
                self.var_names = None
                self.T = None
                self.N = None

        super().__init__(dataframe=PlaceholderDataframe(), cond_ind_test=PlaceholderTest(), **pcmci_obj_init_args)
        self.data_format = data_format
        self.mci_transition_callback = mci_transition_callback
        self.runtime_args = pcmci_obj_run_args
        if mode == "PCMCI":
            self.run = self.run_pcmci
        elif mode == "PCMCIplus" or mode == "PCMCI+":
            self.run = self.run_pcmciplus
        else:
            raise ValueError("Unknown mode for PCMCI, supported values are 'PCMCI' or 'PCMCI+'. Did you want to use WrapLPCMCI instead?")

    def run_pc_stable(self, *args_tuple, **args_dict):
        self.mci_transition_callback.enter_pc1()
        return super().run_pc_stable(*args_tuple, **args_dict)
    def run_mci(self, *args_tuple, **args_dict):
        self.mci_transition_callback.enter_mci()
        return super().run_mci(*args_tuple, **args_dict)
    def _pcmciplus_mci_skeleton_phase(self, *args_tuple, **args_dict):
        self.mci_transition_callback.enter_mci()
        return super()._pcmciplus_mci_skeleton_phase(*args_tuple, **args_dict)

    def __call__(self, generalized_cit: abstract_cit_t):
        self.T, self.N = {0: self.data_format.total_sample_size()}, self.data_format.number_of_variables()
        self.cond_ind_test = WrapCIT_TigramiteIndexing(generalized_cit)
        result = self.run(**self.runtime_args)
        return result['graph']


class WrapLPCMCI(LPCMCI):
    def __init__(self, data_format: IManageData, lpcmci_obj_init_args: dict={}, lpcmci_obj_run_args: dict={}):
        class PlaceholderTest:
            def set_dataframe(self, not_a_dataframe):
                pass
        class PlaceholderDataframe:
            def __init__(self):
                self.var_names = None
                self.T = None
                self.N = None
        super().__init__(dataframe=PlaceholderDataframe(), cond_ind_test=PlaceholderTest(), **lpcmci_obj_init_args)
        self.data_format = data_format
        self.runtime_args = lpcmci_obj_run_args

    def __call__(self, generalized_cit: abstract_cit_t):
        self.T, self.N = {0: self.data_format.total_sample_size()}, self.data_format.number_of_variables()
        self.cond_ind_test = WrapCIT_TigramiteIndexing(generalized_cit)
        result = self.run_lpcmci(**self.runtime_args)
        return result['graph']


def alg_pcmci(data_format: IManageData, mci_transition_callback: IHandleExplicitTransitionToMCI=None, pcmci_obj_init_args: dict={}, pcmci_obj_run_args: dict={}) -> abstract_cd_t:
    """
    Get PCMCI [RNK+19]_ implementation from tigramite. Use with :py:class:`ControllerTimeseries<GLDF.hccd.ControllerTimeseries>`.

    :param data_format: data manager
    :type data_format: IManageData
    :param mci_transition_callback: transition callback (typically :py:class:`IndependenceAtoms_TimeSeries<GLDF.independence_atoms.IndependenceAtoms_TimeSeries>`)
    :type mci_transition_callback: IHandleExplicitTransitionToMCI
    :param pcmci_obj_init_args: forwarded to tigramites PCMCI constructor
    :type pcmci_obj_init_args: dict
    :param pcmci_obj_run_args: forwarded to tigramites PCMCI.run_pcmci
    :type pcmci_obj_run_args: dict
    :return: PCMCI as abstract CD-algorithm.
    :rtype: abstract_cd_t
    """
    return WrapPCMCI(data_format=data_format, mode="PCMCI", mci_transition_callback=mci_transition_callback,
                     pcmci_obj_init_args=pcmci_obj_init_args, pcmci_obj_run_args=pcmci_obj_run_args)

def alg_pcmciplus(data_format: IManageData, mci_transition_callback: IHandleExplicitTransitionToMCI=None, pcmci_obj_init_args: dict={}, pcmci_obj_run_args: dict={}) -> abstract_cd_t:
    """
    Get PCMCI+ [R20]_ implementation from tigramite. Use with :py:class:`ControllerTimeseries<GLDF.hccd.ControllerTimeseries>`.

    :param data_format: data manager
    :type data_format: IManageData
    :param mci_transition_callback: transition callback (typically :py:class:`IndependenceAtoms_TimeSeries<GLDF.independence_atoms.IndependenceAtoms_TimeSeries>`)
    :type mci_transition_callback: IHandleExplicitTransitionToMCI
    :param pcmci_obj_init_args: forwarded to tigramites PCMCI constructor
    :type pcmci_obj_init_args: dict
    :param pcmci_obj_run_args: forwarded to tigramites PCMCI.run_pcmciplus
    :type pcmci_obj_run_args: dict
    :return: PCMCI+ as abstract CD-algorithm.
    :rtype: abstract_cd_t
    """
    return WrapPCMCI(data_format=data_format, mode="PCMCI+", mci_transition_callback=mci_transition_callback,
                     pcmci_obj_init_args=pcmci_obj_init_args, pcmci_obj_run_args=pcmci_obj_run_args)

def alg_lpcmci(data_format: IManageData, lpcmci_obj_init_args: dict={}, lpcmci_obj_run_args: dict={}) -> abstract_cd_t:
    """
    Get LPCMCI [GR20]_ implementation from tigramite. Use with :py:class:`ControllerTimeseriesLPCMCI<GLDF.hccd.ControllerTimeseriesLPCMCI>`.

    *Note: In this case a transition callback* :py:class:`IHandleExplicitTransitionToMCI<GLDF.hccd.IHandleExplicitTransitionToMCI>` *is*
    *notified by the controller* :py:class:`ControllerTimeseriesLPCMCI<GLDF.hccd.ControllerTimeseriesLPCMCI>`\\ *.*

    :param data_format: data manager
    :type data_format: IManageData
    :param lpcmci_obj_init_args: forwarded to tigramites LPCMCI constructor
    :type lpcmci_obj_init_args: dict
    :param lpcmci_obj_run_args: forwarded to tigramites LPCMCI.run_lpcmci
    :type lpcmci_obj_run_args: dict
    :return: LPCMCI as abstract CD-algorithm.
    :rtype: abstract_cd_t
    """
    return WrapLPCMCI(data_format=data_format, lpcmci_obj_init_args=lpcmci_obj_init_args, lpcmci_obj_run_args=lpcmci_obj_run_args)
