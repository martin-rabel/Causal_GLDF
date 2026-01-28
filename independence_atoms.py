from .hccd import IProvideIndependenceAtoms, IHandleExplicitTransitionToMCI
from .data_processing import ITestMarkedCI, ITestIndicatorImplications
from .data_management import CI_Identifier, CI_Identifier_TimeSeries, IManageData

class IndependenceAtoms_Backend(IProvideIndependenceAtoms):
    """
        Integrate data-provider with mCIT and implication-testing into a independence-atom backend
        for use in :py:mod:`hccd<GLDF.hccd>`.
    """
    
    def __init__(self, data_manager: IManageData, m_cit: ITestMarkedCI, implication_test: ITestIndicatorImplications=None):
        """Consstruct from data-manager, mCIT and implication-test.

        :param data_manager: data-manager
        :type data_manager: IManageData
        :param m_cit: mCIT
        :type m_cit: ITestMarkedCI
        :param implication_test: indicator implication test, defaults to None
        :type implication_test: ITestIndicatorImplications, optional
        """
        self.data_manager = data_manager
        self.m_cit = m_cit
        self.implication_test = implication_test
        self._found_globally_independent_for_some_Z = set()

    def marked_independence(self, ci: CI_Identifier) -> ITestMarkedCI.Result:
        result = self.m_cit.marked_independence(self.data_manager.get_patterned_data(ci))
        if result.is_globally_independent():
            self._found_globally_independent_for_some_Z.add( ci.undirected_link() )
        return result
    
    def regime_implication(self, lhs: list[CI_Identifier], rhs: CI_Identifier) -> bool:
        return self.implication_test.is_implied_regime(
            [self.data_manager.get_patterned_data(ci) for ci in lhs],
            self.data_manager.get_patterned_data(rhs)
        )
    
    def found_globally_independent_for_some_Z(self, undirected_link: tuple) -> bool:
        return undirected_link in self._found_globally_independent_for_some_Z

class IndependenceAtoms_TimeSeries(IProvideIndependenceAtoms, IHandleExplicitTransitionToMCI):
    """Dual-phase/transitionable backend to account for different configurations in PC1 and MCI
    phases of PCMCI-family algorithms.
    """
    def __init__(self, independence_atoms_pc1: IProvideIndependenceAtoms, independence_atoms_mci: IProvideIndependenceAtoms):
        """Construct from two separate backends for PC1 and MCI phases.

        :param independence_atoms_pc1: backend for PC1 phase
        :type independence_atoms_pc1: IProvideIndependenceAtoms
        :param independence_atoms_mci: backend for MCI phase
        :type independence_atoms_mci: IProvideIndependenceAtoms
        """
        self.independence_atoms_pc1 = independence_atoms_pc1
        self.independence_atoms_mci = independence_atoms_mci
        self._active_backend = independence_atoms_pc1

    def marked_independence(self, ci: CI_Identifier_TimeSeries) -> ITestMarkedCI.Result:
        return self._active_backend.marked_independence(ci=ci)
    
    def regime_implication(self, lhs: list[CI_Identifier_TimeSeries], rhs: CI_Identifier_TimeSeries) -> bool:
        return self._active_backend.regime_implication(lhs, rhs)
    
    def found_globally_independent_for_some_Z(self, undirected_link: tuple) -> bool:
        return ( self.independence_atoms_mci.found_globally_independent_for_some_Z(undirected_link)
             or self.independence_atoms_pc1.found_globally_independent_for_some_Z(undirected_link) )
    
    def enter_pc1(self) -> None:
        self._active_backend = self.independence_atoms_pc1

    def enter_mci(self) -> None:
        self._active_backend = self.independence_atoms_mci

    def _extract_cache_id(self, fname: str, **args) -> tuple:
        # In principle can also cache the multi-mode backend if tracking
        # state (active backend). Usually not necessary, if indiviudal
        # backends are cached (cf. frontend).
        return (self._active_backend, self._active_backend._extract_cache_id(fname, **args))
    