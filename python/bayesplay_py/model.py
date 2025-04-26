from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union, overload

import bayesplay_py._lib as _lib

if TYPE_CHECKING:
    from ._lib import PythonModel, PythonPosterior
    from .likelihood import Likelihood
    from .prior import Prior


@dataclass
class Evidence:
    evidence: float
    likelihood: Likelihood
    null_prior: Prior

    @overload
    def __truediv__(self, other: Union[float, int]) -> float: ...

    @overload
    def __truediv__(self, other: Evidence) -> float: ...

    def __truediv__(self, other: Union[Evidence, float, int]) -> float:
        if isinstance(other, Evidence):
            if other.likelihood != self.likelihood:
                raise ValueError(
                    "Likelihoods must be the same to divide Evidence objects."
                )
            return self.evidence / other.evidence
        if other == 1:
            return self.evidence / other
        raise ValueError("Divide by 1 to invert the model comparison.")

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.evidence)


class Posterior:
    def __init__(self, likelihood: Likelihood, prior: Prior):
        self._likelihood = likelihood
        self._prior = prior
        self._posterior_obj: PythonPosterior = _lib.init_posterior(
            self._likelihood._interface.model_dump(),
            self._prior._interface.model_dump(),
        )

    @overload
    def __call__(self, x: float) -> float: ...
    @overload
    def __call__(self, x: List[float]) -> List[float]: ...

    def __call__(self, x: Union[float, List[float]]) -> Union[float, List[float]]:
        return self.function(x)

    @overload
    def integrate(self) -> float: ...

    @overload
    def integrate(
        self, lb: Optional[float] = None, ub: Optional[float] = None
    ) -> float: ...

    def integrate(
        self, lb: Optional[float] = None, ub: Optional[float] = None
    ) -> float:
        return self._posterior_obj.integrate(lb, ub)

    @overload
    def function(self, x: float) -> float: ...
    @overload
    def function(self, x: List[float]) -> List[float]: ...
    def function(self, x: Union[float, List[float]]) -> Union[float, List[float]]:
        if isinstance(x, list):
            return self._posterior_obj.function_vec(x)
        else:
            return self._posterior_obj.function(x)


class Model:
    def __init__(self, likelihood: Likelihood, prior: Prior):
        self._likelihood = likelihood
        self._prior = prior
        self._model_obj: PythonModel = _lib.init_model(
            self._likelihood._interface.model_dump(),
            self._prior._interface.model_dump(),
        )

    @property
    def likelihood(self) -> Likelihood:
        return self._likelihood

    @property
    def prior(self) -> Prior:
        return self._prior

    def integrate(self) -> Evidence:
        return Evidence(self._model_obj.integral(), self._likelihood, self._prior)

    @property
    def posterior(self) -> Posterior:
        return Posterior(self._likelihood, self._prior)
