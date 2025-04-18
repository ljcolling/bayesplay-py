from __future__ import annotations
import typing
from dataclasses import dataclass
from typing import overload

import bayesplay_py._lib as _lib

if typing.TYPE_CHECKING:
    from ._lib import PythonModel
    from .likelihood import Likelihood
    from .prior import Prior


@dataclass
class Evidence:
    evidence: float
    likelihood: Likelihood
    null_prior: Prior

    @overload
    def __truediv__(self, other: float | int) -> float: ...

    @overload
    def __truediv__(self, other: Evidence) -> float: ...

    def __truediv__(self, other: Evidence | float | int) -> float:
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
        self._likelihood: Likelihood = likelihood
        self._prior: Prior = prior
        self._posterior_obj = None

    @overload
    def __call__(self, x: float) -> float: ...
    @overload
    def __call__(self, x: list[float]) -> list[float]: ...

    def __call__(self, x: float | list[float]) -> float | list[float]:
        if self._posterior_obj is None:
            self._posterior_obj = _lib.init_posterior(
                self._likelihood._interface.model_dump(),
                self._prior._interface.model_dump(),
            )
        return self.function(x)

    @overload
    def integrate(self) -> float: ...

    @overload
    def integrate(self, lb: float | None = None, ub: float | None = None) -> float: ...

    def integrate(self, lb: float | None = None, ub: float | None = None) -> float:
        if self._posterior_obj is None:
            self._posterior_obj = _lib.init_posterior(
                self._likelihood._interface.model_dump(),
                self._prior._interface.model_dump(),
            )
        return self._posterior_obj.integrate(lb, ub)

    @overload
    def function(self, x: float) -> float: ...
    @overload
    def function(self, x: list[float]) -> list[float]: ...
    def function(self, x: float | list[float]) -> float | list[float]:
        if self._posterior_obj is None:
            self._posterior_obj = _lib.init_posterior(
                self._likelihood._interface.model_dump(),
                self._prior._interface.model_dump(),
            )
        if isinstance(x, list):
            return self._posterior_obj.function_vec(x)
        else:
            return self._posterior_obj.function(x)


class Model:
    def __init__(self, likelihood: Likelihood, prior: Prior):
        self._likelihood: Likelihood = likelihood
        self._prior: Prior = prior
        self._model_obj: PythonModel = None

    @property
    def likelihood(self) -> Likelihood:
        return self._likelihood

    @property
    def prior(self) -> Prior:
        return self._prior

    def integrate(self) -> Evidence:
        if self._model_obj is None:
            self._model_obj = _lib.init_model(
                self._likelihood._interface.model_dump(),
                self._prior._interface.model_dump(),
            )
        return Evidence(self._model_obj.integral(), self._likelihood, self._prior)

    @property
    def posterior(self) -> Posterior:
        return Posterior(self._likelihood, self._prior)
