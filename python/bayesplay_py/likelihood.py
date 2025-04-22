from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, overload

from pydantic import BaseModel

import bayesplay_py._lib as _lib
from .common import Param, ParamList

if TYPE_CHECKING:
    from ._lib import PythonLikelihood
    from .prior import Prior


class LikelihoodFamily(str, Enum):
    normal = "normal"
    noncentral_d = "noncentral_d"
    noncentral_d2 = "noncentral_d2"
    noncentral_t = "noncentral_t"
    student_t = "student_t"
    binomial = "binomial"


class LikelihoodInterface(BaseModel):
    family: LikelihoodFamily
    params: ParamList


class Likelihood:
    _initialisation_func: Callable[[dict[str, Any]], PythonLikelihood] = (
        _lib.init_likelihood
    )

    def __init__(self, family: LikelihoodFamily, **kwargs: float | None):
        self._family = family

        params: dict[str, float] = {
            name: value for name, value in kwargs.items() if value is not None
        }
        self._params = ParamList(
            [Param(name=name, value=value) for name, value in params.items()]
        )

        self._interface: LikelihoodInterface = LikelihoodInterface(
            family=self._family, params=self._params
        )
        model_dump: dict[str, Any] = self._interface.model_dump()
        self._object: PythonLikelihood = self._initialisation_func(model_dump)

    @classmethod
    def normal(cls, mean: float, se: float):
        return cls(family=LikelihoodFamily.normal, mean=mean, se=se)

    @classmethod
    def noncentral_d(cls, d: float, n: float):
        return cls(family=LikelihoodFamily.noncentral_d, d=d, n=n)

    @classmethod
    def noncentral_d2(cls, d: float, n1: float, n2: float):
        return cls(family=LikelihoodFamily.noncentral_d2, d=d, n1=n1, n2=n2)

    @classmethod
    def noncentral_t(cls, t: float, df: float):
        return cls(family=LikelihoodFamily.noncentral_t, t=t, df=df)

    @classmethod
    def binomial(cls, successes: float, trials: float):
        return cls(family=LikelihoodFamily.binomial, successes=successes, trials=trials)

    @classmethod
    def student_t(cls, mean: float, sd: float, df: float):
        return cls(family=LikelihoodFamily.student_t, mean=mean, sd=sd, df=df)

    @overload
    def __call__(self, x: float) -> float: ...

    @overload
    def __call__(self, x: list[float]) -> list[float]: ...

    def __call__(self, x: float | list[float]) -> float | list[float]:
        return self.function(x)

    @overload
    def function(self, x: float) -> float: ...
    @overload
    def function(self, x: list[float]) -> list[float]: ...
    def function(self, x: float | list[float]) -> float | list[float]:
        if isinstance(x, list):
            return self._object.function_vec(x)
        else:
            return self._object.function(x)

    def __mul__(self, other: Prior):
        from .model import Model

        return Model(self, other)
