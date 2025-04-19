from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, overload

from pydantic import BaseModel

import bayesplay_py._lib as _lib
from .common import Param, ParamList
from .model import Model

if TYPE_CHECKING:
    from bayesplay_py._lib import PythonLikelihood
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
    _family: LikelihoodFamily
    _params: ParamList
    _object: PythonLikelihood
    _interface: LikelihoodInterface
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

        self._object = None
        self._interface: LikelihoodInterface = LikelihoodInterface(
            family=self._family, params=self._params
        )

    def initialise_object(self):
        if self._object is None:
            model_dump: dict[str, Any] = self._interface.model_dump()
            self._object: PythonLikelihood = self._initialisation_func(model_dump)

    @staticmethod
    def normal(mean: float, se: float) -> Likelihood:
        return Likelihood(family=LikelihoodFamily.normal, mean=mean, sd=se)

    @staticmethod
    def noncentral_d(d: float, n: float) -> Likelihood:
        return Likelihood(family=LikelihoodFamily.noncentral_d, d=d, n=n)

    @staticmethod
    def noncentral_d2(d: float, n1: float, n2: float) -> Likelihood:
        return Likelihood(family=LikelihoodFamily.noncentral_d2, d=d, n1=n1, n2=n2)

    @staticmethod
    def noncentral_t(t: float, df: float) -> Likelihood:
        return Likelihood(family=LikelihoodFamily.noncentral_t, t=t, df=df)

    @staticmethod
    def binomial(successes: float, trials: float) -> Likelihood:
        return Likelihood(
            family=LikelihoodFamily.binomial, successes=successes, trials=trials
        )

    @staticmethod
    def student_t(mean: float, sd: float, df: float) -> Likelihood:
        return Likelihood(family=LikelihoodFamily.student_t, mean=mean, sd=sd, df=df)

    @overload
    def __call__(self, x: float) -> float:
        """
        Evaluate the likelihood function at the value `x`
        """

    @overload
    def __call__(self, x: list[float]) -> list[float]:
        """
        Evaluate the likelihood function at the values `x`
        """

    def __call__(self, x: float | list[float]) -> float | list[float]:
        self.initialise_object()
        return self.function(x)

    @overload
    def function(self, x: float) -> float: ...
    @overload
    def function(self, x: list[float]) -> list[float]: ...
    def function(self, x: float | list[float]) -> float | list[float]:
        self.initialise_object()
        if isinstance(x, list):
            return self._object.function_vec(x)
        else:
            return self._object.function(x)

    def __mul__(self, other: Prior):
        return Model(self, other)
