from __future__ import annotations
import typing
from enum import Enum
from typing import overload

from pydantic import BaseModel

import bayesplay_py._lib as _lib
from .common import Definition, Param, ParamList
from .model import Model

if typing.TYPE_CHECKING:
    from .prior import PriorDefinition


class LikelihoodFamily(str, Enum):
    normal = "normal"
    noncentral_d = "noncentral_d"


class LikelihoodInterface(BaseModel):
    family: LikelihoodFamily
    params: ParamList


class Likelihood(Definition):
    _family: LikelihoodFamily
    _params: ParamList
    _object: _lib.Likelihood | None
    _interface: LikelihoodInterface
    _initialisation_func = _lib.init_likelihood

    def __init__(self, **kwargs):
        self._family = kwargs.pop("family")

        params = {n: v for n, v in kwargs.items()}
        self._params = ParamList(
            [Param(name=name, value=value) for name, value in params.items()]
        )

        self._object = None
        self._interface = LikelihoodInterface(family=self._family, params=self._params)

    @staticmethod
    def normal(mean: float, se: float) -> Likelihood:
        return Likelihood(family="normal", mean=mean, sd=se)

    @staticmethod
    def noncentral_d(d: float, n: float) -> Likelihood:
        return Likelihood(family="noncentral_d", d=d, n=n)

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
        super().initialise_object()
        return self.function(x)

    @overload
    def function(self, x: float) -> float: ...
    @overload
    def function(self, x: list[float]) -> list[float]: ...
    def function(self, x: float | list[float]) -> float | list[float]:
        super().initialise_object()
        if isinstance(x, list):
            return self._object.function_vec(x)
        else:
            return self._object.function(x)

    def __mul__(self, other: PriorDefinition):
        return Model(self, other)
