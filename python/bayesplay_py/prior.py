from __future__ import annotations
import typing
from enum import Enum
from typing import overload

from pydantic import BaseModel

import bayesplay_py._lib as _lib
from .common import Definition, Param, ParamList

if typing.TYPE_CHECKING:
    from .likelihood import LikelihoodDefinition
from .model import Model


class PriorFamily(str, Enum):
    normal = "normal"
    cauchy = "cauchy"
    point = "point"


class PriorInterface(BaseModel):
    family: PriorFamily
    params: ParamList


class Prior(Definition):
    _family: PriorFamily
    _params: ParamList
    _object: _lib.Prior | None
    _interface: PriorInterface
    _initialisation_func = _lib.init_prior

    def __init__(self, **kwargs: float | None):
        self._family = kwargs.pop("family")

        params = {n: v for n, v in kwargs.items() if v is not None}
        self._params = ParamList(
            [
                Param(name=name, value=value)
                for name, value in params.items()
                if value is not None
            ]
        )

        self._obj = None
        self._interface = PriorInterface(family=self._family, params=self._params)

    @staticmethod
    def normal(
        mean: float, se: float, ll: float | None = None, ul: float | None = None
    ) -> Prior:
        return Prior(family="normal", mean=mean, sd=se, ll=ll, ul=ul)

    @staticmethod
    def cauchy(
        location: float, scale: float, ll: float | None = None, ul: float | None = None
    ) -> Prior:
        return Prior(family="cauchy", location=location, scale=scale, ll=ll, ul=ul)

    @staticmethod
    def point(point: float) -> Prior:
        return Prior(family="point", point=point)

    @overload
    def __call__(self, x: float) -> float: ...
    @overload
    def __call__(self, x: list[float]) -> list[float]: ...
    def __call__(self, x: float | list[float]) -> float | list[float]:
        if self._prior_obj is None:
            self._prior_obj = _lib.init_prior(self._prior_interface.model_dump())
        return self.function(x)

    def integrate(self, lb: float | None = None, ub: float | None = None) -> float:
        if self._prior_obj is None:
            self._prior_obj = _lib.init_prior(self._prior_interface.model_dump())
        return self._prior_obj.integrate(lb, ub)

    @overload
    def function(self, x: float) -> float: ...
    @overload
    def function(self, x: list[float]) -> list[float]: ...
    def function(self, x: float | list[float]) -> float | list[float]:
        if self._prior_obj is None:
            self._prior_obj = _lib.init_prior(self._prior_interface.model_dump())
        if isinstance(x, list):
            return self._prior_obj.function_vec(x)
        else:
            return self._prior_obj.function(x)

    def __mul__(self, other: LikelihoodDefinition):
        return Model(other, self)

    @property
    def ll(self) -> float:
        return self._params.get("ll")

    @property
    def ul(self) -> float:
        return self._params.get("ul")
